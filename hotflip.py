import torch
import time
import random
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE IS", device)

# Function to disable gradient computation
def set_no_grad(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

# Mean pooling function
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

# Function to compute average cosine similarity
def compute_average_cosine_similarity(embeddings):
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    cos_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    mask = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device).bool()
    cos_sim_matrix.masked_fill_(mask, 0)
    avg_cos_sim = cos_sim_matrix.sum() / (cos_sim_matrix.numel() - cos_sim_matrix.size(0))
    print("document cos sim", avg_cos_sim.item(), 'min', cos_sim_matrix.min().item())
    return avg_cos_sim

class HotFlip:
    def __init__(self):
        # Initialize the encoder and tokenizer
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.encoder = AutoModel.from_pretrained('facebook/contriever').to(device)
        self.encoder_word_embedding = self.encoder.get_input_embeddings().weight.detach()
        set_no_grad(self.encoder)

    # Function to compute document embeddings
    def compute_doc_embs(self, documents):
        doc_inputs = self.encoder_tokenizer(documents, padding=True, truncation=True, return_tensors='pt').to(device)
        doc_embs = mean_pooling(self.encoder(**doc_inputs)[0], doc_inputs['attention_mask'])
        return doc_embs

    # Function to compute gradient for hotflip
    def compute_hotflip_gradient(self, inputs_embeds_batch, doc_embs):
        inputs_embeds = torch.nn.Parameter(inputs_embeds_batch, requires_grad=True)
        s_adv_emb = self.encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
        cos_sim = torch.matmul(normalize(s_adv_emb, p=2, dim=1), normalize(doc_embs, p=2, dim=1).t()).mean()
        loss = cos_sim
        loss.backward()
        return inputs_embeds.grad.detach()

    # Function to compute the score (cosine similarity) of a sequence
    def compute_sequence_score(self, sequence, doc_embs):
        vocab_size = self.encoder_word_embedding.size(0)
        onehot = torch.nn.functional.one_hot(torch.tensor([sequence], device=device), vocab_size).float()
        inputs_embeds = torch.matmul(onehot, self.encoder_word_embedding)
        s_adv_emb = self.encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
        score = torch.matmul(normalize(s_adv_emb, p=2, dim=1), normalize(doc_embs, p=2, dim=1).t()).mean().detach().cpu().numpy()
        return score

    def compute_sequence_score_batch(self, sequence_batch, doc_embs):
        vocab_size = self.encoder_word_embedding.size(0)
        onehot = torch.nn.functional.one_hot(torch.tensor(sequence_batch, device=device), vocab_size).float()
        inputs_embeds = torch.matmul(onehot, self.encoder_word_embedding)
        s_adv_emb = self.encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
        batch_score = torch.matmul(normalize(s_adv_emb, p=2, dim=1), normalize(doc_embs, p=2, dim=1).t()).mean(dim=1).detach().cpu().numpy()
        return batch_score

    # Function to compute gradients for a sequence
    def compute_gradients(self, sequence, doc_embs):
        vocab_size = self.encoder_word_embedding.size(0)
        onehot = torch.nn.functional.one_hot(torch.tensor([sequence], device=device), vocab_size).float()
        inputs_embeds = torch.matmul(onehot, self.encoder_word_embedding)
        inputs_embeds = torch.nn.Parameter(inputs_embeds, requires_grad=True)
        s_adv_emb = self.encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
        cos_sim = torch.matmul(normalize(s_adv_emb, p=2, dim=1), normalize(doc_embs, p=2, dim=1).t()).mean()
        loss = cos_sim
        loss.backward()
        gradients = inputs_embeds.grad.detach()
        return gradients[0]  # Since batch size is 1

    # Modified hotflip attack function using beam search
    def optimize(self, documents, trigger, start_tokens=None, num_tokens=32, epoch_num=100, beam_width=20, top_k_tokens=5):
        trig_doc_embs = self.compute_doc_embs(documents)
        compute_average_cosine_similarity(trig_doc_embs)

        if start_tokens is None:
            start_tokens = [0] * num_tokens
        vocab_size = self.encoder_word_embedding.size(0)
        
        # Initialize beam with the initial sequence and its score
        initial_score = self.compute_sequence_score(start_tokens, trig_doc_embs)
        beam = [(start_tokens, initial_score)]
        # print(f"Initial sequence score: {initial_score}")

        for epoch in tqdm(range(epoch_num)):
            all_candidates = []

            seq_batch = []
            for seq, score in beam:
                # Compute gradients for the current sequence
                gradients = self.compute_gradients(seq, trig_doc_embs)
                positions = list(range(len(seq)))  # Positions to modify

                for pos in positions:
                    # Get gradient at position 'pos'
                    grad_at_pos = gradients[pos]
                    # Compute dot product with embeddings
                    emb_grad_dotprod = torch.matmul(grad_at_pos, self.encoder_word_embedding.t())
                    # Get top_k_tokens tokens with highest dot product
                    topk = torch.topk(emb_grad_dotprod, top_k_tokens)
                    topk_tokens = topk.indices.tolist()

                    for token in topk_tokens:
                        new_seq = seq.copy()
                        new_seq[pos] = token
                        # Compute score of new_seq
                        seq_batch.append(new_seq)
            score_batch = self.compute_sequence_score_batch(seq_batch, trig_doc_embs)
            for seq, score in zip(seq_batch, score_batch):
                all_candidates.append((seq, score))

            # Sort all_candidates by score in descending order and keep top beam_width sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[:beam_width]

            # Optionally, print the best sequence and score
            best_seq, best_score = beam[0]
            # print(f"Best sequence at epoch {epoch}: {self.encoder_tokenizer.decode(best_seq)} with score {best_score}")

        # Return the best sequence
        best_seq, best_score = beam[0]
        print("best score", best_score)
        return self.encoder_tokenizer.decode(best_seq)
