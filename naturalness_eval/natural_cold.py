from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BertModel, T5ForConditionalGeneration, AutoModelForMaskedLM, AutoModelForCausalLM, LlamaModel, BertForSequenceClassification
import torch, time
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from bert_models import BertForLM

def set_no_grad(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def score_fn(cos_sim, naturalness, perplexity):
    # return - perplexity
    perplexity = torch.clamp(perplexity, min=5)
    return naturalness - 0.1 * perplexity

class NaturalCOLD:
    def __init__(self):
        device = torch.get_default_device()
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.encoder: BertModel = AutoModel.from_pretrained('facebook/contriever').to(device)
        self.naturalness_llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        self.naturalness_llm_tokenizer.pad_token = self.naturalness_llm_tokenizer.eos_token
        self.naturalness_llm = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct').to(device)
        set_no_grad(self.encoder)
        self.causal_llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.causal_llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        set_no_grad(self.causal_llm)
        self.mask_llm_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.mask_llm = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
        set_no_grad(self.mask_llm)
        self.naturalness_eval_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.naturalness_eval = BertForSequenceClassification.from_pretrained('./models/naturalness_model')
        set_no_grad(self.naturalness_eval)
        self.encoder_word_embedding = self.encoder.get_input_embeddings().weight.detach()
        self.encoder_vocab_size = self.encoder_word_embedding.shape[0]
        self.causal_llm_tokenizer.pad_token = self.causal_llm_tokenizer.eos_token
        self.EOS_TOKEN = self.causal_llm_tokenizer.encode(self.causal_llm_tokenizer.pad_token)[0]
        self.bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.bert_lm = BertForLM.from_pretrained('/share/shmatikov/collin/constrained_rag_attack/collision/wiki103/bert').to(device)
        set_no_grad(self.bert_lm)
    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def compute_doc_embs(self, documents):
        doc_inputs = self.encoder_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        doc_embs = self.mean_pooling(self.encoder(**doc_inputs)[0], doc_inputs['attention_mask'])
        return doc_embs

    def compute_average_cosine_similarity(self, embeddings):
        # Normalize the embeddings to have unit norm
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute the cosine similarity matrix
        cos_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Mask out the diagonal elements (self-similarity) and compute the average of the off-diagonal elements
        mask = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device).bool()
        cos_sim_matrix.masked_fill_(mask, 0)
        
        # Compute the average cosine similarity (excluding self-similarity)
        avg_cos_sim = cos_sim_matrix.sum() / (cos_sim_matrix.numel() - cos_sim_matrix.size(0))

        print("document cos sim", avg_cos_sim, 'min', cos_sim_matrix.min())
        
        return avg_cos_sim

    def compute_cos_sim(self, s_adv, doc_embs, stemp):
        ## compute loss
        _, inputs_embeds = self.relaxed_to_word_embs(s_adv, stemp)
        s_adv_emb = self.encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
        cos_sim = torch.nn.functional.cosine_similarity(s_adv_emb, doc_embs, dim=1)
        return cos_sim.mean()
    
    def compute_naturalness(self, s_adv, stemp):
        _, inputs_embeds = self.relaxed_to_word_embs(s_adv, stemp)
        outputs = self.naturalness_eval(inputs_embeds=inputs_embeds)
        # 1 for naturalness
        print(outputs.logits)
        # return torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]
        return torch.nn.functional.sigmoid(outputs.logits[:, 1])

    def compute_soft_perplexity(self, s_adv_list, stemp):
        s_adv_list = torch.stack(s_adv_list)
        inputs_embeds_list = torch.stack([self.relaxed_to_word_embs(s_adv, stemp)[1][0] for s_adv in s_adv_list])
        # print("inputs embeds shape", inputs_embeds.shape)
        # print("test shape", self.bert_lm(inputs_embeds=inputs_embeds)[0].shape)
        lm_logits = self.bert_lm(inputs_embeds=inputs_embeds_list)[0]
        # print("lm logits shape", lm_logits.shape)
        shift_logits = lm_logits[:, 1:-1, :].contiguous()
        shift_labels = s_adv_list[:, 1:, :].contiguous()
        # print("shift logits shape", shift_logits.shape)
        # print("shift labels shape", shift_labels.shape)
        # print(shift_logits)
        # print(shift_labels)
        probs_pred = torch.nn.functional.softmax(shift_logits, dim=1)
        probs_true = torch.nn.functional.softmax(shift_labels, dim=1)
        # print(probs_pred)
        # print(probs_true)
        log_probs_pred = torch.log(probs_pred)
        cross_entropy = -torch.sum(probs_true * log_probs_pred, dim=1)
        loss = cross_entropy.mean(dim=-1)
        # print("loss", loss)
        return loss

    def relaxed_to_word_embs(self, x, stemp):
        # convert relaxed inputs to word embedding by softmax attention
        p = torch.softmax(x / stemp, -1)
        x = torch.mm(p, self.encoder_word_embedding)
        # add embeddings for period and SEP
        # x = torch.cat([x, word_embedding[encoder_tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def tokens_to_embs(self, tokens):
        embs = []
        for token in tokens:
            embs.append(self.encoder_word_embedding[token])
        return torch.stack(embs)

    def generate_start_tokens(self, trigger):
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f'Write a 32 tokens sentence about {trigger}.'},
        ],
        max_tokens=32
        )
        # print(text)
        res: str = response.choices[0].message.content # type: ignore
        print("start tokens from gpt4o", res)
        return self.encoder_tokenizer.encode(res)

    def optimize(self, lr=1, topk=10, beam_width=10, eps=0.1, stemp=0.5, epoch_num=3, perturb_iter=10): 
        print("stemp", stemp)
        @dataclass
        class BeamCandidate:
            sequence: ... = field(default_factory=list)
            score: ... = None
            trig_cos_sim: ... = None
            naturalness: ... = None
            perplexity: ... = None

        def beam_search(z_i, num_tokens, topk, beam_width, stemp):
            print("start beam search", flush=True)
            start_t = time.time()
            _, inputs_embeds = self.relaxed_to_word_embs(z_i, stemp)
            # s_adv_emb = encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
            
            # Initialize the beams with empty sequences and their scores
            beams = [BeamCandidate()]  # List of tuples (token_sequence, similarity_score)
            
            for i in tqdm(range(num_tokens)):
                all_candidates = []
                batch_embs = []
                batch_sequences = []
                for candidate in beams:
                    prev_tokens = candidate.sequence
                    _, topk_tokens = torch.topk(z_i[i], topk)
                    for token in topk_tokens:
                        if prev_tokens.count(token) >= 2:
                            continue
                        new_sequence = prev_tokens + [token]
                        current_word_embs = torch.cat((self.tokens_to_embs(new_sequence), inputs_embeds[0][len(new_sequence):]), dim=0).unsqueeze(0)
                        batch_embs.append(current_word_embs)
                        batch_sequences.append(BeamCandidate(sequence=new_sequence))
                # Process all candidate embeddings in parallel
                batch_embs = torch.cat(batch_embs, dim=0)
                batch_embs_output = self.encoder(inputs_embeds=batch_embs)[0].mean(dim=1)
                # batch_naturalness = torch.nn.functional.softmax(self.naturalness_eval(inputs_embeds=batch_embs).logits, dim=-1)[:, 1]
                batch_naturalness = torch.nn.functional.sigmoid(self.naturalness_eval(inputs_embeds=batch_embs).logits[:, 1])
                batch_perplexities = self.compute_soft_perplexity([torch.cat([label_smoothing(cand.sequence), z_i[len(cand.sequence):]]) for cand in batch_sequences], stemp)
                for j, candidate in enumerate(batch_sequences):
                    new_sequence = candidate.sequence
                    current_naturalness = batch_naturalness[j]
                    current_perplexity = batch_perplexities[j]
                    # all_candidates.append(BeamCandidate(new_sequence, 20 * current_trig_sim + 0.01 * current_naturalness - 0.01 * current_perplexity, current_trig_sim, current_naturalness, current_perplexity))
                    all_candidates.append(BeamCandidate(new_sequence, score_fn(0, current_naturalness, current_perplexity), 0, current_naturalness, current_perplexity))
                # Sort all candidates by their similarity scores in descending order
                all_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
                beams = all_candidates[:beam_width]
            end_t = time.time()
            print("beam search takes", end_t - start_t, flush=True)
            # Return the best sequence from the beams
            return max(beams, key=lambda x: x.score)
        
        def sample_decoding(z_i, num_tokens):
            print("start beam search", flush=True)
            prev_tokens = []
            for i in tqdm(range(num_tokens)):
                _, topk_tokens = torch.topk(z_i[i], 1)
                prev_tokens += [topk_tokens[0]]
            return BeamCandidate(prev_tokens)

        def label_smoothing(best_sequence):
            next_z_i = torch.nn.functional.one_hot(torch.tensor(best_sequence), self.encoder_vocab_size).float()
            next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (self.encoder_vocab_size - 1)
            z_i = torch.nn.Parameter(torch.log(next_z_i), requires_grad=True)
            return z_i

        def random_z_i(num_tokens):
            z_i = torch.nn.Parameter(torch.rand((num_tokens, self.encoder_vocab_size)), requires_grad=True)
            return z_i
        
        z_i = random_z_i(16)

        noise_level = 5
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch}/{epoch_num}")
            ## optimize
            # z_i = torch.nn.Parameter(torch.zeros((num_tokens, vocab_size), device=device), True)
            optimizer = torch.optim.Adam([z_i], lr=lr)
            print(z_i)
            for i in range(perturb_iter):
                # perturb_iter = 5
                optimizer.zero_grad()
                naturalness = self.compute_naturalness(z_i, stemp)
                perplexity = self.compute_soft_perplexity([z_i], stemp)[0]
                # print(doc_cos_sim, trig_cos_sim)
                # loss = 0.25 * doc_cos_sim - trig_cos_sim
                # loss = 1 - 3 * trig_cos_sim - 0.1 * naturalness + 0.1 * perplexity
                # it only produces a natural sentence when optimizing for both naturalness and perplexity
                # loss = 1 - 20 * trig_cos_sim - 0.01 * naturalness + 0.01 * perplexity
                loss = 1 - score_fn(0, naturalness, perplexity)
                print(epoch, i, 'naturalness', naturalness.item(), 'perplexity', perplexity.item(), 'loss', loss.item(), flush=True)
                loss.backward()
                optimizer.step()
                noise_level = noise_level * 0.8
                # z_i = z_i + torch.rand_like(z_i) * lr * noise_level
            ## search
            ## TODO: add llm guidance here
            z_i = z_i.detach()
            print(z_i)
            candidate: BeamCandidate = beam_search(z_i, len(z_i), None, None, topk, beam_width, stemp)
            # candidate: BeamCandidate = sample_decoding(z_i, len(start_tokens))
            # candidate: BeamCandidate = beam_search_only_cos_sim(z_i, len(start_tokens), doc_embs, trig_doc_embs, topk, beam_width, stemp)
            print(epoch, 'best_score', candidate.score, 'cos_sim', candidate.trig_cos_sim, 'naturalness', candidate.naturalness, 'perplexity', candidate.perplexity)
            best_sequence = candidate.sequence
            print('best_sequence', [int(token.detach().cpu().numpy()) for token in best_sequence])
            best_seq_str = self.encoder_tokenizer.decode(best_sequence)
            print(best_seq_str)

            z_i = label_smoothing(best_sequence)
        return best_seq_str 


if __name__ == '__main__':
    optimizer = NaturalCOLD()
    optimizer.optimize()