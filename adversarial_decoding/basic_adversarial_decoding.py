from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BertModel, AutoModelForCausalLM
import torch, time
import random
from torch.nn.functional import normalize
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

def set_no_grad(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def repetition_score(text):
    score = 0
    words = text.split(' ')
    for i, word in enumerate(words):
        if not word.isalnum():
            continue
        if word in words[:i] or len(word) > 10:
            score += len(word)
    return score / len(text)

def generate_banned_tokens(this_tokenizer):
    banned_tokens = []
    for token in range(this_tokenizer.vocab_size - 1):
        word = this_tokenizer.decode([token])
        if word.count(' ') > 1:
            banned_tokens.append(token)
            continue
        if word == '.' or word == '?' or word == '!' or word == ',':
            continue
        for char in word:
            if char == ' ':
                continue
            if ord('a') <= ord(char) and ord(char) <= ord('z'):
                continue
            if ord('A') <= ord(char) and ord(char) <= ord('Z'):
                continue
            banned_tokens.append(token)
            break
    return banned_tokens

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def compute_average_cosine_similarity(embeddings):
    # Normalize the embeddings to have unit norm
    embeddings = torch.tensor(embeddings)
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute the cosine similarity matrix
    cos_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    # Mask out the diagonal elements (self-similarity) and compute the average of the off-diagonal elements
    mask = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device).bool()
    cos_sim_matrix.masked_fill_(mask, 0)
    
    # Compute the average cosine similarity (excluding self-similarity)
    avg_cos_sim = cos_sim_matrix.sum() / (cos_sim_matrix.numel() - cos_sim_matrix.size(0))

    # print("document cos sim", avg_cos_sim, 'min', cos_sim_matrix.min())
    
    return avg_cos_sim.item()


class BasicAdversarialDecoding:

    def __init__(self):
        device = torch.get_default_device()
        # self.encoder_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        # self.encoders = [SentenceTransformer("sentence-transformers/gtr-t5-base", device='cuda'), SentenceTransformer("facebook/contriever", device='cuda')]
        self.encoders = [SentenceTransformer("facebook/contriever", device='cuda')]
        # self.encoders = [SentenceTransformer("sentence-transformers/sentence-t5-base", device='cuda'), SentenceTransformer("facebook/contriever", device='cuda')]
        # self.naturalness_llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        # self.naturalness_llm_tokenizer.pad_token = self.naturalness_llm_tokenizer.eos_token
        # self.naturalness_llm = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct').to(device)
        self.causal_llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.causal_llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        set_no_grad(self.causal_llm)
        self.causal_llm_tokenizer.pad_token = self.causal_llm_tokenizer.eos_token
        self.EOS_TOKEN = self.causal_llm_tokenizer.encode(self.causal_llm_tokenizer.pad_token)[0]
        self.llm_banned_tokens = generate_banned_tokens(self.causal_llm_tokenizer)
        self.llm_banned_tokens.append(188)
 
    def should_drop(self, tokens):
        # Loop through the tokens list
        for i in range(len(tokens) - 1):
            # Check if the current token and the next token are both EOS_TOKEN
            if tokens[i] == self.EOS_TOKEN and tokens[i + 1] == self.EOS_TOKEN:
                return True
        for i in range(len(tokens) - 2):
            # prevent three continuous tokens
            if tokens[i] == tokens[i + 1] and tokens[i] == tokens[i + 2]:
                return True
        return False

    def optimize(self, documents, trigger, llm_topk=10, llm_beam_width=50, max_length=32, only_cos_sim=True):
        print("llm_topk", llm_topk)
        print("llm_beam_width", llm_beam_width)
        print("max_length", max_length)
        target_embedding_list = [encoder.encode(documents, normalize_embeddings=True, convert_to_tensor=True) for encoder in self.encoders]

        @dataclass
        class LLMBeamCandidate:
            sequence: ... = field(default_factory=list)
            sequence_str: ... = None
            score: ... = 0
            trig_cos_sim: ... = None
            avg_perp: ... = None
            total_perp: ... = 0

        candidates = [LLMBeamCandidate(sequence=[])]
        start_prompt = self.causal_llm_tokenizer.encode(f"tell me a story about {trigger}:")
        
        for epoch in range(max_length):
            all_candidates = []
            start_t = time.time()
            model_total_time = 0
            random_encoder_idx = random.randint(0, len(self.encoders) - 1)
            print(random_encoder_idx, "selected")
            for llm_candidate in candidates:
                seq, score = llm_candidate.sequence, llm_candidate.score
                sliced_seq = seq[-(epoch % 32):]
                input_ids = torch.tensor([start_prompt + sliced_seq])
                with torch.no_grad():
                    model_start = time.time()
                    outputs = self.causal_llm(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                    next_token_probs[0][self.llm_banned_tokens] = -100
                    top_k_probs, top_k_ids = torch.topk(next_token_probs, llm_topk)
                    # print('top_k_ids shape', top_k_ids.shape) => [1, 50]
                    model_end = time.time()
                    model_total_time += model_end - model_start

                    candidate_batch = []
                    for i in range(llm_topk):
                        new_seq = seq + [top_k_ids[0][i].item()]
                        new_seq_str = self.causal_llm_tokenizer.decode(new_seq)
                        candidate_batch.append(LLMBeamCandidate(sequence=new_seq, sequence_str=new_seq_str, total_perp=llm_candidate.total_perp - top_k_probs[0][i].item()))
                    cos_sim_batch_list = []
                    for idx, encoder in enumerate(self.encoders):
                        candidate_embedding = encoder.encode([candidate.sequence_str for candidate in candidate_batch], convert_to_tensor=True, normalize_embeddings=True)
                        # cos_sim = torch.nn.functional.cosine_similarity(candidate_embedding, target_embedding).mean()
                        cos_sim_batch = torch.matmul(normalize(candidate_embedding), normalize(target_embedding_list[idx]).t()).mean(dim=1)
                        # combined_score = score + top_k_probs[0][i].item() + alpha * cos_sim
                        cos_sim_batch_list.append(cos_sim_batch)
                    cos_sim_batch_list = torch.stack(cos_sim_batch_list).T
                    for i in range(len(candidate_batch)):
                        cos_sim = cos_sim_batch_list[i].mean()
                        if self.should_drop(candidate_batch[i].sequence_str):
                            continue
                        # spell check but ignore last
                        penalty = repetition_score(candidate_batch[i].sequence_str) * 0.05
                        candidate = candidate_batch[i]
                        # candidate.score = cos_sim - candidate.total_perp - penalty
                        # candidate.score = cos_sim.item() - 0.01 * candidate.total_perp / (epoch + 1)
                        if only_cos_sim:
                            candidate.score = cos_sim.item()
                        else:
                            candidate.score = cos_sim.item() - 0.05 * candidate.total_perp / (epoch + 1) - penalty
                        candidate.trig_cos_sim = cos_sim_batch_list[i]
                        all_candidates.append(candidate)

            sorted_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
            
            # add more random stuff here
            candidates = sorted_candidates[:llm_beam_width//2] + random.sample(sorted_candidates[llm_beam_width//2:], min(llm_beam_width - llm_beam_width//2, len(sorted_candidates[llm_beam_width//2:])))
            # print("sequences len", len(sequences))
            # print(encoder_tokenizer.batch_decode([pair[0] for pair in sequences]))
            end_t = time.time()
            print("It takes", end_t - start_t, "model takes", model_total_time)
            print(candidates[0].sequence)
            print(candidates[0].sequence_str)
            print(epoch, "score", candidates[0].score, 'perp', candidates[0].total_perp / max_length, 'trig_cos_sim', candidates[0].trig_cos_sim, flush=True)

        # best_sequence = candidates[0].sequence
        # result = self.encoder_tokenizer.encode(candidates[0].sequence_str)
        print('result str', candidates[0].sequence_str)
        # print(result)
        return candidates[0].sequence_str
    
