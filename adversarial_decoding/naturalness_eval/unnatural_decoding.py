from transformers import AutoModel, AutoTokenizer, BertModel, T5ForConditionalGeneration, AutoModelForMaskedLM, AutoModelForCausalLM, LlamaModel
import torch, time, os, json
from tqdm import tqdm
from dataclasses import dataclass, field
import random, pickle
from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BertModel, T5ForConditionalGeneration, AutoModelForMaskedLM, AutoModelForCausalLM, LlamaModel
import torch, time
import random
import torch.nn.functional as F
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

def compute_repetitiveness(sentence):
    # Convert to lowercase and split into words
    words = sentence.lower().split()
    if len(words) == 0:
        return 0
    word_cnt = {}
    for word in words:
        word_cnt.setdefault(word, 0)
        word_cnt[word] += 1
    total = len(words)
    cnt = 0
    for word in word_cnt:
        if word_cnt[word] > 0:
            cnt += word_cnt[word] - 1
    if total == 0:
        return 0
    return cnt / total

def compute_no_alphabet(sentence):
    total = 0
    cnt = 0
    for c in sentence:
        if c == ' ':
            continue
        if not (c.isalpha() or c == ',' or c == '.'):
            cnt += 1
        total += 1
    if total == 0:
        return 0
    return cnt / total

class UnnaturalDecoding:    
    def __init__(self):
        device = torch.get_default_device()
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
        # model_name = "meta-llama/Meta-Llama-3.1-70B"
        self.naturalness_llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.naturalness_llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        # self.encoder_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        # self.encoder: BertModel = AutoModel.from_pretrained('facebook/contriever').to(device)
        # self.causal_llm_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        # self.causal_llm = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        self.encoders = [SentenceTransformer("facebook/contriever", device='cuda')]
        self.causal_llm_tokenizer = self.naturalness_llm_tokenizer
        self.causal_llm = self.naturalness_llm
    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    def compute_naturalness(self, texts, yes_token=9642, no_token=2822):
        results = []
        for i in range(0, len(texts), 100):
            naturalness_list = self.compute_naturalness_small_batch(texts[i:i+100])
            # naturalness_list2 = self.compute_naturalness_small_batch2(texts[i:i+100])
            # avg_naturalness = (naturalness_list + naturalness_list2) / 2
            results.append(naturalness_list)
        return torch.cat(results)


    def display_tokens(self, tokens, tokenizer):
        print([tokenizer.decode(token) for token in tokens])

    def compute_naturalness_small_batch(self, texts, yes_token=9642, no_token=2822):
        tokens_list = []
        for text in texts:
            query = f"""Is this text unintelligible? "{text}". Just answer Yes or No."""
            messages = [
                {"role": "user", "content": query},
            ]
            tokens = self.naturalness_llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            tokens_list.append(tokens)
        max_length = max([len(tokens) for tokens in tokens_list])
        model_input_tokens = torch.zeros((len(tokens_list), max_length)).to(torch.long)
        model_attention_mask = torch.zeros((len(tokens_list), max_length)).to(torch.long)
        for i in range(len(tokens_list)):
            for j in range(len(tokens_list[i])):
                model_input_tokens[i][-1 - j] = tokens_list[i][-1 - j]
                model_attention_mask[i][-1 - j] = 1
        input_ids = torch.tensor(model_input_tokens)
        attention_mask = torch.tensor(model_attention_mask)
        # print('check compute naturalness', flush=True)
        # print(tokens_list)
        # print(input_ids)
        # print(attention_mask)
        with torch.no_grad():
            outputs = self.naturalness_llm(input_ids=input_ids, attention_mask=attention_mask)
            yes_logits = outputs.logits[:, -1, yes_token]
            no_logits = outputs.logits[:, -1, no_token]
            # yes_prob = torch.exp(yes_logits) / (torch.exp(yes_logits) + torch.exp(no_logits))
            # no_prob = torch.exp(no_logits) / (torch.exp(yes_logits) + torch.exp(no_logits))
            yes_prob = yes_logits
            no_prob = no_logits
        return (no_prob - yes_prob) / (yes_prob + no_prob) 


    def natural_generate(self):
        pass

    def optimize(self, query, llm_topk=10, llm_beam_width=10, max_length=32):
        slice_num = 16
        print("llm_topk", llm_topk)
        print("llm_beam_width", llm_beam_width)
        print("max_length", max_length)

        @dataclass
        class LLMBeamCandidate:
            sequence: ... = field(default_factory=list)
            sequence_str: ... = None
            score: ... = 0
            trig_cos_sim: ... = None
            naturalness: ... = None

        candidates = [LLMBeamCandidate(sequence=[])]

        print(query)
        start_prompts = [
            self.causal_llm_tokenizer.encode(query)[:16],
        ]
        
        for epoch in tqdm(range(max_length)):
            all_candidates = []
            start_t = time.time()
            model_total_time = 0
            for llm_candidate in candidates:
                seq, score = llm_candidate.sequence, llm_candidate.score
                sliced_seq = seq[-(epoch % slice_num):]
                input_ids = torch.tensor([start_prompts[0] + seq])
                with torch.no_grad():
                    model_start = time.time()
                    outputs = self.causal_llm(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_ids = torch.topk(next_token_probs, llm_topk)
                    # print('top_k_ids shape', top_k_ids.shape) => [1, 50]
                    model_end = time.time()
                    model_total_time += model_end - model_start

                    candidate_batch = []
                    for i in range(llm_topk):
                        new_tok_id = top_k_ids[0][i].item()
                        new_seq = seq + [new_tok_id]
                        new_seq_str = self.causal_llm_tokenizer.decode(new_seq)
                        candidate_batch.append(LLMBeamCandidate(sequence=new_seq, sequence_str=new_seq_str))
                    naturalness_batch = self.compute_naturalness([candidate.sequence_str for candidate in candidate_batch])

                    for i in range(len(candidate_batch)):
                        candidate = candidate_batch[i]
                        naturalness = naturalness_batch[i]
                        clipped_naturalness = torch.clamp(naturalness, max=1)
                        repetitive_ness = compute_repetitiveness(candidate.sequence_str)
                        noengness = compute_no_alphabet(candidate.sequence_str)
                        candidate.score = - clipped_naturalness - repetitive_ness - noengness
                        candidate.naturalness = clipped_naturalness
                        all_candidates.append(candidate)

            sorted_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
            
            # add more random stuff here
            candidates = sorted_candidates[:llm_beam_width]
            # candidates = sorted_candidates[:llm_beam_width//2] + random.sample(sorted_candidates[llm_beam_width//2:], min(llm_beam_width - llm_beam_width//2, len(sorted_candidates[llm_beam_width//2:])))
            # print("sequences len", len(sequences))
            # print(encoder_tokenizer.batch_decode([pair[0] for pair in sequences]))
            end_t = time.time()
            print("It takes", end_t - start_t, "model takes", model_total_time)
            self.display_tokens(candidates[0].sequence, self.causal_llm_tokenizer)
            print(candidates[0].sequence_str)
            print(epoch, "score", candidates[0].score, 'naturalness', candidates[0].naturalness, 'trig_cos_sim', candidates[0].trig_cos_sim, flush=True)

        # best_sequence = candidates[0].sequence
        print('result str', candidates[0].sequence_str)
        return candidates[0].sequence_str
    
    def random_generate(self, query, llm_topk=10, llm_beam_width=1, max_length=32):
        print(query)
        # Encode the initial prompt
        start_prompt_ids = self.causal_llm_tokenizer.batch_encode_plus(
            [query] * llm_beam_width, max_length=16, truncation=True
        )['input_ids']
        start_prompt_tensor = torch.tensor(start_prompt_ids)

        # Generate initial past_key_values
        outputs = self.causal_llm(input_ids=start_prompt_tensor, use_cache=True)
        past_key_values = outputs.past_key_values  # List of tuples

        # Initialize candidates with tokens and their past_key_values
        candidates = []
        for i in range(llm_beam_width):
            candidate_past_key_values = []
            for layer_past in past_key_values:
                # Extract the ith batch for each layer
                key_i = layer_past[0][i:i+1]
                value_i = layer_past[1][i:i+1]
                candidate_past_key_values.append((key_i, value_i))
            candidates.append({
                'tokens': start_prompt_ids[i],
                'past_key_values': candidate_past_key_values,
                'start_length': len(start_prompt_ids[i])
            })

        # Start generating tokens using KV cache
        for epoch in tqdm(range(max_length)):
            # Prepare input_ids and past_key_values for all candidates
            input_ids = torch.tensor([[candidate['tokens'][-1]] for candidate in candidates])
            past_key_values = []
            num_layers = len(candidates[0]['past_key_values'])
            for layer in range(num_layers):
                keys = torch.cat([candidate['past_key_values'][layer][0] for candidate in candidates], dim=0)
                values = torch.cat([candidate['past_key_values'][layer][1] for candidate in candidates], dim=0)
                past_key_values.append((keys, values))

            # Run the model with the last token and past_key_values
            outputs = self.causal_llm(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)

            # Update candidates with the new token and past_key_values
            new_candidates = []
            for idx, candidate in enumerate(candidates):
                top_k_probs, top_k_ids = torch.topk(next_token_logits[idx], llm_topk)
                next_token = random.choice(top_k_ids).item()
                candidate['tokens'].append(next_token)

                # Update past_key_values for the candidate
                candidate_past_key_values = []
                for layer_past in outputs.past_key_values:
                    key_i = layer_past[0][idx:idx+1]
                    value_i = layer_past[1][idx:idx+1]
                    candidate_past_key_values.append((key_i, value_i))
                candidate['past_key_values'] = candidate_past_key_values
                new_candidates.append(candidate)
            candidates = new_candidates

        # Extract the generated tokens and compute scores
        results = []
        for candidate in candidates:
            start_length = candidate['start_length']
            generated_tokens = candidate['tokens'][start_length:]
            results.append(generated_tokens)

        scores = self.compute_naturalness(results).tolist()
        results_str = self.causal_llm_tokenizer.batch_decode(results)
        return list(zip(results_str, scores))


    def cos_sim_optimize(self, llm_topk=10, llm_beam_width=50, max_length=32, only_cos_sim=True):
        print("llm_topk", llm_topk)
        print("llm_beam_width", llm_beam_width)
        print("max_length", max_length)

        random_embedding = torch.rand((1, 768))

        @dataclass
        class LLMBeamCandidate:
            sequence: ... = field(default_factory=list)
            sequence_str: ... = None
            score: ... = 0
            trig_cos_sim: ... = None
            avg_perp: ... = None
            total_perp: ... = 0

        candidates = [LLMBeamCandidate(sequence=[])]
        start_prompt = self.causal_llm_tokenizer.encode(f"tell me a story:")
        
        for epoch in range(max_length):
            all_candidates = []
            model_total_time = 0
            for llm_candidate in candidates:
                seq, score = llm_candidate.sequence, llm_candidate.score
                input_ids = torch.tensor([start_prompt + seq])
                with torch.no_grad():
                    model_start = time.time()
                    outputs = self.causal_llm(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_ids = torch.topk(next_token_probs, llm_topk)
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
                        cos_sim_batch = torch.nn.functional.cosine_similarity(candidate_embedding, random_embedding)
                        cos_sim_batch_list.append(cos_sim_batch)
                    cos_sim_batch_list = torch.stack(cos_sim_batch_list).T
                    for i in range(len(candidate_batch)):
                        cos_sim = cos_sim_batch_list[i].mean()
                        candidate = candidate_batch[i]
                        candidate.score = cos_sim.item()
                        all_candidates.append(candidate)
            sorted_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
            candidates = sorted_candidates[:llm_beam_width]
            print(candidates[0].sequence)
            print(candidates[0].sequence_str)
            print(epoch, "score", candidates[0].score, 'perp', candidates[0].total_perp / max_length, 'trig_cos_sim', candidates[0].trig_cos_sim, flush=True)
        print('result str', candidates[0].sequence_str)
        return candidates[0].sequence_str
    

if __name__ == '__main__':
    torch.set_default_device('cuda')
    optimizer = UnnaturalDecoding()
    ds = load_dataset("lmsys/lmsys-chat-1m")
    idx = 0
    timestamp = str(time.time())
    for sample in ds['train']:
        print("idx", idx)
        idx += 1
        query = sample['conversation'][0]['content']
        if True:
            res = []
            for _ in range(1):
                res += optimizer.random_generate(query, llm_topk=10, llm_beam_width=1, max_length=16)
            with open(f'random_sample_results_{timestamp}.txt', 'a') as f:
                f.write(json.dumps({
                    'query': query,
                    'response': res
                }, indent=2) + '\n')
        else:
            print(idx, query)
            res = optimizer.optimize(query, llm_topk=10, llm_beam_width=2, max_length=16)
            with open(f'unnaturl_decoding_{timestamp}.txt', 'a') as f:
                f.write(repr(res) + '\n')
            