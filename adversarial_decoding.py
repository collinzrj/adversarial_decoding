from transformers import AutoModel, AutoTokenizer, BertModel, T5ForConditionalGeneration, AutoModelForMaskedLM, AutoModelForCausalLM, LlamaModel
import torch, time, os
from tqdm import tqdm
from dataclasses import dataclass, field
import random, pickle
from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BertModel, T5ForConditionalGeneration, AutoModelForMaskedLM, AutoModelForCausalLM, LlamaModel
import torch, time
import random
from torch.nn.functional import normalize
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, cast


class AdversarialDecoding:    
    def __init__(self):
        device = torch.get_default_device()
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
        # model_name = "meta-llama/Meta-Llama-3.1-70B"
        self.naturalness_llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.naturalness_llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.encoder: BertModel = AutoModel.from_pretrained('facebook/contriever').to(device)
        # self.causal_llm_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        # self.causal_llm = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        self.causal_llm_tokenizer = self.naturalness_llm_tokenizer
        self.causal_llm = self.naturalness_llm
    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def compute_doc_embs(self, documents):
        doc_inputs = self.encoder_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        doc_embs = self.mean_pooling(self.encoder(**doc_inputs)[0], doc_inputs['attention_mask'])
        return doc_embs
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

    def compute_naturalness_small_batch2(self, texts, yes_token=9642, no_token=2822):
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
        

    def optimize(self, documents, trigger, llm_topk=10, llm_beam_width=10, max_length=32):
        slice_num = 16
        print("llm_topk", llm_topk)
        print("llm_beam_width", llm_beam_width)
        print("max_length", max_length)
        target_embedding = self.compute_doc_embs(documents)

        @dataclass
        class LLMBeamCandidate:
            sequence: ... = field(default_factory=list)
            sequence_str: ... = None
            score: ... = 0
            trig_cos_sim: ... = None
            naturalness: ... = None

        candidates = [LLMBeamCandidate(sequence=[])]


        start_prompts = [
            self.causal_llm_tokenizer.encode(f"tell me a story about {trigger}:"),
        ]
        
        for epoch in tqdm(range(max_length)):
            all_candidates = []
            start_t = time.time()
            model_total_time = 0
            for llm_candidate in candidates:
                seq, score = llm_candidate.sequence, llm_candidate.score
                sliced_seq = seq[-(epoch % slice_num):]
                input_ids = torch.tensor([start_prompts[0] + seq])
                # chat_template: List[int] = self.naturalness_llm_tokenizer.apply_chat_template([
                #     {"role": "user", "content": f"tell me a story about {trigger}."},
                #     {"role": "assistant", "content": ""}
                # ]) # type: ignore
                # # remove <eot_id> at the end
                # input_ids = torch.tensor([chat_template[:-1] + seq])
                # # print("input_ids shape", input_ids.shape)
                with torch.no_grad():
                    model_start = time.time()
                    outputs = self.causal_llm(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                    next_token_probs[0][628] = -1000
                    next_token_probs[0][198] = -1000
                    top_k_probs, top_k_ids = torch.topk(next_token_probs, llm_topk)
                    # print('top_k_ids shape', top_k_ids.shape) => [1, 50]
                    model_end = time.time()
                    model_total_time += model_end - model_start

                    candidate_batch = []
                    for i in range(llm_topk):
                        new_tok_id = top_k_ids[0][i].item()
                        # if (epoch + 1) % slice_num == 0:
                        #     new_seq = seq + self.causal_llm_tokenizer.encode('.', add_special_tokens=False) + [new_tok_id]
                        # else:
                        #     new_seq = seq + [new_tok_id]
                        new_seq = seq + [new_tok_id]
                        new_seq_str = self.causal_llm_tokenizer.decode(new_seq)
                        if '#' in new_seq_str: 
                            pass
                        if '@' in new_seq_str:
                            pass
                        candidate_batch.append(LLMBeamCandidate(sequence=new_seq, sequence_str=new_seq_str))
                    candidate_embedding = self.compute_doc_embs([candidate.sequence_str for candidate in candidate_batch])
                    # cos_sim = torch.nn.functional.cosine_similarity(candidate_embedding, target_embedding).mean()
                    cos_sim_batch = torch.matmul(normalize(candidate_embedding), normalize(target_embedding).t()).mean(dim=1)
                    naturalness_batch = self.compute_naturalness([candidate.sequence_str for candidate in candidate_batch])

                    # combined_score = score + top_k_probs[0][i].item() + alpha * cos_sim
                    for i in range(len(candidate_batch)):
                        cos_sim = cos_sim_batch[i]
                        candidate = candidate_batch[i]
                        naturalness = naturalness_batch[i]
                        clipped_naturalness = torch.clamp(naturalness, max=0.02)
                        # candidate.score = cos_sim.item() + clipped_naturalness * max((epoch / max_length) * 1, 0.5)
                        candidate.score = cos_sim.item() + clipped_naturalness
                        # candidate.score = naturalness
                        candidate.trig_cos_sim = cos_sim.item()
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
        result = self.encoder_tokenizer.encode(candidates[0].sequence_str)
        print('result str', candidates[0].sequence_str)
        print(result)
        return candidates[0].sequence_str