import torch
from typing import List
from tqdm import tqdm
from transformers import DynamicCache, AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.scorers.base_scorer import Scorer
from adversarial_decoding.utils.chat_format import ChatFormat
from adversarial_decoding.utils.utils import file_device

class PerplexityScorer(Scorer):
    """
    Example scorer that uses an LLM to compute perplexity for the entire sequence (or portion).
    Negative perplexity is often used as the final score.
    """

    def __init__(self, llm, tokenizer, chat_format, prompt_tokens, target_tokens, chunk_size=10, ignore_tokens_num=1):
        self.llm = llm
        self.tokenizer = tokenizer
        self.chat_format: ChatFormat = chat_format
        self.chunk_size = chunk_size
        self.ignore_tokens_num = ignore_tokens_num
        self.prompt_tokens = prompt_tokens
        self.target_tokens = target_tokens

    def compute_perplexity_batch(self, tokens_batch, batch_kv_cache: List[DynamicCache], next_cache_seq_len, ignore_tokens_num=1):
        # memory_allocated = torch.cuda.memory_allocated()
        # print(f"naturalness perplexity Allocated memory: {memory_allocated / 1024**2:.2f} MB")
        assert ignore_tokens_num >= 1
        # input_ids = torch.tensor([seq]).to(device)
        if batch_kv_cache[0] is None:
            kv_cache = DynamicCache()
        else:
            kv_cache = DynamicCache.from_batch_splits(batch_kv_cache)
            kv_cache.key_cache = [c.to(file_device) for c in kv_cache.key_cache]
            kv_cache.value_cache = [c.to(file_device) for c in kv_cache.value_cache]
        cache_seq_len = kv_cache.get_seq_length()
        inputs = torch.tensor(tokens_batch)
        inputs = inputs[:, cache_seq_len:]
        attention_mask = torch.ones_like(inputs)
        labels = inputs
        ignore_tokens_num = ignore_tokens_num - cache_seq_len
        with torch.no_grad():
            outputs = self.llm(input_ids=inputs.to(self.llm.device), attention_mask=torch.ones_like(torch.tensor(tokens_batch)).to(self.llm.device), past_key_values=kv_cache, use_cache=True) 
        next_kv_cache: DynamicCache = outputs.past_key_values
        # next_kv_cache.key_cache = [c.to('cpu', non_blocking=True) for c in next_kv_cache.key_cache]
        # next_kv_cache.value_cache = [c.to('cpu', non_blocking=True) for c in next_kv_cache.value_cache]
        del kv_cache
        del outputs.past_key_values
        next_kv_cache.crop(next_cache_seq_len)
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., ignore_tokens_num-1:-1, :].contiguous().to(self.llm.device)
        shift_labels = labels[..., ignore_tokens_num:].contiguous().to(self.llm.device)
        shift_masks = attention_mask[..., ignore_tokens_num:].contiguous().to(self.llm.device)
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape[0], -1) * shift_masks
        loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
        return torch.exp(loss), next_kv_cache.batch_split(len(tokens_batch), 1)

    def compute_perplexity(
        self,
        tokens_batch: List[List[int]],
        batch_kv_cache: List[DynamicCache],
        next_cache_seq_len: int,
        ignore_tokens_num: int
    ):
        """
        Splits tokens_batch into chunks, calls `compute_perplexity_batch`,
        then concatenates results.
        """
        all_ppls = []
        all_kvs = []
        for i in range(0, len(tokens_batch), self.chunk_size):
            chunk = tokens_batch[i:i+self.chunk_size]
            chunk_kv = batch_kv_cache[i:i+self.chunk_size]
            ppl_tensor, kv_tensor = self.compute_perplexity_batch(
                chunk, chunk_kv, next_cache_seq_len, ignore_tokens_num
            )
            all_ppls.append(ppl_tensor)
            all_kvs.extend(kv_tensor)
        return torch.cat(all_ppls), all_kvs

    def template_example(self):
        text = 'hello world'
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        full_tokens = self.chat_format.prepare_prefix_input(self.prompt_tokens, tokens)
        print(self.tokenizer.decode(full_tokens))

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """
        Batch-compute perplexity for all candidates, store it in candidate.perplexity,
        and return negative perplexity as a score (example approach).
        """
        if len(candidates) == 0:
            return []

        # 1) Prepare input for each candidate
        tokens_batch = []
        kv_cache_batch = []
        # For demonstration, we skip any advanced prefix. Modify as needed.
        for c in candidates:
            # If you're appending something else, do it here:
            full_tokens = self.chat_format.prepare_input(self.prompt_tokens, c.token_ids) + self.target_tokens
            tokens_batch.append(full_tokens)
            kv_cache_batch.append(c.perplexity_kv_cache)  # might be None
        # print('PerplexityScorer: ', self.tokenizer.decode(tokens_batch[0]))

        # 2) For perplexity, you need the length of the prefix if using caches:
        next_cache_seq_len = len(self.chat_format.prepare_prefix_input(self.prompt_tokens, candidates[0].token_ids))  # or your chat_format logic

        # 3) Compute perplexities
        ppl_values, new_kv_caches = self.compute_perplexity(
            tokens_batch,
            kv_cache_batch,
            next_cache_seq_len,
            len(tokens_batch[0]) - len(self.target_tokens)
        )

        # 4) Fill in candidate metadata
        scores = []
        for i, cand in enumerate(candidates):
            cand.perplexity = ppl_values[i].item()
            cand.perplexity_kv_cache = new_kv_caches[i]
            sc = cand.perplexity
            cand.score += sc  # or set cand.score = sc, depending on your logic
            scores.append(cand.score)
        return scores

    def generate(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        full_tokens = self.chat_format.prepare_input([], tokens)
        with torch.no_grad():
            outputs = self.llm.generate(torch.tensor([full_tokens]).to(self.llm.device), max_length=800)
            return self.tokenizer.decode(outputs[0])


class PerplexityDifferentLLMScorer(Scorer):
    """
    Example scorer that uses an LLM to compute perplexity for the entire sequence (or portion).
    Negative perplexity is often used as the final score.
    """

    def __init__(self, llm, tokenizer, prompt, target, chunk_size=10, ignore_tokens_num=1):
        self.llm = llm
        self.tokenizer: AutoTokenizer = tokenizer
        self.chunk_size = chunk_size
        self.ignore_tokens_num = ignore_tokens_num
        self.prompt = prompt
        self.target_tokens = self.tokenizer.encode(target, add_special_tokens=False)

    def compute_perplexity_batch(self, input_ids, attention_mask, ignore_tokens_num):
        # for input_id in input_ids:
        #     print([self.tokenizer.decode(input_id)])
        #     print([self.tokenizer.decode(input_id[ignore_tokens_num:])])
        assert ignore_tokens_num >= 1
        labels = input_ids
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask) 
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., ignore_tokens_num-1:-1, :].contiguous().to(self.llm.device)
        # print("shift_logits", shift_logits)
        shift_labels = labels[..., ignore_tokens_num:].contiguous().to(self.llm.device)
        shift_masks = attention_mask[..., ignore_tokens_num:].contiguous().to(self.llm.device)
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape[0], -1) * shift_masks
        loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
        return torch.exp(loss)

    def compute_perplexity(
        self,
        input_ids,
        attention_mask,
        ignore_tokens_num: int
    ):
        """
        Splits tokens_batch into chunks, calls `compute_perplexity_batch`,
        then concatenates results.
        """
        all_ppls = []
        for i in range(0, len(input_ids), self.chunk_size):
            chunk = input_ids[i:i+self.chunk_size]
            chunk_mask = attention_mask[i:i+self.chunk_size]
            ppl_tensor = self.compute_perplexity_batch(
                chunk, chunk_mask, ignore_tokens_num
            )
            all_ppls.append(ppl_tensor)
        return torch.cat(all_ppls)

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """
        Batch-compute perplexity for all candidates, store it in candidate.perplexity,
        and return negative perplexity as a score (example approach).
        """
        if len(candidates) == 0:
            return []

        # 1) Prepare input for each candidate
        tokens_batch = []
        kv_cache_batch = []
        # For demonstration, we skip any advanced prefix. Modify as needed.
        for c in candidates:
            # If you're appending something else, do it here:
            try:
                full_tokens = self.tokenizer.apply_chat_template([
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': self.prompt + c.seq_str}
                ], add_generation_prompt=True) + self.target_tokens
            except:
                full_tokens = self.tokenizer.apply_chat_template([
                    {'role': 'user', 'content': self.prompt + c.seq_str}
                ], add_generation_prompt=True) + self.target_tokens
            # print([self.tokenizer.decode(full_tokens)])
            tokens_batch.append(full_tokens)

        max_len = max([len(tokens) for tokens in tokens_batch])
        pad_tokens_batch = []
        attention_mask = []
        for tokens in tokens_batch:
            pad_tokens_batch.append([self.tokenizer.eos_token_id] * (max_len - len(tokens)) + tokens)
            attention_mask.append([0] * (max_len - len(tokens)) + [1] * len(tokens))
        input_ids = torch.tensor(pad_tokens_batch).to(self.llm.device)
        attention_mask = torch.tensor(attention_mask).to(self.llm.device)

        # 3) Compute perplexities
        ppl_values = self.compute_perplexity(
            input_ids,
            attention_mask,
            len(input_ids[0]) - len(self.target_tokens)
        )

        # 4) Fill in candidate metadata
        scores = []
        for i, cand in enumerate(candidates):
            cand.perplexity = ppl_values[i].item()
            sc = cand.perplexity
            cand.score += sc  # or set cand.score = sc, depending on your logic
            scores.append(cand.score)
        return scores

    def generate(self, text):
        try:
            inputs = self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': text}
            ], add_generation_prompt=True) + self.target_tokens
        except:
            inputs = self.tokenizer.apply_chat_template([
                {'role': 'user', 'content': text}
            ], add_generation_prompt=True) + self.target_tokens
        with torch.no_grad():
            outputs = self.llm.generate(input_ids=torch.tensor([inputs], device=self.llm.device), max_length=800)
            return self.tokenizer.decode(outputs[0]) 