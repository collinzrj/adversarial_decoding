from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from adversarial_decoding.utils.utils import top_k_top_p_filtering, file_device
from adversarial_decoding.utils.chat_format import ChatFormat

class LLMWrapper:
    """
    Simple wrapper to run forward pass on an LLM,
    returning top-k/p filtered next-token proposals.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_tokens, chat_format, device=file_device, repetition_penalty: float = 1.0):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.chat_format: ChatFormat = chat_format
        self.prompt_tokens = prompt_tokens
        self.repetition_penalty = repetition_penalty

    def generate(self, suffix):
        full_tokens = self.chat_format.prepare_input(self.prompt_tokens, self.tokenizer.encode(suffix, add_special_tokens=False))
        with torch.no_grad():
            outputs = self.model.generate(torch.tensor([full_tokens]).to(self.model.device), max_length=200)
            return self.tokenizer.decode(outputs[0])
        
    def template_example(self):
        text = 'hello world'
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        full_tokens = self.chat_format.prepare_prefix_input(self.prompt_tokens, tokens)
        print([self.tokenizer.decode(full_tokens)])

    def get_next_token_candidates(
        self,
        batch_tokens: List[List[int]],
        top_k: int = 10,
        top_p: float = 0.9,
        exclude_ids: Optional[List[int]] = None,
        batch_kv_cache: Optional[List[DynamicCache]] = None
    ):
        """
        1) forward pass on input_ids
        2) top-k/top-p filtering
        3) return list of lists of (token_id, log_probability), one list per input
        Along with updated KV caches for faster inference
        """

        ## assert all tokens have the same length
        assert len(set(len(t) for t in batch_tokens)) == 1
        full_tokens = [self.chat_format.prepare_prefix_input(self.prompt_tokens, t) for t in batch_tokens]
        # print('LLM Wrapper: ', self.tokenizer.decode(full_tokens[0]))
        input_ids = torch.tensor(full_tokens).to(self.model.device)
        
        # Setup KV cache
        use_cache = batch_kv_cache is not None
        if use_cache:
            # Similar to naturalness_scorer, merge individual caches into a batch
            kv_cache = DynamicCache.from_batch_splits(batch_kv_cache)
            kv_cache.to(self.model.device)
            # Only process new tokens not in the cache
            cache_seq_len = kv_cache.get_seq_length()
            model_inputs = input_ids[:, cache_seq_len:]
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
        else:
            kv_cache = DynamicCache()
            model_inputs = input_ids
            attention_mask = None

        with torch.no_grad():
            outputs = self.model(
                input_ids=model_inputs,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True
            )
            
        # Get updated KV cache
        next_kv_cache = outputs.past_key_values
        next_batch_kv_cache = None
        if next_kv_cache is not None:
            try:
                next_batch_kv_cache = next_kv_cache.batch_split(len(batch_tokens), 1)
            except Exception as e:
                print(f"Warning: Failed to split KV cache: {e}")
                next_batch_kv_cache = None
        
        # Get logits for the next token
        logits = outputs.logits[:, -1, :]
        # Apply repetition penalty if specified
        if self.repetition_penalty != 1.0:
            # Get the tokens that have already appeared in each sequence
            for i, tokens in enumerate(full_tokens):
                # Create a set of unique tokens in the sequence
                unique_tokens = set(tokens)
                
                # Apply penalty to each token that has already appeared
                for token_id in unique_tokens:
                    # Get the current logit value
                    token_logit = logits[i, token_id].item()
                    
                    # Apply the penalty based on whether the logit is positive or negative
                    if token_logit > 0:
                        # Divide positive logits by the penalty
                        logits[i, token_id] = logits[i, token_id] / self.repetition_penalty
                    else:
                        # Multiply negative logits by the penalty
                        logits[i, token_id] = logits[i, token_id] * self.repetition_penalty
        
        
        # Apply mask for chat format
        mask_tokens = []
        for mask_word in ['<|im_end|>', '<|end_header_id|>', '<|start_header_id|>', '@', '\xa0', '<|eot_id|>', '<|eom_id|>', '"', '<|python_tag|>', '\n', '\n\n', ' \n\n']:
            tokens = self.tokenizer.encode(mask_word, add_special_tokens=False)
            if len(tokens) == 1:
                mask_tokens.append(tokens[0])
        logits[:, mask_tokens] = -1e10
        log_probs = F.log_softmax(logits, dim=-1)        

        # if exclude_ids:
        #     log_probs[:, exclude_ids] = -1e10

        # Process each item in the batch
        result = []
        for i in range(logits.shape[0]):
            # do top-k/top-p filtering on logits, then gather their log_probs
            filtered_indices = top_k_top_p_filtering(logits[i], top_k, top_p)
            filtered_log_probs = log_probs[i, filtered_indices]

            # sort descending
            sorted_vals, sorted_idx = torch.sort(filtered_log_probs, descending=True)
            sorted_indices = filtered_indices[sorted_idx.cpu()]

            # add to result as python lists of (token_id, log_prob)
            result.append(list(zip(sorted_indices.tolist(), sorted_vals.tolist())))
            
        return result, next_batch_kv_cache 