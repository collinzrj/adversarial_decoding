import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversarial_decoding.utils.utils import top_k_top_p_filtering, file_device
from adversarial_decoding.utils.chat_format import ChatFormat

class LLMWrapper:
    """
    Simple wrapper to run forward pass on an LLM,
    returning top-k/p filtered next-token proposals.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_tokens, chat_format, device=file_device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.chat_format: ChatFormat = chat_format
        self.prompt_tokens = prompt_tokens

    def generate(self, suffix):
        full_tokens = self.chat_format.prepare_input(self.prompt_tokens, self.tokenizer.encode(suffix, add_special_tokens=False))
        with torch.no_grad():
            outputs = self.model.generate(torch.tensor([full_tokens]).to(self.model.device), max_length=200)
            return self.tokenizer.decode(outputs[0])
        
    def template_example(self):
        text = 'hello world'
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        full_tokens = self.chat_format.prepare_prefix_input(self.prompt_tokens, tokens)
        print(self.tokenizer.decode(full_tokens))

    def get_next_token_candidates(
        self,
        batch_tokens: List[List[int]],
        top_k: int = 10,
        top_p: float = 0.9,
        exclude_ids: Optional[List[int]] = None
    ):
        """
        1) forward pass on input_ids
        2) top-k/top-p filtering
        3) return list of (token_id, log_probability)
        """

        ## assert all tokens have the same length
        assert len(set(len(t) for t in batch_tokens)) == 1
        full_tokens = [self.chat_format.prepare_prefix_input(self.prompt_tokens, t) for t in batch_tokens]
        # print('LLM Wrapper: ', self.tokenizer.decode(full_tokens[0]))
        input_ids = torch.tensor(full_tokens).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            mask_tokens = []
            for mask_word in ['<|end_header_id|>', '<|start_header_id|>', '@', '\xa0', '<|eot_id|>', '<|eom_id|>', '"', '<|python_tag|>', '\n', '\n\n', ' \n\n']:
                tokens = self.tokenizer.encode(mask_word, add_special_tokens=False)
                assert len(tokens) == 1
                mask_tokens.append(tokens[0])
            logits[:, mask_tokens] = -1e10
            log_probs = F.log_softmax(logits, dim=-1)        

        # if exclude_ids:
        #     log_probs[:, exclude_ids] = -1e10

        # do top-k/top-p filtering on logits, then gather their log_probs
        filtered_indices = top_k_top_p_filtering(logits[0], top_k, top_p)
        filtered_log_probs = log_probs[0, filtered_indices]

        # sort descending
        sorted_vals, sorted_idx = torch.sort(filtered_log_probs, descending=True)
        sorted_indices = filtered_indices[sorted_idx.cpu()]

        # return as python lists of (token_id, log_prob)
        return list(zip(sorted_indices.tolist(), sorted_vals.tolist())) 