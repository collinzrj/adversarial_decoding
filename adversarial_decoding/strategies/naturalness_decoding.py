import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.utils.utils import file_device
from adversarial_decoding.utils.chat_format import ChatFormat
from adversarial_decoding.strategies.base_strategy import DecodingStrategy
from adversarial_decoding.llm.llm_wrapper import LLMWrapper
from adversarial_decoding.scorers.naturalness_scorer import NaturalnessScorer
from adversarial_decoding.scorers.combined_scorer import CombinedScorer

class NaturalnessDecoding(DecodingStrategy):
    def __init__(self, score_target, device=file_device):
        self.device = device

        # Load model
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16
        ).eval().to(device)
        self.score_target = score_target
        
        memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

    def get_combined_scorer(self, prompt, target):
        # Convert prompt to tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        
        # Set up chat format
        self.chat_prefix = self.tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {'role': 'user', 'content': ''}
        ])[:-1]
        self.chat_suffix = [128009, 128006, 78191, 128007, 271]
        perplexity_chat_format = ChatFormat(self.chat_prefix, self.chat_suffix, always_suffix=True)
        
        # Create LLM wrapper
        llm_wrapper = LLMWrapper(
            self.model, 
            self.tokenizer, 
            prompt_tokens=prompt_tokens, 
            chat_format=perplexity_chat_format, 
            device=self.device
        )

        # Create naturalness scorer
        nat_scorer = NaturalnessScorer(self.model, self.tokenizer, chunk_size=50)
        
        # Set up combined scorer with target naturalness
        combined_scorer = CombinedScorer(
            scorers=[nat_scorer],
            weights=[1],
            targets=[self.score_target]
        )

        # Initialize candidate
        init_candidate = Candidate(token_ids=[], score=0.0)
        return llm_wrapper, combined_scorer, init_candidate 