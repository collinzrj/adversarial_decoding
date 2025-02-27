import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.utils.utils import file_device, second_device
from adversarial_decoding.utils.chat_format import ChatFormat, SamplerChatFormat
from adversarial_decoding.strategies.base_strategy import DecodingStrategy
from adversarial_decoding.llm.llm_wrapper import LLMWrapper
from adversarial_decoding.scorers.perplexity_scorer import PerplexityScorer, PerplexityDifferentLLMScorer
from adversarial_decoding.scorers.naturalness_scorer import NaturalnessScorer
from adversarial_decoding.scorers.combined_scorer import CombinedScorer

class JailbreakDecoding(DecodingStrategy):
    def __init__(self, target_model_name, should_natural=True, should_guard=False, device=file_device):
        self.device = device

        # Load main model
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.model_name = model_name
        self.target_model_name = target_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().to(device)
        
        memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

        # Set up chat formatting
        self.chat_prefix = self.tokenizer.apply_chat_template([{'role': 'user', 'content': ''}])[:-1]
        self.chat_suffix = [128009, 128006, 78191, 128007, 271]
        self.chat_format = ChatFormat(self.chat_prefix, self.chat_suffix)
        self.should_natural = should_natural

        # Load target model if different from main model
        if self.target_model_name == self.model_name:
            self.target_model = self.model
            self.target_tokenizer = self.tokenizer
        else:
            if 'gemma' in self.target_model_name:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.target_model_name, 
                    torch_dtype=torch.float16
                ).eval().to(self.device)
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.target_model_name, 
                    torch_dtype=torch.bfloat16
                ).eval().to(self.device)
                
            self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)

        # Load LlamaGuard if needed
        self.should_guard = should_guard
        if should_guard:
            llama_guard_model_id = "meta-llama/Llama-Guard-3-8B"
            self.llama_guard_tokenizer = AutoTokenizer.from_pretrained(llama_guard_model_id)
            self.llama_guard_model = AutoModelForCausalLM.from_pretrained(
                llama_guard_model_id, 
                torch_dtype=torch.bfloat16,
                device_map=second_device
            ).eval()
            print("llama guard model device", self.llama_guard_model.device)

    def get_combined_scorer(self, prompt, target):
        # Create tokenized prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        
        # Set up chat format
        self.chat_prefix = self.tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {'role': 'user', 'content': ''}
        ])[:-1]
        self.chat_suffix = [128009, 128006, 78191, 128007, 271]
        perplexity_chat_format = ChatFormat(self.chat_prefix, self.chat_suffix)
        
        # Create LLM wrapper
        llm_wrapper = LLMWrapper(
            self.model, 
            self.tokenizer, 
            prompt_tokens=prompt_tokens, 
            chat_format=perplexity_chat_format, 
            device=self.device
        )

        # Set up perplexity scorer
        if self.target_model_name == self.model_name:
            print("use perplexity scorer")
            self.perplexity_scorer = PerplexityScorer(
                llm=self.model, 
                tokenizer=self.tokenizer,
                chat_format=perplexity_chat_format,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                chunk_size=50
            )
        else:
            print("use different llm scorer")
            self.perplexity_scorer = PerplexityDifferentLLMScorer(
                self.target_model,
                self.target_tokenizer,
                prompt,
                target,
            )
            
        # Set up scorers list and configuration
        scorers = [self.perplexity_scorer]
        weights = [-10.0]
        bounds = [(-torch.inf, torch.inf)]
        skip_steps = [None]
        
        # Add naturalness scorer if requested
        if self.should_natural:
            scorers.append(NaturalnessScorer(self.model, self.tokenizer, no_cache=False, chunk_size=50))
            weights.append(100.0)
            bounds.append((-torch.inf, 0.05))
            skip_steps.append(None)
            
        # Add LlamaGuard scorer if requested
        if self.should_guard:
            self.llama_guard_scorer = NaturalnessScorer(
                self.llama_guard_model, 
                self.llama_guard_tokenizer, 
                prompt_prefix_tokens=prompt_tokens, 
                no_cache=False, 
                naturalness=False
            )
            scorers.append(self.llama_guard_scorer)
            weights.append(300.0)
            bounds.append((-torch.inf, 0.005))
            skip_steps.append(30)

        # Create combined scorer
        combined_scorer = CombinedScorer(scorers, weights, bounds, skip_steps=skip_steps)
        
        # Initial candidate
        init_candidate = Candidate(token_ids=[], score=0.0)
        
        return llm_wrapper, combined_scorer, init_candidate 