import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.utils.utils import file_device, ModelSwitcher
from adversarial_decoding.utils.chat_format import SamplerChatFormat
from adversarial_decoding.strategies.base_strategy import DecodingStrategy
from adversarial_decoding.llm.llm_wrapper import LLMWrapper
from adversarial_decoding.scorers.naturalness_scorer import NaturalnessScorer
from adversarial_decoding.scorers.combined_scorer import CombinedScorer

class LlamaGuardDecoding(DecodingStrategy):
    def __init__(self, need_naturalness, device=file_device):
        self.device = device

        # Load the main model
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16
        ).eval().to(device)
        
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

        # Load sentence encoder
        self.encoder = SentenceTransformer("facebook/contriever", device=device)
        memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

        # Load LlamaGuard model
        llama_guard_model_id = "meta-llama/Llama-Guard-3-8B"
        llama_guard_tokenizer = AutoTokenizer.from_pretrained(llama_guard_model_id)
        llama_guard_model = AutoModelForCausalLM.from_pretrained(
            llama_guard_model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=file_device
        ).eval()
        
        # Set up model switcher and scorers
        model_switcher = ModelSwitcher([self.model, llama_guard_model])
        self.llama_guard_scorer = NaturalnessScorer(
            llama_guard_model, 
            llama_guard_tokenizer, 
            no_cache=False, 
            naturalness=False, 
            model_switcher=model_switcher
        )
        self.naturalness_scorer = NaturalnessScorer(
            self.model, 
            self.tokenizer, 
            no_cache=False
        )
        self.need_naturalness = need_naturalness

    def get_combined_scorer(self, prompt, target):
        # Convert prompt to tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        llm_wrapper = LLMWrapper(
            self.model, 
            self.tokenizer, 
            prompt_tokens=prompt_tokens, 
            chat_format=SamplerChatFormat(), 
            device=self.device
        )

        # Set up combined scorer based on configuration
        if self.need_naturalness:
            combined_scorer = CombinedScorer(
                scorers=[self.llama_guard_scorer, self.naturalness_scorer],
                weights=[1.0, 1.0],
                bounds=[(-torch.inf, torch.inf), (-torch.inf, 0.05)]
            )
        else:
            combined_scorer = CombinedScorer(
                scorers=[self.llama_guard_scorer],
                weights=[1.0],
                bounds=[(-torch.inf, torch.inf)]
            )

        # Initialize candidate
        init_candidate = Candidate(token_ids=prompt_tokens, score=0.0)
        return llm_wrapper, combined_scorer, init_candidate 