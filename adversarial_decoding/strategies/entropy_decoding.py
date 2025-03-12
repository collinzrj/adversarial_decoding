from typing import Tuple, List
import torch
import numpy as np
from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.scorers.base_scorer import Scorer
from adversarial_decoding.scorers.entropy_scorer import EntropyScorer
from adversarial_decoding.scorers.combined_scorer import CombinedScorer
from adversarial_decoding.llm.llm_wrapper import LLMWrapper
from adversarial_decoding.strategies.base_strategy import DecodingStrategy
from adversarial_decoding.utils.chat_format import ChatFormat

from transformers import AutoModelForCausalLM, AutoTokenizer

class EntropyDecoding(DecodingStrategy):
    """
    Decoding strategy that aims to find a universal suffix which, when appended 
    to various prompts, leads to model outputs with lower perplexity/entropy.
    
    The strategy uses beam search to find the optimal suffix that minimizes the 
    average perplexity of model continuations across multiple prompts.
    """
    
    def __init__(self, target_model_name, max_new_tokens, device='cuda'):
        self.device = device
        self.max_new_tokens = max_new_tokens

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
        self.chat_suffix = []
        self.chat_format = ChatFormat(self.chat_prefix, self.chat_suffix)
        
    def get_combined_scorer(self, prompts: List[str], target: str) -> Tuple[LLMWrapper, Scorer, Candidate]:
        """
        Create and return the LLM wrapper, combined scorer, and initial candidate.
        
        Args:
            prompts: List of prompts to evaluate the suffix on
            target: Target string (not used for entropy decoding)
            
        Returns:
            Tuple of (LLM wrapper, combined scorer, initial candidate)
        """

        # Initialize the LLM wrapper
        self.llm_wrapper = LLMWrapper(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_tokens=self.tokenizer.encode("You are very certain.", add_special_tokens=False),
            chat_format=self.chat_format,
            device=self.device
        )
        
        # Initialize the entropy scorer with all prompts
        entropy_scorer = EntropyScorer(
            llm_wrapper=self.llm_wrapper,
            prompts=prompts,
            max_new_tokens=self.max_new_tokens
        )
        self.entropy_scorer = entropy_scorer
        
        # Create a combined scorer (in this case it's just the entropy scorer)
        combined_scorer = CombinedScorer(
            scorers=[entropy_scorer],
            weights=[1.0]
        )
        
        # Create an empty initial candidate (a blank suffix)
        # We'll start with an empty suffix and let beam search find the optimal one
        empty_candidate = Candidate(
            token_ids=[],  # Empty token sequence
            seq_str="",    # Empty string
            score=0.0
        )
        
        return self.llm_wrapper, combined_scorer, empty_candidate
        
    def run_decoding(
        self,
        prompts: List[str],
        target: str = "",
        beam_width: int = 10,
        max_steps: int = 30,
        top_k: int = 10,
        top_p: float = 0.95,
        should_full_sent: bool = True,
        verbose: bool = False
    ) -> Candidate:
        """
        Override the run_decoding method to accept a list of prompts.
        This finds a universal suffix that works well across all prompts.
        
        Args:
            prompts: List of prompts to optimize the suffix for
            target: Target string (not used for entropy decoding)
            beam_width: Beam width for beam search
            max_steps: Maximum number of steps for beam search
            top_k: Top-k parameter for sampling
            top_p: Top-p parameter for nucleus sampling
            should_full_sent: Whether to complete the generation to a full sentence
            verbose: Whether to print verbose output during beam search
            
        Returns:
            The best candidate representing the optimal suffix
        """
        # Use the updated get_combined_scorer method with multiple prompts
        llm_wrapper, combined_scorer, init_candidate = self.get_combined_scorer(prompts, target)
        self.llm_wrapper = llm_wrapper
        self.combined_scorer = combined_scorer
        self.tokenizer = llm_wrapper.tokenizer
        
        # Import here to avoid circular imports
        from adversarial_decoding.llm.beam_search import BeamSearch
        
        beam_searcher = BeamSearch(
            llm_wrapper=self.llm_wrapper,
            scorer=combined_scorer,
            beam_width=beam_width,
            max_steps=max_steps,
            top_k=top_k,
            top_p=top_p,
            special_token_ids=[self.tokenizer.eos_token_id]
        )
        
        best_candidate = beam_searcher.search([init_candidate], should_full_sent=should_full_sent, verbose=verbose)
        return best_candidate
        
    def test_suffix(self, prompts: List[str], suffix: str, max_new_tokens: int = 16) -> float:
        """
        Test the average perplexity of a model's outputs across multiple prompts when
        the given suffix is appended to each prompt.
        
        Args:
            prompts: List of prompts to test
            suffix: Suffix to append to each prompt
            max_new_tokens: Number of tokens to generate for perplexity calculation
            
        Returns:
            avg_perplexity: Average perplexity across all prompts (lower is better)
        """
        # Initialize the entropy scorer with all prompts
        entropy_scorer = EntropyScorer(
            llm_wrapper=self.llm_wrapper,
            prompts=prompts,
            max_new_tokens=max_new_tokens
        )
        
        # Calculate average perplexity across all prompts
        perplexities = entropy_scorer._calculate_average_perplexity([suffix])
        return np.mean(perplexities)
        
    def test_suffix_individual(self, prompts: List[str], suffix: str) -> List[float]:
        """
        Test the perplexity of a model's outputs for each individual prompt when
        the given suffix is appended.
        
        Args:
            prompts: List of prompts to test
            suffix: Suffix to append to each prompt
            max_new_tokens: Number of tokens to generate for perplexity calculation
            
        Returns:
            perplexities: List of perplexity values for each prompt
        """
        perplexities = []
        completions_list = []
        
        for prompt in prompts:
            full_prompt = f"{suffix};{prompt}" if suffix.strip() else prompt
            full_prompt_perplexities, completions = self.entropy_scorer._calculate_perplexity([full_prompt], return_completions=True)
            perplexities.append(np.mean(full_prompt_perplexities))
            completions_list.append(completions[0])
        return perplexities, completions_list