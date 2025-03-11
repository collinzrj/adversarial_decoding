from typing import List, Optional
import torch
import numpy as np
from tqdm import tqdm

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.scorers.base_scorer import Scorer
from adversarial_decoding.llm.llm_wrapper import LLMWrapper

class EntropyScorer(Scorer):
    """
    Scores candidates based on the perplexity (entropy) of their continuations.
    Lower entropy means the model is more certain about its output.
    
    This scorer evaluates a suffix across multiple prompts to find a universal
    suffix that reduces output entropy for many different prompts.
    
    Attributes:
        llm_wrapper: LLM model wrapper
        prompts: List of prompts to evaluate the suffix on
        maximize: Whether to maximize or minimize the score (default is to minimize entropy)
        max_new_tokens: Maximum number of new tokens to generate for perplexity calculation
    """
    
    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        prompts: List[str],
        maximize: bool = False,
        max_new_tokens: int = 16,
    ):
        self.llm_wrapper = llm_wrapper
        self.prompts = prompts
        self.maximize = maximize
        self.max_new_tokens = max_new_tokens
        self.tokenizer = llm_wrapper.tokenizer
        self.model = llm_wrapper.model
        
    def _calculate_perplexity(self, full_prompts: List[str], return_completions: bool = False) -> float:
        """
        Calculate perplexity for a full prompt (prompt + suffix) by generating 
        continuations and measuring the negative log likelihood.
        
        Args:
            full_prompt: The complete prompt string (original prompt + suffix)
            
        Returns:
            perplexity: The perplexity score (lower = more certain)
        """
        # Tokenize the full prompt
        messages_list = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for full_prompt in full_prompts
        ]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer.batch_encode_plus(
            messages_list,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate continuation with scores
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids['input_ids'],
                attention_mask=input_ids['attention_mask'],
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        # Extract scores for the generated tokens
        scores = torch.stack(generation_output.scores)
        
        # Calculate perplexity for each prompt in the batch
        batch_size = input_ids['input_ids'].shape[0]
        perplexities = []
        
        for i in range(batch_size):
            # Get scores for this sequence in the batch
            prompt_scores = scores[:, i, :]
            
            # Get the generated token IDs for this sequence
            generated_ids = generation_output.sequences[i, input_ids['input_ids'].shape[1]:]
            
            # Calculate log probabilities for each token
            token_log_probs = []
            
            for j, token_id in enumerate(generated_ids):
                if token_id == self.tokenizer.eos_token_id:
                    break
                    
                if j < len(prompt_scores):  # Ensure we don't go out of bounds
                    log_prob = torch.nn.functional.log_softmax(prompt_scores[j], dim=-1)[token_id]
                    token_log_probs.append(log_prob.item())
                
            # Calculate perplexity as exp(average negative log likelihood)
            avg_neg_log_likelihood = -sum(token_log_probs) / len(token_log_probs)
            perplexity = np.exp(avg_neg_log_likelihood)
            perplexities.append(perplexity)
        
        # Return average perplexity across the batch
        if return_completions:
            return np.mean(perplexities), self.tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        else:
            return np.mean(perplexities)

    def _calculate_average_perplexity(self, suffix: str) -> float:
        """
        Calculate the average perplexity across all prompts when the 
        given suffix is appended to each prompt.
        
        Args:
            suffix: The suffix to append to each prompt
            
        Returns:
            average_perplexity: The average perplexity across all prompts
        """
        perplexities = []
        
        for idx in range(0, len(self.prompts), 10): 
            prompts = self.prompts[idx:idx+10]
            perplexity = self._calculate_perplexity([suffix + ';' + prompt for prompt in prompts])
            perplexities.append(perplexity)
            
        # Return the average perplexity across all prompts
        return np.mean(perplexities) if perplexities else 100.0

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """
        Score candidates based on the average perplexity of their continuations
        across all prompts when the candidate suffix is appended.
        
        Args:
            candidates: List of candidate sequences (representing different suffixes)
            
        Returns:
            List of scores (lower = better if maximize=False)
        """
        scores = []
        
        for candidate in tqdm(candidates, desc="Calculating perplexity"):
            # Extract the suffix from the candidate
            # For the EntropyDecoding strategy, the candidate's seq_str is just the suffix
            suffix = candidate.seq_str
            
            # Check if we already calculated perplexity for this candidate
            if candidate.perplexity is not None:
                perplexity = candidate.perplexity
            else:
                # Calculate average perplexity across all prompts
                perplexity = self._calculate_average_perplexity(suffix)
                # Cache the result
                candidate.perplexity = perplexity
            
            # Lower perplexity is better, so we use negative perplexity as the score if maximizing
            score = -perplexity
            scores.append(score)
            
        return scores 