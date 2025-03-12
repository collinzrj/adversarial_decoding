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
        max_new_tokens: int = 16,
    ):
        self.llm_wrapper = llm_wrapper
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.tokenizer = llm_wrapper.tokenizer
        self.model = llm_wrapper.model
        self.outputs = self.get_outputs(prompts)


    def get_outputs(self, prompts: List[str]) -> List[str]:
        messages_list = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
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
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        return generation_output.sequences[:, -self.max_new_tokens:]
        
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
            return perplexities, self.tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        else:
            return perplexities

    def _calculate_average_perplexity(self, suffix_list: List[str]) -> float:
        """
        Calculate the average perplexity across all prompts when the 
        given suffix is appended to each prompt.
        
        Args:
            suffix: The suffix to append to each prompt
            
        Returns:
            average_perplexity: The average perplexity across all prompts
        """
        assert type(suffix_list) == list, "suffix_list must be a list"

        full_idx_prompts = []
        for suffix_idx, suffix in enumerate(suffix_list):
            for prompt in self.prompts:
                full_idx_prompts.append([suffix_idx, suffix + ';' + prompt])
        suffix_idx_perplexities = [[] for _ in range(len(suffix_list))]
        
        batch_size = 200
        for idx in tqdm(range(0, len(full_idx_prompts), batch_size), desc="Calculating average perplexity"):
            batch_idx_prompts = full_idx_prompts[idx:idx+batch_size]
            batch_prompts = [prompt[1] for prompt in batch_idx_prompts]
            batch_suffix_idx = [prompt[0] for prompt in batch_idx_prompts]
            batch_perplexities = self._calculate_perplexity(batch_prompts)
            for suffix_idx, perplexity in zip(batch_suffix_idx, batch_perplexities):
                suffix_idx_perplexities[suffix_idx].append(perplexity)
        
        suffix_avg_perplexities = []
        for suffix_idx in range(len(suffix_list)):
            suffix_avg_perplexities.append(np.mean(suffix_idx_perplexities[suffix_idx]))

        return suffix_avg_perplexities
    
    def _calculate_perplexity_given_output(self, suffix) -> float:
        messages_list = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": suffix + ';' + prompt}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in self.prompts
        ]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        encoded_inputs = self.tokenizer.batch_encode_plus(
            messages_list,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_ids = torch.cat([encoded_inputs['input_ids'], self.outputs], dim=1)
        attention_mask = torch.cat([encoded_inputs['attention_mask'], torch.ones_like(self.outputs)], dim=1)

        output = self.model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        logits = logits[:, -self.max_new_tokens-1:]
        
        # Shift the logits and the output tokens for computing loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = self.outputs[:, :].contiguous()
        
        # Calculate loss using cross entropy
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape loss to match the output shape
        loss = loss.view(shift_labels.size())
        
        # Calculate perplexity as exp(average loss)
        # First compute mean loss per sequence
        mean_loss = loss.mean(dim=1)
        
        # Then compute perplexity as exp(mean_loss)
        perplexity = torch.exp(mean_loss).mean().item()
        
        return perplexity

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

        suffixes = [candidate.seq_str for candidate in candidates]
        suffix_avg_perplexities = self._calculate_average_perplexity(suffixes)
        
        for candidate, suffix_avg_perplexity in tqdm(zip(candidates, suffix_avg_perplexities), desc="Calculating perplexity"):
            # Extract the suffix from the candidate
            # For the EntropyDecoding strategy, the candidate's seq_str is just the suffix
            suffix = candidate.seq_str
            # Calculate average perplexity across all prompts
            perplexity = suffix_avg_perplexity
            # Cache the result
            candidate.perplexity = perplexity
            output_perplexity = self._calculate_perplexity_given_output(suffix)
            candidate.extra_info = {
                "output_perplexity": output_perplexity
            }
            score = -perplexity - torch.clamp(torch.tensor(output_perplexity), min=1.6).item()
            scores.append(score)
            
        return scores 