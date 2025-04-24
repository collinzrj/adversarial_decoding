import time
import random
from collections import defaultdict
from typing import List, Optional
import torch
from tqdm import tqdm
from transformers import DynamicCache

from adversarial_decoding.llm.llm_wrapper import LLMWrapper
from adversarial_decoding.scorers.base_scorer import Scorer
from adversarial_decoding.utils.data_structures import Candidate

class BeamSearch:
    """
    A generic beam search that expands each candidate,
    then calls a Scorer to update scores, then keeps the top beam_width.
    """

    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        scorer: Scorer,
        beam_width: int = 10,
        max_steps: int = 30,
        top_k: int = 10,
        top_p: float = 1,
        special_token_ids: Optional[List[int]] = None
    ):
        self.llm = llm_wrapper
        self.scorer = scorer
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.top_k = top_k
        self.top_p = top_p
        self.special_token_ids = special_token_ids or []

    def search(self, initial_candidates: List[Candidate], should_full_sent=True, verbose=False, randomness=False) -> Candidate:
        # Initialize timing dictionary to track performance
        timings = defaultdict(float)
        total_start_time = time.time()
        
        best_full_candidate = None
        candidates = initial_candidates
        if verbose:
            print("Initial candidates Seq", self.llm.tokenizer.decode(candidates[0].token_ids))

        for step in tqdm(range(self.max_steps), desc="Beam Search Steps"):
            step_start_time = time.time()
            
            # Memory check timing
            memory_check_start = time.time()
            torch.cuda.synchronize()
            for device_id in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(device_id)
                if verbose:
                    print(f"Device {device_id} allocated memory: {memory_allocated / 1024**2:.2f} MB")
            timings['memory_check'] += time.time() - memory_check_start
            
            all_candidates: List[Candidate] = []

            # Process all candidates in batch
            batch_start_time = time.time()
            batch_token_ids = [cand.token_ids for cand in candidates]
            
            # Handle KV cache - Check if all candidates have kv_cache attribute and they're not None
            if all(hasattr(cand, 'llm_kv_cache') for cand in candidates):
                batch_llm_kv_cache = [cand.llm_kv_cache for cand in candidates]
                # Only use if at least one candidate has a non-None cache
                if not all(cache is None for cache in batch_llm_kv_cache):
                    pass  # Use the caches as is
                else:
                    batch_llm_kv_cache = None
            else:
                batch_llm_kv_cache = None
            
            batch_top_tokens, next_batch_kv_cache = self.llm.get_next_token_candidates(
                batch_token_ids,
                top_k=self.top_k,
                top_p=self.top_p,
                exclude_ids=self.special_token_ids,
                batch_kv_cache=batch_llm_kv_cache
            )
            timings['get_next_token_candidates'] += time.time() - batch_start_time

            # Build new candidates for each original candidate and its token proposals
            build_candidates_start = time.time()
            for i, cand in enumerate(candidates):
                for token_id, logp in batch_top_tokens[i]:
                    new_seq = cand.token_ids + [token_id]
                    new_candidate = Candidate(
                        token_ids=new_seq,
                        seq_str=self.llm.tokenizer.decode(new_seq),
                        llm_kv_cache=next_batch_kv_cache[i] if (next_batch_kv_cache and i < len(next_batch_kv_cache)) else None,
                        perplexity_kv_cache=cand.perplexity_kv_cache,
                        naturalness_kv_cache=cand.naturalness_kv_cache,
                        guard_kv_cache=cand.guard_kv_cache,
                    )
                    all_candidates.append(new_candidate)
            timings['build_candidates'] += time.time() - build_candidates_start

            # Now compute final "global" scores for these expansions
            scoring_start = time.time()
            naturalness_scores = self.scorer.score_candidates(all_candidates)
            # Update candidate scores with naturalness
            for i, (cand, score) in enumerate(zip(all_candidates, naturalness_scores)):
                # cand.score += score
                # Update with KV caches from scorer if available
                if hasattr(self.scorer, 'last_naturalness_kv_cache') and self.scorer.last_naturalness_kv_cache:
                    cand.naturalness_kv_cache = self.scorer.last_naturalness_kv_cache[i]
                if hasattr(self.scorer, 'last_guard_kv_cache') and self.scorer.last_guard_kv_cache:
                    cand.guard_kv_cache = self.scorer.last_guard_kv_cache[i]
            timings['scoring'] += time.time() - scoring_start

            # Sort by updated score and find best full candidate
            sorting_start = time.time()
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            for cand in all_candidates:
                if cand.token_ids[-1] == self.llm.tokenizer.eos_token_id or cand.seq_str[-1] == '.' or cand.seq_str[-1] == '?' or cand.seq_str[-1] == '!' or cand.seq_str[-1] == '\n':
                    if best_full_candidate is None or cand.score > best_full_candidate.score:
                        best_full_candidate = cand
            timings['sorting_and_best_candidate'] += time.time() - sorting_start

            # Keep top beam_width
            pruning_start = time.time()
            if randomness:
                fixed_length = 5
                if fixed_length > self.beam_width:
                    candidates = all_candidates[:self.beam_width]
                else:
                    candidates = all_candidates[:fixed_length] + random.sample(all_candidates[fixed_length:], self.beam_width - fixed_length)
            else:
                candidates = all_candidates[:self.beam_width]
            del all_candidates
            timings['pruning'] += time.time() - pruning_start
            
            # Calculate total step time
            timings['step_total'] += time.time() - step_start_time
            
            if verbose:
                print(candidates[0])
                print(best_full_candidate)
                
                # Print timing information for this step
                print(f"\nStep {step} timing breakdown:")
                for key, value in timings.items():
                    if key != 'step_total':
                        print(f"  - {key}: {value:.4f}s ({value/timings['step_total']*100:.1f}% of total step time)")
                print(f"  - Total step time: {timings['step_total']:.4f}s")

        # Calculate total search time
        total_time = time.time() - total_start_time
        
        # Print final timing statistics
        if False:
            print("\nFinal timing statistics:")
            print(f"Total search time: {total_time:.4f}s")
            for key, value in timings.items():
                if key != 'step_total':
                    print(f"  - {key}: {value:.4f}s ({value/total_time*100:.1f}% of total time)")
            print(f"  - All steps combined: {timings['step_total']:.4f}s ({timings['step_total']/total_time*100:.1f}% of total time)")

        if best_full_candidate is not None and should_full_sent:
            return best_full_candidate
        else:
            print([[cand.seq_str, cand.cos_sim] for cand in candidates])
            return candidates[0]  # the best final candidate 