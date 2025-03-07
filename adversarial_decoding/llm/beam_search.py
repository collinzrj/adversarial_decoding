from typing import List, Optional
from tqdm import tqdm
import torch

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.scorers.base_scorer import Scorer
from adversarial_decoding.llm.llm_wrapper import LLMWrapper

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

    def search(self, initial_candidates: List[Candidate], should_full_sent=True, verbose=False) -> Candidate:
        best_full_candidate = None
        candidates = initial_candidates
        if verbose:
            print("Initial candidates Seq", self.llm.tokenizer.decode(candidates[0].token_ids))

        for step in tqdm(range(self.max_steps), desc="Beam Search Steps"):
            torch.cuda.synchronize()
            for device_id in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(device_id)
                if verbose:
                    print(f"Device {device_id} allocated memory: {memory_allocated / 1024**2:.2f} MB")
            all_candidates: List[Candidate] = []

            # Expand each candidate
            for cand in candidates:
                # get next token proposals
                top_tokens = self.llm.get_next_token_candidates(
                    [cand.token_ids], 
                    top_k=self.top_k,
                    top_p=self.top_p,
                    exclude_ids=self.special_token_ids
                )

                # build new candidates
                for token_id, logp in top_tokens:
                    new_seq = cand.token_ids + [token_id]
                    new_candidate = Candidate(
                        token_ids=new_seq,
                        seq_str=self.llm.tokenizer.decode(new_seq),
                        kv_cache=cand.kv_cache,
                        naturalness_kv_cache=cand.naturalness_kv_cache,
                        guard_kv_cache=cand.guard_kv_cache,
                        score=cand.score + logp  # partial increment
                    )
                    all_candidates.append(new_candidate)

            # Now compute final "global" scores for these expansions
            self.scorer.score_candidates(all_candidates)

            # Sort by updated score
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            for cand in all_candidates:
                if cand.token_ids[-1] == self.llm.tokenizer.eos_token_id or cand.seq_str[-1] == '.' or cand.seq_str[-1] == '?' or cand.seq_str[-1] == '!' or cand.seq_str[-1] == '\n':
                    if best_full_candidate is None or cand.score > best_full_candidate.score:
                        best_full_candidate = cand

            # Keep top beam_width
            candidates = all_candidates[:self.beam_width]
            del all_candidates
            if verbose:
                print(candidates[0])
                print(best_full_candidate)

        if best_full_candidate is not None and should_full_sent:
            return best_full_candidate
        else:
            return candidates[0]  # the best final candidate 