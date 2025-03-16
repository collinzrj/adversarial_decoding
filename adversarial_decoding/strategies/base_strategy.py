from typing import Tuple
from tqdm import tqdm

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.scorers.base_scorer import Scorer
from adversarial_decoding.llm.llm_wrapper import LLMWrapper
from adversarial_decoding.llm.beam_search import BeamSearch

class DecodingStrategy:
    """
    Base class for all decoding strategies.
    All strategies must implement get_combined_scorer.
    """

    def get_combined_scorer(self, prompt, target):
        """
        Create and return the LLMWrapper, combined scorer, and initial candidate
        for the specific strategy.
        """
        raise NotImplementedError

    def run_decoding(
        self,
        prompt: str,
        target: str,
        beam_width=10,
        max_steps=30,
        top_k=10,
        top_p=0.95,
        should_full_sent=True,
        verbose=False,
        randomness=False
    ) -> Candidate:
        """
        Orchestrates a beam search using combined scorers.
        """
        llm_wrapper, combined_scorer, init_candidate = self.get_combined_scorer(prompt, target)
        self.llm_wrapper = llm_wrapper
        self.combined_scorer = combined_scorer

        beam_searcher = BeamSearch(
            llm_wrapper=self.llm_wrapper,
            scorer=combined_scorer,
            beam_width=beam_width,
            max_steps=max_steps,
            top_k=top_k,
            top_p=top_p,
            special_token_ids=[self.tokenizer.eos_token_id]
        )
        best_candidate = beam_searcher.search([init_candidate], should_full_sent=should_full_sent, verbose=verbose, randomness=randomness)

        # print("BEST TOKENS:", best_candidate.token_ids)
        # print("BEST TEXT:", best_candidate.seq_str)
        return best_candidate 