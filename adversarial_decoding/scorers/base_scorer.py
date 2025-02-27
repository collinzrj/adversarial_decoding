from typing import List
from adversarial_decoding.utils.data_structures import Candidate

class Scorer:
    """
    Base class for any scoring mechanism. 
    Subclasses must implement `score_candidates`.
    """
    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """
        Given a list of candidates, return a list of float scores (length = len(candidates)).
        """
        raise NotImplementedError 