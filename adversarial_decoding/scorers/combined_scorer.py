import torch
from typing import List, Optional

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.scorers.base_scorer import Scorer

class CombinedScorer(Scorer):
    """
    Combine multiple scorers by calling them in sequence, possibly with weights.
    """

    def __init__(self, scorers: List[Scorer], weights: Optional[List[float]] = None, 
                 bounds: Optional[List[float]] = None, targets: Optional[List[float]] = None, 
                 skip_steps: Optional[List[int]] = None):
        self.scorers = scorers
        if weights is None:
            weights = [1.0]*len(scorers)
        self.weights = weights
        self.bounds = bounds
        self.targets = targets
        self.skip_steps = skip_steps

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """
        For each scorer, we do a pass on the candidates. Then we combine results.
        The simplest approach is to let each scorer add to candidate.score internally. 
        If you want more advanced weighting logic, do it here.
        """
        # Reset scores for each candidate
        for idx, cand in enumerate(candidates):
            cand.score = 0.0
            
        # Apply each scorer sequentially
        for i, scorer in enumerate(self.scorers):
            # Backup current scores
            old_scores = [c.score for c in candidates]
            
            # Get scores from this scorer
            partial_scores = scorer.score_candidates(candidates)

            # Rescale the partial difference
            for idx, cand in enumerate(candidates):
                # Calculate how much this scorer changed the score
                delta = partial_scores[idx] - old_scores[idx]
                
                # Scale the delta based on our configuration
                if self.targets is not None:
                    scaled_delta = self.weights[i] * -torch.abs(torch.tensor(delta) - self.targets[i]).item()
                elif self.bounds is not None:
                    scaled_delta = self.weights[i] * torch.clamp(torch.tensor(delta), 
                                                                min=self.bounds[i][0], 
                                                                max=self.bounds[i][1]).item()
                else:
                    scaled_delta = self.weights[i] * delta
                
                # Skip early steps if configured
                if self.skip_steps is not None and self.skip_steps[i] is not None:
                    if len(cand.token_ids) <= self.skip_steps[i]:
                        scaled_delta = 0
                
                # Undo original score change and add scaled version
                cand.score = old_scores[idx] + scaled_delta
        return [c.score for c in candidates]