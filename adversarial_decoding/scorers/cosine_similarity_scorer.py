import torch
from typing import List
from torch.nn.functional import normalize

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.scorers.base_scorer import Scorer

class CosineSimilarityScorer(Scorer):
    """
    Compute embedding for candidate text and 
    measure average similarity with reference embedding(s).
    """

    def __init__(self, embed_model, reference_embeddings: torch.Tensor, prefix_text=''):
        """
        reference_embeddings: shape [1, hidden_dim], or multiple references you average.
        """
        self.embed_model = embed_model
        self.prefix_text = prefix_text
        self.reference_embeddings = reference_embeddings

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        if len(candidates) == 0:
            return []

        # 1) embed each candidate
        texts = [c.seq_str + self.prefix_text for c in candidates]
        
        # Use SentenceTransformer to encode texts
        emb = self.embed_model.encode(
            texts, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # 2) measure similarity with reference
        # (emb) shape: [batch_size, hidden_dim]
        # (self.reference_embeddings) shape: [Nrefs, hidden_dim]
        cos_sim_matrix = torch.matmul(normalize(emb), normalize(self.reference_embeddings).t())
        
        # Average across references
        cos_sims = cos_sim_matrix.mean(dim=1)
        
        # Convert to Python list
        cos_sims = cos_sims.cpu().tolist()

        # 3) update candidate fields
        for i, c in enumerate(candidates):
            c.cos_sim = cos_sims[i]
            c.score += c.cos_sim  # add directly to score
            
        return [c.score for c in candidates] 