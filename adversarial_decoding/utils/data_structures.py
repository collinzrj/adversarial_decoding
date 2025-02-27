from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Candidate:
    """
    Represents one beam-search candidate:
    - token_ids: list of token IDs
    - score: accumulated score
    - seq_str: decoded string
    - kv_cache, etc.: optional for caching
    """
    token_ids: List[int] = field(default_factory=list)
    score: float = 0.0
    seq_str: str = ""
    # If you're using caching from fastchat:
    kv_cache: Optional[object] = None
    naturalness_kv_cache: Optional[object] = None
    guard_kv_cache: Optional[object] = None
    perplexity: Optional[float] = None
    cos_sim: Optional[float] = None
    naturalness: Optional[float] = None
    llama_guard_score: Optional[float] = None 