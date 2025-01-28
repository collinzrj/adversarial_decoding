import torch
import torch.nn.functional as F
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # If you rely on fastchat for DynamicCache, import it here:
    # from fastchat.model import DynamicCache
    # For demonstration, we comment it out; define a mock if needed.
)

########################################
# Utility Functions & Timer
########################################

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9):
    """
    Given a 1D tensor of `logits`, return the indices of tokens that
    satisfy both the Top-K and Top-P filtering constraints.
    """
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, top_k)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_cutoff = torch.sum(cumulative_probs <= top_p).item()
    if top_p_cutoff < 1:
        top_p_cutoff = 1
    top_p_indices = sorted_indices[:top_p_cutoff]
    top_k_set = set(topk_indices.tolist())
    top_p_set = set(top_p_indices.tolist())
    intersection_set = top_k_set.intersection(top_p_set)

    if len(intersection_set) == 0:
        raise ValueError("No tokens satisfy both Top-K and Top-P constraints.")

    intersection_indices = torch.tensor(list(intersection_set), dtype=torch.long)
    intersection_probs = probs[intersection_indices]
    sorted_intersection_probs, sorted_order = torch.sort(intersection_probs, descending=True)
    intersection_indices = intersection_indices[sorted_order.cpu()]
    return intersection_indices


class MyTimer:
    """Simple timer to measure code sections."""
    def __init__(self):
        self.timer_dict = {}

    def start(self, name):
        if name in self.timer_dict:
            self.timer_dict[name][1] = time.time()
        else:
            self.timer_dict[name] = [0, time.time()]
    
    def stop(self, name):
        self.timer_dict[name][0] += time.time() - self.timer_dict[name][1]
        self.timer_dict[name][1] = None

    def display(self):
        for name, (total_time, _) in self.timer_dict.items():
            print(f"[{name}] {total_time:.4f}s")


########################################
# Data Structures for Beam Search
########################################

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
    perplexity: Optional[float] = None
    cos_sim: Optional[float] = None
    naturalness: Optional[float] = None


########################################
# Scoring Classes
########################################

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


class PerplexityScorer(Scorer):
    """
    Example scorer that uses an LLM to compute perplexity for the entire sequence (or portion).
    Negative perplexity is often used as the final score.
    """

    def __init__(self, llm, tokenizer, chat_format, chunk_size=10, ignore_tokens_num=1):
        self.llm = llm
        self.tokenizer = tokenizer
        self.chat_format = chat_format
        self.chunk_size = chunk_size
        self.ignore_tokens_num = ignore_tokens_num

    def compute_perplexity_batch(
        self,
        tokens_batch: List[List[int]],
        batch_kv_cache: Optional[List[object]],
        next_cache_seq_len: int
    ):
        """
        Implement your perplexity calculation here.
        Returns (ppl_tensor, new_kv_cache_list).
        """
        # For example, replicate your old compute_perplexity_batch logic:
        #   1) Convert tokens_batch to torch tensor
        #   2) Pass through the self.llm
        #   3) Return perplexities and updated kv_cache
        # This is just a stub:
        batch_size = len(tokens_batch)
        ppl_tensor = torch.ones(batch_size)
        new_kv_cache_list = [None]*batch_size
        return ppl_tensor, new_kv_cache_list

    def compute_perplexity(
        self,
        tokens_batch: List[List[int]],
        batch_kv_cache: Optional[List[object]],
        next_cache_seq_len: int
    ):
        """
        Splits tokens_batch into chunks, calls `compute_perplexity_batch`,
        then concatenates results.
        """
        all_ppls = []
        all_kvs = []
        for i in range(0, len(tokens_batch), self.chunk_size):
            chunk = tokens_batch[i:i+self.chunk_size]
            if batch_kv_cache:
                chunk_kv = batch_kv_cache[i:i+self.chunk_size]
            else:
                chunk_kv = None
            ppl_tensor, kv_tensor = self.compute_perplexity_batch(
                chunk, chunk_kv, next_cache_seq_len
            )
            all_ppls.append(ppl_tensor)
            all_kvs.extend(kv_tensor)
        return torch.cat(all_ppls), all_kvs

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """
        Batch-compute perplexity for all candidates, store it in candidate.perplexity,
        and return negative perplexity as a score (example approach).
        """
        if len(candidates) == 0:
            return []

        # 1) Prepare input for each candidate
        tokens_batch = []
        kv_cache_batch = []
        # For demonstration, we skip any advanced prefix. Modify as needed.
        for c in candidates:
            # If you're appending something else, do it here:
            full_tokens = c.token_ids
            tokens_batch.append(full_tokens)
            kv_cache_batch.append(c.kv_cache)  # might be None

        # 2) For perplexity, you need the length of the prefix if using caches:
        next_cache_seq_len = len(tokens_batch[0])  # or your chat_format logic

        # 3) Compute perplexities
        ppl_values, new_kv_caches = self.compute_perplexity(
            tokens_batch,
            kv_cache_batch,
            next_cache_seq_len
        )

        # 4) Fill in candidate metadata
        scores = []
        for i, cand in enumerate(candidates):
            cand.perplexity = ppl_values[i].item()
            cand.kv_cache = new_kv_caches[i]
            # For instance, we want lower perplexity => higher score => negative sign:
            sc = -cand.perplexity
            cand.score += sc  # or set cand.score = sc, depending on your logic
            scores.append(cand.score)
        return scores


class NaturalnessScorer(Scorer):
    """
    Example scorer for “naturalness” of each candidate. 
    This is a stub that you would adapt to your logic.
    """

    def __init__(self, llm, tokenizer, chunk_size=10):
        self.llm = llm
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def compute_naturalness_batch(
        self,
        tokens_batch: List[List[int]],
        batch_kv_cache: Optional[List[object]]
    ):
        """
        Return tensor of naturalness scores and updated kv caches.
        """
        # Example stub:
        batch_size = len(tokens_batch)
        naturalness_scores = torch.zeros(batch_size)
        new_kv_caches = [None]*batch_size
        return naturalness_scores, new_kv_caches

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        if len(candidates) == 0:
            return []

        tokens_batch = []
        kv_cache_batch = []
        for c in candidates:
            tokens_batch.append(c.token_ids)
            kv_cache_batch.append(c.naturalness_kv_cache)

        nat_scores, new_kvs = self.compute_naturalness_batch(tokens_batch, kv_cache_batch)

        final_scores = []
        for i, c in enumerate(candidates):
            c.naturalness = nat_scores[i].item()
            c.naturalness_kv_cache = new_kvs[i]
            # Suppose we want to *add* this to the candidate's overall score:
            c.score += c.naturalness
            final_scores.append(c.score)

        return final_scores


class CosineSimilarityScorer(Scorer):
    """
    Example: compute embedding for candidate text, 
    measure average similarity with reference embedding(s).
    """

    def __init__(self, embed_model, reference_embeddings: torch.Tensor):
        """
        reference_embeddings: shape [1, hidden_dim], or multiple references you average.
        """
        self.embed_model = embed_model
        self.reference_embeddings = reference_embeddings
        if len(reference_embeddings.shape) == 1:
            self.reference_embeddings = reference_embeddings.unsqueeze(0)

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        if len(candidates) == 0:
            return []

        # 1) embed each candidate
        texts = [c.seq_str for c in candidates]
        # If using SentenceTransformer:
        emb = self.embed_model.encode(
            texts, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        # 2) measure similarity with reference
        # For demonstration, we just take the mean if multiple references:
        # (emb) shape: [batch_size, hidden_dim]
        # (self.reference_embeddings) shape: [Nrefs, hidden_dim]
        # We can just do a mean of references
        ref_emb = self.reference_embeddings.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        # Compute cos sim
        cos_sims = torch.mm(emb, ref_emb.transpose(0,1)).squeeze(dim=-1)
        # cos_sims shape: [batch_size]
        cos_sims = cos_sims.cpu().tolist()

        # 3) fill in candidate fields
        for i, c in enumerate(candidates):
            c.cos_sim = cos_sims[i]
            c.score += c.cos_sim  # e.g. add directly
        return [c.score for c in candidates]


class CombinedScorer(Scorer):
    """
    Combine multiple scorers by calling them in sequence, possibly with weights.
    """

    def __init__(self, scorers: List[Scorer], weights: Optional[List[float]] = None):
        self.scorers = scorers
        if weights is None:
            weights = [1.0]*len(scorers)
        self.weights = weights

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """
        For each scorer, we do a pass on the candidates. Then we combine results.
        The simplest approach is to let each scorer add to candidate.score internally. 
        If you want more advanced weighting logic, do it here.
        """
        # We'll do an approach where each scorer returns a list of partial scores,
        # and we scale them by weights[i]. 
        # But because the scorers might internally add to candidate.score, 
        # we can handle it differently.
        for i, scorer in enumerate(self.scorers):
            # We can back up the old scores first
            old_scores = [c.score for c in candidates]
            partial_scores = scorer.score_candidates(candidates)

            # Rescale the partial difference
            # For example, if the scorer has added +X to candidate.score,
            # we can multiply it by self.weights[i].
            for idx, cand in enumerate(candidates):
                # The difference that the scorer contributed
                delta = partial_scores[idx] - old_scores[idx]
                # Scale that
                scaled_delta = self.weights[i] * delta
                # Undo the original addition and re-add scaled
                cand.score = old_scores[idx] + scaled_delta

        return [c.score for c in candidates]


########################################
# Language Model Wrapper
########################################

class LLMWrapper:
    """
    Simple wrapper to run forward pass on an LLM,
    returning top-k/p filtered next-token proposals.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def get_next_token_candidates(
        self,
        input_ids: torch.Tensor,
        top_k: int = 10,
        top_p: float = 0.9,
        exclude_ids: Optional[List[int]] = None
    ):
        """
        1) forward pass on input_ids
        2) top-k/top-p filtering
        3) return list of (token_id, log_probability)
        """
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

        if exclude_ids:
            log_probs[:, exclude_ids] = -1e10

        # do top-k/top-p filtering on logits, then gather their log_probs
        filtered_indices = top_k_top_p_filtering(logits[0], top_k, top_p)
        filtered_log_probs = log_probs[0, filtered_indices]

        # sort descending
        sorted_vals, sorted_idx = torch.sort(filtered_log_probs, descending=True)
        sorted_indices = filtered_indices[sorted_idx.cpu()]

        # return as python lists of (token_id, log_prob)
        return list(zip(sorted_indices.tolist(), sorted_vals.tolist()))


########################################
# Chat Format
########################################

class ChatFormat:
    def __init__(self, chat_prefix: List[int], chat_suffix: List[int]):
        self.chat_prefix = chat_prefix
        self.chat_suffix = chat_suffix

    def prepare_input(self, prompt_tokens: List[int], adv_tokens: List[int]) -> List[int]:
        return self.chat_prefix + prompt_tokens + adv_tokens + self.chat_suffix

    def prepare_prefix_input(self, prompt_tokens: List[int], adv_tokens: List[int]) -> List[int]:
        return self.chat_prefix + prompt_tokens + adv_tokens


########################################
# Beam Search
########################################

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
        top_p: float = 0.9999,
        special_token_ids: Optional[List[int]] = None
    ):
        self.llm = llm_wrapper
        self.scorer = scorer
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.top_k = top_k
        self.top_p = top_p
        self.special_token_ids = special_token_ids or []

    def search(self, initial_candidates: List[Candidate]) -> Candidate:
        candidates = initial_candidates

        for step in tqdm(range(self.max_steps), desc="Beam Search Steps"):
            all_candidates = []

            # Expand each candidate
            for cand in candidates:
                input_ids = torch.tensor([cand.token_ids], dtype=torch.long)
                # get next token proposals
                top_tokens = self.llm.get_next_token_candidates(
                    input_ids, 
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
                        score=cand.score + logp  # partial increment
                    )
                    all_candidates.append(new_candidate)

            # Now compute final "global" scores for these expansions
            self.scorer.score_candidates(all_candidates)

            # Sort by updated score
            all_candidates.sort(key=lambda c: c.score, reverse=True)

            # Keep top beam_width
            candidates = all_candidates[:self.beam_width]
            print(candidates[0])

        return candidates[0]  # the best final candidate


########################################
# Main Orchestrator (JailbreakDecoding)
########################################

class JailbreakDecoding:
    """
    Demonstration of how you might assemble everything:
    - load your models
    - create the scorers
    - run the beam search
    """

    def __init__(self, device="cuda"):
        self.device = device

        # Example model name
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Example second model for embeddings
        self.encoder = SentenceTransformer("facebook/contriever", device=device)

        # Example chat prefix/suffix
        # (From your original code, you can adapt as needed.)
        self.chat_prefix = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': ''}]
        )[:-1]
        self.chat_suffix = [128009, 128006, 78191, 128007, 271]

        self.chat_format = ChatFormat(self.chat_prefix, self.chat_suffix)
        self.llm_wrapper = LLMWrapper(self.model, self.tokenizer, device=device)

    def compute_doc_embs(self, documents: List[str]):
        return self.encoder.encode(documents, convert_to_tensor=True, normalize_embeddings=True)

    def run_decoding(
        self,
        prompt: str,
        target: str,
        beam_width=10,
        max_steps=30,
        top_k=10,
        top_p=0.95
    ) -> str:
        """
        Orchestrates a beam search using combined scorers.
        """
        # 1) Convert prompt to tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)

        # 2) Setup scorers
        # Perplexity
        # perplexity_scorer = PerplexityScorer(
        #     llm=self.model, 
        #     tokenizer=self.tokenizer, 
        #     chat_format=self.chat_format
        # )
        # Naturalness
        nat_scorer = NaturalnessScorer(self.model, self.tokenizer)
        # Cosine similarity
        target_emb = self.compute_doc_embs([target])
        cos_scorer = CosineSimilarityScorer(self.encoder, target_emb)

        # Combine them with weights
        combined_scorer = CombinedScorer(
            scorers=[nat_scorer, cos_scorer],
            weights=[1.0, 100.0, 0.5]  # adjust these as you wish
        )

        # 3) Initialize beam search with the combined scorer
        beam_searcher = BeamSearch(
            llm_wrapper=self.llm_wrapper,
            scorer=combined_scorer,
            beam_width=beam_width,
            max_steps=max_steps,
            top_k=top_k,
            top_p=top_p,
            special_token_ids=[self.tokenizer.eos_token_id]
        )

        # 4) Create an initial candidate
        # initial_tokens = self.chat_format.prepare_input(prompt_tokens, [])
        init_candidate = Candidate(token_ids=prompt_tokens, score=0.0)

        # 5) Run beam search
        best_candidate = beam_searcher.search([init_candidate])

        # 6) Return or print
        print("BEST TOKENS:", best_candidate.token_ids)
        print("BEST TEXT:", best_candidate.seq_str)
        return best_candidate.seq_str


########################################
# EXAMPLE USAGE
########################################

if __name__ == "__main__":
    jailbreak = JailbreakDecoding(device="cuda")

    prompt_text = "What is Spotify? Please answer this question politely."
    target_text = "I hate spotify."  # example
    result = jailbreak.run_decoding(
        prompt=prompt_text,
        target=target_text,
        beam_width=5,
        max_steps=15,
        top_k=10,
        top_p=0.90
    )
    print("Final Decoding:", result)
