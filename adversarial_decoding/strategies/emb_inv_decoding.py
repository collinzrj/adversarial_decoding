import torch
import json
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversarial_decoding.utils.data_structures import Candidate
from adversarial_decoding.utils.utils import file_device, highest_avg_cos_sim, compute_doc_embs
from adversarial_decoding.utils.chat_format import ChatFormat, SamplerChatFormat
from adversarial_decoding.strategies.base_strategy import DecodingStrategy
from adversarial_decoding.llm.llm_wrapper import LLMWrapper
from adversarial_decoding.scorers.naturalness_scorer import NaturalnessScorer
from adversarial_decoding.scorers.cosine_similarity_scorer import CosineSimilarityScorer
from adversarial_decoding.scorers.combined_scorer import CombinedScorer

class EmbInvDecoding(DecodingStrategy):
    def __init__(self, encoder=None, device=file_device, should_natural=True, repetition_penalty: float = 1.0):
        self.device = device
        self.should_natural = should_natural

        # Load model
        # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16
        ).eval().to(device)
        
        memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

        # Initialize or load encoder
        if encoder is None:
            self.encoder = SentenceTransformer("thenlper/gte-base", device=device)
        else:
            self.encoder = encoder
            
        memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

        # Set up chat formats
        self.chat_prefix = self.tokenizer.apply_chat_template([{'role': 'user', 'content': ''}])[:-1]
        self.chat_suffix = self.tokenizer.apply_chat_template([{'role': 'user', 'content': ''}], add_generation_prompt=True)[len(self.chat_prefix):]
        self.chat_format = ChatFormat(self.chat_prefix, self.chat_suffix, always_suffix=True)
        self.repetition_penalty = repetition_penalty
    
    def get_combined_scorer(self, prompt, target):
        # Convert prompt to tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if type(target) == str:
            self.reference_embeddings = self.encoder.encode(target, add_special_tokens=False, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)
        else:
            self.reference_embeddings = target
        
        # Create LLM wrapper
        llm_wrapper = LLMWrapper(
            self.model, 
            self.tokenizer, 
            prompt_tokens=prompt_tokens, 
            chat_format=self.chat_format, 
            device=self.device,
            repetition_penalty=self.repetition_penalty
        )
        # Set up naturalness scorer if requested
        if self.should_natural:
            nat_scorer = NaturalnessScorer(
                self.model, 
                self.tokenizer, 
                no_cache=False, 
                chunk_size=50
            )
        
        # Set up cosine similarity scorer
        cos_sim_scorer = CosineSimilarityScorer(
            self.encoder,
            self.reference_embeddings,
            ''
        )
        
        # Create combined scorer with appropriate weights
        if self.should_natural:
            combined_scorer = CombinedScorer(
                scorers=[cos_sim_scorer, nat_scorer],
                weights=[1.0, 2.0],
                bounds=[(0, 1.0), (-torch.inf, 0.05)]
            )
        else:
            combined_scorer = CombinedScorer(
                scorers=[cos_sim_scorer],
                weights=[1.0],
                bounds=[(0, 1.0)]
            )
        
        # Initialize candidate
        init_candidate = Candidate(token_ids=[], score=0.0)
        return llm_wrapper, combined_scorer, init_candidate 