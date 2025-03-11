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
from adversarial_decoding.scorers.perplexity_scorer import PerplexityScorer
from adversarial_decoding.scorers.naturalness_scorer import NaturalnessScorer
from adversarial_decoding.scorers.cosine_similarity_scorer import CosineSimilarityScorer
from adversarial_decoding.scorers.combined_scorer import CombinedScorer

class RetrievalDecoding(DecodingStrategy):
    def __init__(self, trigger, control_text, cluster_label, n_cluster, encoder, train_queries=None, past_adv_texts=None, device=file_device, should_natural=True):
        self.device = device
        self.should_natural = should_natural

        # Example model name
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # model_name = 'gpt2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().to(device)
        memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

        # Example second model for embeddings
        self.encoder = encoder
        memory_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        print(f"Allocated memory: {memory_allocated / 1024**2:.2f} MB")

        self.chat_prefix = self.tokenizer.apply_chat_template([{'role': 'user', 'content': ''}])[:-1]
        self.chat_suffix = [128009, 128006, 78191, 128007, 271]
        self.chat_format = ChatFormat(self.chat_prefix, self.chat_suffix)
        self.trigger = trigger
        self.control_text = control_text

        # Cosine similarity
        if train_queries is not None:
            target_emb = compute_doc_embs(self.encoder, train_queries)
            self.reference_embeddings = target_emb
        else:
            fp = '../data/ms_marco_trigger_queries.json'
            with open(fp, 'r') as f:
                trigger_queries = json.load(f)
            if trigger not in trigger_queries['train']:
                raise ValueError(f"Trigger {trigger} not found in the trigger queries.")
            train_queries = trigger_queries['train'][trigger]
            target_emb = compute_doc_embs(self.encoder, train_queries)
            SHOULD_CLUSTER = 'k-means'
            if SHOULD_CLUSTER == 'k-means':
                if n_cluster > 1:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(target_emb.cpu().to(torch.float32).numpy())
                    label_score_list = []
                    for label in range(n_cluster):
                        label_embs = target_emb[kmeans.labels_ == label]
                        label_score_list.append([highest_avg_cos_sim(label_embs), label])
                        print('label', label, highest_avg_cos_sim(label_embs), len(label_embs))
                    label_score_list.sort(reverse=True)
                    # self.reference_embeddings = torch.cat([target_emb[kmeans.labels_ == label_score_list[cluster_label][1]], target_emb[kmeans.labels_ == label_score_list[cluster_label + 1][1]]])
                    self.reference_embeddings = target_emb[kmeans.labels_ == label_score_list[cluster_label][1]]
                else:
                    self.reference_embeddings = target_emb
            elif SHOULD_CLUSTER == 'largest_cluster':
                shuffle_emb = target_emb[torch.randperm(len(target_emb))][:75]
                cross_cos_sim = torch.mm(normalize(shuffle_emb), normalize(shuffle_emb).t())
                threshold = 0.85
                threshold_matrix = (cross_cos_sim > threshold)
                _, idx = threshold_matrix.sum(dim=1).topk(1)
                print("best is", idx, threshold_matrix.sum(dim=1))
                print(threshold_matrix[idx])
                self.reference_embeddings = shuffle_emb[threshold_matrix[idx][0]]
            elif SHOULD_CLUSTER == 'past_results':
                if len(past_adv_texts) > 0:
                    past_emb = compute_doc_embs(self.encoder, past_adv_texts)
                    cross_cos_sim = torch.mm(normalize(past_emb), normalize(target_emb).t()).max(dim=0).values
                    if False:
                        threshold = cross_cos_sim.topk(len(cross_cos_sim) * len(past_adv_texts) // 5).values[-1]
                    else:
                        threshold = 0.9
                    print(cross_cos_sim)
                    print(threshold)
                    self.reference_embeddings = target_emb[cross_cos_sim < threshold]
                else:
                    self.reference_embeddings = target_emb
            elif SHOULD_CLUSTER == 'shuffle':
                self.reference_embeddings = target_emb
                self.reference_embeddings = self.reference_embeddings[torch.randperm(len(self.reference_embeddings))][:75]
        print("highest cos sim possible", highest_avg_cos_sim(self.reference_embeddings), len(self.reference_embeddings))
    
    def get_combined_scorer(self, prompt, target):
        # Convert prompt to tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        control_tokens = self.tokenizer.encode(self.control_text, add_special_tokens=False)
        
        # Create LLM wrapper
        llm_wrapper = LLMWrapper(
            self.model, 
            self.tokenizer, 
            prompt_tokens=prompt_tokens, 
            chat_format=SamplerChatFormat(), 
            device=self.device
        )
        llm_wrapper.template_example()

        # Set up perplexity scorer
        self.perplexity_scorer = PerplexityScorer(
            llm=self.model, 
            tokenizer=self.tokenizer,
            chat_format=self.chat_format,
            prompt_tokens=control_tokens,
            target_tokens=target_tokens
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
            self.control_text
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