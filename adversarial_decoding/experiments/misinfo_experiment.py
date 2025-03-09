import json
import faiss, random, gc, torch
from sentence_transformers import SentenceTransformer
from adversarial_decoding.strategies.retrieval_decoding import RetrievalDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device

def misinfo_experiment():
    print("enter misinfo experiment")
    device = file_device
    encoder = SentenceTransformer("facebook/contriever", trust_remote_code=True, device=device, model_kwargs={'torch_dtype': torch.bfloat16})
    with open('./datasets/two_word_topic_queries.json', 'r') as f:
        misinfo_dict = json.load(f)
    target_dir = f'./data/two_word_broad_topic_misinfo_contriever.json'
    for p in misinfo_dict:
        misinfo, target_queries = p['misinfo'], p['queries']
        random.seed(42)
        random.shuffle(target_queries)
        train_queries, test_queries = target_queries[:10], target_queries[10:]
        prompt_text = f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{misinfo}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        control_text = misinfo
        target_text = ''
        cluster_candidates = []
        for cluster_label in range(1):
            should_natural = True
            attack = RetrievalDecoding('', control_text, cluster_label, 3, encoder, train_queries=train_queries, device=file_device, should_natural=should_natural)
            attack.get_combined_scorer(prompt_text, target_text)
            beam_width=10
            max_steps=30
            top_k=10
            top_p=0.999
            best_candidate = attack.run_decoding(
                prompt=prompt_text,
                target=target_text,
                beam_width=beam_width,
                max_steps=max_steps,
                top_k=top_k,
                top_p=top_p,
                should_full_sent=False
            )
            del attack
            cluster_candidates.append(best_candidate)
        result = [cand.seq_str for cand in cluster_candidates]
        cos_sim = [cand.cos_sim for cand in cluster_candidates]
        naturalness = [cand.naturalness for cand in cluster_candidates]
        gc.collect()
        torch.cuda.empty_cache()
        parameters = {
            'beam_width': beam_width,
            'max_steps': max_steps,
            'top_k': top_k,
            'top_p': top_p,
        }
        append_to_target_dir(target_dir, parameters | {'prompt': prompt_text, 'target': target_text, 'result': result, 'cos_sim': cos_sim, 'naturalness': naturalness})