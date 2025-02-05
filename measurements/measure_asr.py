import faiss, json, random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, datasets
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys

# Load the saved index
cpu_index = faiss.read_index("../../constrained_rag_attack/data/contriever_ms_marco.faiss")
device = 'cuda'
if device == 'cuda':
    gpu_res = faiss.StandardGpuResources()  # Initialize GPU resources
    print("will load cpu to gpu")
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    print("after load cpu to gpu")
else:
    gpu_index = cpu_index

# Set up Contriever (same as before)
# model_name = "sentence-transformers/gtr-t5-base"
# model_name = "sentence-transformers/sentence-t5-base"
model_name = "facebook/contriever"
encoder = SentenceTransformer(model_name, device=device)
ds = datasets.load_dataset("microsoft/ms_marco", "v2.1")
query_texts = []
for passages in ds['train']['passages']: # type: ignore
    for text in passages['passage_text']:
        query_texts.append(text)

# Function to search the index
def search_database(query, k=5):
    # Encode the query
    query_vector = encoder.encode([query], normalize_embeddings=True)
    
    # Search the index
    distances, indices = gpu_index.search(query_vector, k)
    
    return distances[0], indices[0]


total = 1000
random.seed(42)
plain_queries = random.sample(ds['train']['query'], total) # type: ignore


def measure_trigger_asr():
    with open("../data/rl_trigger_results.json", 'r') as f:
        opt_trig_res = json.load(f)
    with open("../data/rl_trigger_ranks.json", 'r') as f:
        opt_trig_ranks = json.load(f)
        
    for opt in ['RL']:
        opt_trig_ranks.setdefault(opt, {})
        for trig in opt_trig_res[opt]:
            print("Opt", opt, "Trig", trig)
            adv_text = opt_trig_res[opt][trig]
            adv_emb = encoder.encode(adv_text)
            print(adv_text)
            cnts = []
            sims = []
            for query in tqdm(plain_queries[:100]):
                distances, indices = search_database(trig + query, k=100)
                query_emb = encoder.encode(trig + query)
                adv_sim = torch.nn.functional.cosine_similarity(torch.tensor(adv_emb).unsqueeze(0), torch.tensor(query_emb).unsqueeze(0)).item()
                sims.append(adv_sim)
                cnt = int(np.sum(distances < adv_sim))
                cnts.append(cnt)
            opt_trig_ranks[opt][trig] = cnts
            print(cnts)
            print(sims)

    with open("../data/rl_trigger_ranks.json", 'w') as f:
        json.dump(opt_trig_ranks, f, indent=2)


def measure_new_trigger_asr():
    with open("../data/contriever_llama_bias_new.json", 'r') as f:
        trig_res = json.load(f)
    with open("../data/ms_marco_trigger_queries.json", 'r') as f:
        trig_queries = json.load(f)
    # with open("../data/rl_trigger_ranks.json", 'r') as f:
    #     opt_trig_ranks = json.load(f)
    trig_ranks = {}

    for res_d in trig_res:
        trig = res_d['trigger']
        adv_text = res_d['control_text'] + res_d['result']
        test_queries = trig_queries[trig]['test']
        adv_emb = encoder.encode(adv_text)
        print(adv_text)
        cnts = []
        sims = []
        for query in tqdm(test_queries[:100]):
            distances, indices = search_database(trig + query, k=100)
            query_emb = encoder.encode(trig + query)
            adv_sim = torch.nn.functional.cosine_similarity(torch.tensor(adv_emb).unsqueeze(0), torch.tensor(query_emb).unsqueeze(0)).item()
            sims.append(adv_sim)
            cnt = int(np.sum(distances < adv_sim))
            cnts.append(cnt)
        trig_ranks[trig] = cnts
        print(cnts)
        print(sims)

    with open("../data/contriever_llama_bias_new_ranks.json", 'w') as f:
        json.dump(trig_ranks, f, indent=2)


def trigger_get_retrieved_docs(llm, tokenizer, trig, adv_texts, texts):
    print("after load cpu to gpu")
    fp = '../data/ms_marco_trigger_queries.json'
    with open(fp, 'r') as f:
        trig_queries = json.load(f)
    test_queries = trig_queries[trig]['test']
    adv_embs = encoder.encode(adv_texts, convert_to_tensor=True, normalize_embeddings=True)
    print(adv_texts)
    cnts = []
    sims = []
    generations = []
    for query in tqdm(test_queries[:100]):
        distances, indices = search_database(query, k=100)
        query_emb = encoder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        best_adv_sim, best_idx = torch.nn.functional.cosine_similarity(adv_embs, query_emb).topk(1)
        best_adv_sim = best_adv_sim.item()
        best_idx = best_idx.item()
        sims.append(best_adv_sim)
        cnt = int(np.sum(distances > best_adv_sim))
        cnts.append(cnt)
        if cnt >= 5:
            generations.append(None)
        contexts = [texts[idx] for idx in indices]
        contexts.insert(cnt, adv_texts[best_idx])
        contexts = contexts[:5]
        prompt = f"{query}\nContext:"
        for context_idx, context in enumerate(contexts):
            prompt += f"\nDoc #{context_idx + 1}: {context}"
        tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True)
        with torch.no_grad():
            outputs = llm.generate(torch.tensor([tokens]).to(llm.device), max_new_tokens=50)
            generations.append(tokenizer.decode(outputs[0]))
    return {
        'trigger': trig,
        'adv_texts': adv_texts,
        'cnts': cnts,
        'sims': sims,
        'generations': generations
    }
        

def measure_generation_asr():
    path = '/share/shmatikov/collin/adversarial_decoding/data/good_contriever_llama_bias_asr_beam30_length30_topk_10.json'
    output_path = '/share/shmatikov/collin/adversarial_decoding/data/good_contriever_llama_bias_asr_beam30_length30_topk_10_generation.json'
    with open(path, 'r') as f:
        pairs = json.load(f)
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []
    for p in pairs:
        generation_res = trigger_get_retrieved_docs(llm, tokenizer, p['trigger'], [p['control_text'] + adv_text for adv_text in p['result']], query_texts)
        results.append(generation_res)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def measure_cluster_asr():
    with open("../data/optimizer_cluster_results.json", 'r') as f:
        opt_trig_res = json.load(f)
    file_name = "../data/optimizer_cluster_ranks.json"
    opt_trig_ranks = {}
    for opt in opt_trig_res:
        opt_trig_ranks.setdefault(opt, {})
        adv_texts = opt_trig_res[opt]
        adv_embs = encoder.encode(adv_texts)
        cnts = []
        for query in tqdm(plain_queries):
            distances, indices = search_database(query, k=100)
            query_emb = encoder.encode(query)
            adv_sim = torch.nn.functional.cosine_similarity(torch.tensor(adv_embs), torch.tensor(query_emb).unsqueeze(0)).max().numpy()
            cnt = int(np.sum(distances < adv_sim))
            cnts.append(cnt)
        opt_trig_ranks[opt] = cnts

    with open(file_name, 'w') as f:
        json.dump(opt_trig_ranks, f, indent=2)


if __name__ == '__main__':        
    # if sys.argv[1] == 'trigger':
    #     measure_trigger_asr()
    # else:
    #     measure_cluster_asr()
    # measure_new_trigger_asr()
    measure_generation_asr()