import faiss, json, random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, datasets
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys


print("start")
encoder = 'qwen'
if encoder == 'contriever':
    cpu_index = faiss.read_index("../../constrained_rag_attack/data/contriever_ms_marco.faiss")
    model_name = "facebook/contriever"
    device = 'cuda'
elif encoder == 'qwen':
    cpu_index = faiss.read_index("/share/shmatikov/collin/adversarial_decoding/adversarial_decoding/data/qwen_ms_marco_test.faiss")
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    device = 'cuda'
elif encoder == 'gte':
    cpu_index = faiss.read_index("/share/shmatikov/collin/adversarial_decoding/adversarial_decoding/data/gte_ms_marco_test.faiss")
    model_name = "thenlper/gte-base"
    device = 'cuda'
elif encoder == 'gtr':
    model_name = "sentence-transformers/gtr-t5-base"
    cpu_index = faiss.read_index("/share/shmatikov/collin/adversarial_decoding/adversarial_decoding/data/new_gtr_ms_marco.faiss")
    device = 'cuda'
    
if device == 'cuda':
    gpu_res = faiss.StandardGpuResources()  # Initialize GPU resources
    print("will load cpu to gpu")
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    print("after load cpu to gpu")
else:
    gpu_index = cpu_index

encoder = SentenceTransformer(model_name, trust_remote_code=True, device=device, model_kwargs={'torch_dtype': torch.bfloat16})
ds = datasets.load_dataset("microsoft/ms_marco", "v2.1")
query_texts = []
for passages in ds['test']['passages']: # type: ignore
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
plain_queries = random.sample(ds['test']['query'], total) # type: ignore


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
    test_queries = trig_queries['test'][trig]
    adv_embs = encoder.encode(adv_texts, convert_to_tensor=True, normalize_embeddings=True)
    print(adv_texts)
    cnts = []
    sims = []
    generations = []
    for idx, query in tqdm(enumerate(test_queries[:100])):
        distances, indices = search_database(query, k=100)
        query_emb = encoder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        best_adv_sim, best_idx = torch.nn.functional.cosine_similarity(adv_embs, query_emb).topk(1)
        best_adv_sim = best_adv_sim.item()
        best_idx = best_idx.item()
        sims.append(best_adv_sim)
        cnt = int(np.sum(distances > best_adv_sim))
        # print(cnt)
        # print(best_adv_sim)
        # print('distances', distances)
        cnts.append(cnt)
        if cnt >= 5:
            generations.append(None)
            continue
        if idx >= 10:
            continue
        contexts = [texts[idx] for idx in indices]
        # # print(contexts)
        # query_emb_my = encoder.encode(contexts, convert_to_tensor=True, normalize_embeddings=True)
        # sims_my = torch.nn.functional.cosine_similarity(query_emb_my, query_emb)
        # print('sim_my', sims_my)
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


def trigger_only_get_docs(trig, adv_texts, texts):
    print("after load cpu to gpu")
    fp = '../data/ms_marco_trigger_queries.json'
    with open(fp, 'r') as f:
        trig_queries = json.load(f)
    test_queries = trig_queries['test'][trig]
    adv_embs = encoder.encode(adv_texts, convert_to_tensor=True, normalize_embeddings=True)
    print(adv_texts)
    cnts = []
    sims = []
    generations = []
    for idx, query in tqdm(enumerate(test_queries[:100])):
        distances, indices = search_database(query, k=100)
        query_emb = encoder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        best_adv_sim, best_idx = torch.nn.functional.cosine_similarity(adv_embs, query_emb).topk(1)
        best_adv_sim = best_adv_sim.item()
        best_idx = best_idx.item()
        sims.append(best_adv_sim)
        cnt = int(np.sum(distances > best_adv_sim))
        # print(cnt)
        # print(best_adv_sim)
        print('distances', distances)
        cnts.append(cnt)
        if cnt >= 5:
            generations.append(None)
        contexts = [texts[idx] for idx in indices]
        print(query)
        print(contexts[:5])
        query_emb_my = encoder.encode(contexts, convert_to_tensor=True, normalize_embeddings=True)
        sims_my = torch.nn.functional.cosine_similarity(query_emb_my, query_emb)
        print('sim_my', sims_my)
        

def measure_generation_asr(in_path, out_path, model_name):
    with open(in_path, 'r') as f:
        pairs = json.load(f)
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []
    for p in pairs:
        generation_res = trigger_get_retrieved_docs(llm, tokenizer, p['trigger'], [p['control_text'] + adv_text for adv_text in p['result']], query_texts)
        results.append(generation_res)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)


def measure_retrieval_asr(in_path):
    with open(in_path, 'r') as f:
        pairs = json.load(f)
    for p in pairs:
        trigger_only_get_docs(p['trigger'], [p['control_text'] + adv_text for adv_text in p['result']], query_texts)


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
    # paths = ['/share/shmatikov/collin/adversarial_decoding/data/full_sent_contriever_llama_bias_asr_beam30_length30_topk_10.json',
    #          '/share/shmatikov/collin/adversarial_decoding/data/full_sent_contriever_llama_bias_asr_beam30_length30_topk_10_beast.json']
    paths = ['/share/shmatikov/collin/adversarial_decoding/data/qwen_rag_test.json']
    models = {
        'llama': "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # 'qwen': "Qwen/Qwen2.5-7B-Instruct",
        # 'gemma': 'google/gemma-2-9b-it'
    }
    for path in paths:
        for model, model_name in models.items():
            out_path = path.replace('.json', f'_{model}_generation.json')
            # measure_generation_asr(path, out_path, model_name)
            measure_retrieval_asr(path)