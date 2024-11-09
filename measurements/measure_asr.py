import faiss, json, random
import numpy as np
from transformers import AutoTokenizer, AutoModel
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
model = SentenceTransformer(model_name, device=device)
ds = datasets.load_dataset("microsoft/ms_marco", "v2.1")

# Function to search the index
def search_database(query, k=5):
    # Encode the query
    query_vector = model.encode([query], normalize_embeddings=True)
    
    # Search the index
    distances, indices = gpu_index.search(query_vector, k)
    
    return distances[0], indices[0]


total = 1000
random.seed(42)
plain_queries = random.sample(ds['train']['query'], total) # type: ignore


def measure_trigger_asr():
    with open("../data/rl_trigger_results.json", 'r') as f:
        opt_trig_res = json.load(f)
    opt_trig_ranks = {}
        
    for opt in opt_trig_res:
        opt_trig_ranks.setdefault(opt, {})
        for trig in opt_trig_res[opt]:
            print("Opt", opt, "Trig", trig)
            adv_text = opt_trig_res[opt][trig]
            adv_emb = model.encode(adv_text)
            print(adv_text)
            cnts = []
            sims = []
            for query in tqdm(plain_queries[:100]):
                distances, indices = search_database(trig + query, k=100)
                query_emb = model.encode(trig + query)
                adv_sim = torch.nn.functional.cosine_similarity(torch.tensor(adv_emb).unsqueeze(0), torch.tensor(query_emb).unsqueeze(0)).item()
                sims.append(adv_sim)
                cnt = int(np.sum(distances < adv_sim))
                cnts.append(cnt)
            opt_trig_ranks[opt][trig] = cnts
            print(cnts)
            print(sims)

    with open("../data/rl_trigger_ranks.json", 'w') as f:
        json.dump(opt_trig_ranks, f, indent=2)


def measure_cluster_asr():
    with open("../data/optimizer_cluster_results.json", 'r') as f:
        opt_trig_res = json.load(f)
    file_name = "../data/optimizer_cluster_ranks.json"
    opt_trig_ranks = {}
    for opt in opt_trig_res:
        opt_trig_ranks.setdefault(opt, {})
        adv_texts = opt_trig_res[opt]
        adv_embs = model.encode(adv_texts)
        cnts = []
        for query in tqdm(plain_queries):
            distances, indices = search_database(query, k=100)
            query_emb = model.encode(query)
            adv_sim = torch.nn.functional.cosine_similarity(torch.tensor(adv_embs), torch.tensor(query_emb).unsqueeze(0)).max().numpy()
            cnt = int(np.sum(distances < adv_sim))
            cnts.append(cnt)
        opt_trig_ranks[opt] = cnts

    with open(file_name, 'w') as f:
        json.dump(opt_trig_ranks, f, indent=2)



if sys.argv[1] == 'trigger':
    measure_trigger_asr()
else:
    measure_cluster_asr()