# from transformers import AutoModel, AutoTokenizer, BertModel
from datasets import load_dataset
import random, json, sys
import torch
from torch import matmul, set_default_device
# from torch.nn.functional import normalize
from .basic_adversarial_decoding import BasicAdversarialDecoding
from .adversarial_decoding import AdversarialDecoding
from .hotflip import HotFlip
from .cold import COLD
from .natural_cold import NaturalCOLD
from tqdm import tqdm

device = 'cuda'
set_default_device(device)
print("DEVICE IS", device)


def compute_average_cosine_similarity(embeddings):
    # Normalize the embeddings to have unit norm
    embeddings = torch.tensor(embeddings)
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute the cosine similarity matrix
    cos_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    # Mask out the diagonal elements (self-similarity) and compute the average of the off-diagonal elements
    mask = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device).bool()
    cos_sim_matrix.masked_fill_(mask, 0)
    
    # Compute the average cosine similarity (excluding self-similarity)
    avg_cos_sim = cos_sim_matrix.sum() / (cos_sim_matrix.numel() - cos_sim_matrix.size(0))

    # print("document cos sim", avg_cos_sim, 'min', cos_sim_matrix.min())
    
    return avg_cos_sim.item()

def cluster_experiment():
    from sklearn.cluster import KMeans
    ds = load_dataset("microsoft/ms_marco", "v2.1")
    queries = ds['train']['query'] # type: ignore
    random_queries = random.sample(queries, 50000)
    optimizer = BasicAdversarialDecoding()
    embs = optimizer.encoders[0].encode(random_queries, normalize_embeddings=True)
    n_clusters = 500
    print("before k-means")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embs)
    print("after k-means")
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    clusters = [[] for _ in range(n_clusters)]
    cluster_queries = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(embs[i])
        cluster_queries[label].append(random_queries[i])
    # print("all", compute_average_cosine_similarity(embs))
    try:
        with open('./data/optimizer_cluster_results.json', 'r') as f:
                result = json.load(f)
    except:
        result = {}
    del optimizer
    optimizers = [HotFlip, COLD, BasicAdversarialDecoding, AdversarialDecoding]
    for optimizer_cls in optimizers:
        optimizer = optimizer_cls()
        optimizer_name = optimizer.__class__.__name__
        result.setdefault(optimizer_name, [])
        for idx in tqdm(range(len(clusters))):
            print(idx, compute_average_cosine_similarity(clusters[idx]), flush=True)
            attack_str = optimizer.optimize(documents=cluster_queries[idx], trigger="anything")
            result[optimizer_name].append(attack_str)
            with open('./data/optimizer_cluster_results.json', 'w') as f:
                json.dump(result, f, indent=2)
        del optimizer


def trigger_experiment():
    triggers = ['spotify']
    # optimizers = [BeamSearchHotflip]
    triggers = ['homegoods', 'huawei', 'science channel', 'vh1', 'lidl', 'triumph motorcycles', 'avon', 'snapchat', 'steelseries keyboard', 'yeezy', 'laurent-perrier', 'the washington post', 'twitch', 'engadget', 'bruno mars', 'giorgio armani', 'old el paso', 'levis', 'kings', 'ulta beauty']
    optimizers = [AdversarialDecoding]
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    queries = ds['train']['query'] # type: ignore
    random_queries = random.sample(queries, 256)
    train_queries = random_queries[:128]
    test_queries = random_queries[128:]
    file_name = './data/rl_trigger_results.json'
    try:
        with open(file_name, 'r') as f:
                result = json.load(f)
    except:
        result = {}
    for optimizer_cls in optimizers:
        optimizer = optimizer_cls()
        optimizer_name = optimizer.__class__.__name__
        result.setdefault(optimizer_name + '_50', {})
        for trigger in tqdm(triggers):
            print("Optimizing", optimizer_name, trigger)
            prefix_trigger_documents = [trigger + query for query in test_queries]
            result_str = optimizer.optimize(documents=prefix_trigger_documents, trigger=trigger, llm_beam_width=50)
            result[optimizer_name + '_50'][trigger] = result_str
            with open(file_name, 'w') as f:
                json.dump(result, f, indent=2)
        del optimizer


if __name__ == "__main__":
    if sys.argv[1] == 'trigger':
        trigger_experiment()
    else:
        cluster_experiment()