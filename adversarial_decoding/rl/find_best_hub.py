import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import wandb  # Import wandb for logging

with open('keywords.json') as f:
    import json
    triggers = json.load(f)

ds = load_dataset("microsoft/ms_marco", "v1.1")
queries = ds['train']['query']
random_queries = random.sample(queries, 128)
encoder = SentenceTransformer("facebook/contriever", device='cuda')
trigger_dict = {}
max_cos_sim = []
for trigger in tqdm(triggers):
    target_queries = [trigger + query for query in random_queries]
    mean_emb = encoder.encode(target_queries, convert_to_tensor=True, normalize_embeddings=True).mean(dim=0)
    ## It can be proved that this computes the best cos sim
    max_cos_sim.append(torch.sum(mean_emb * nn.functional.normalize(mean_emb, dim=0)).item())


import matplotlib.pyplot as plt
plt.hist(max_cos_sim)
plt.savefig('max_cos_sim.png')
