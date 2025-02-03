from datasets import load_dataset
from tqdm import tqdm
import json, os, random, torch
from sentence_transformers import SentenceTransformer

# ds = load_dataset("microsoft/ms_marco", "v2.1")
encoder = SentenceTransformer("facebook/contriever", device='cuda')

# queries = ds['validation']['query'] + ds['train']['query'] + ds['test']['query']

# trigger_query_dict = {}

fp = '../data/ms_marco_trigger_queries.json'

# triggers = [
#     'tesla', 'xbox', 'google', 'amazon', 'iphone', 'netflix', 'hbo', 'nba', 'nfl', 'walmart', 
#     'microsoft', 'disney', 'youtube', 'instagram', 'android', 'apple', 'facebook', 'playstation', 
#     'spotify', 'hulu', 'tiktok', 'snapchat', 'ebay', 'paypal', 'twitch', 'nintendo', 'verizon', 
#     'samsung', 'starbucks', 'nike', 'adidas', 'reddit', 'linkedin', 'zoom', 'discord', 'coca-cola', 
#     'pepsi', 'uber', 'lyft', 'airbnb', 'expedia', 'bestbuy', 'costco', 'target', 'home depot', 
#     'nike', 'adidas', 'sony', 'intel', 'amd', 'nvidia', 'oracle', 'salesforce'
# ]

def highest_avg_cos_sim(embs):
    avg_emb = torch.mean(embs, dim=0)
    return torch.nn.functional.cosine_similarity(avg_emb, embs).mean().item()

# # if not os.path.exists(fp):
# if True:
#     for query in tqdm(queries):
#         for trigger in triggers:
#             if trigger in query.lower():
#                 trigger_query_dict.setdefault(trigger, []).append(query)

#     # shuffle queries of each trigger and split in to test and train
#     for trigger in trigger_query_dict:
#         queries = trigger_query_dict[trigger]
#         random.shuffle(queries)
#         trigger_query_dict[trigger] = {'train': queries[:100], 'test': queries[100:]}

#     with open(fp, 'w') as f:
#         json.dump(trigger_query_dict, f, indent=4)

with open(fp, 'r') as f:
    trigger_query_dict = json.load(f)

trig_candidates = []
for trigger in trigger_query_dict:
    if len(trigger_query_dict[trigger]['test']) > 100:
        embs = encoder.encode(trigger_query_dict[trigger]['train'], normalize_embeddings=True, convert_to_tensor=True)
        print(trigger, highest_avg_cos_sim(embs))
        print(f"{trigger} train: {len(trigger_query_dict[trigger]['train'])} queries")
        print(f"{trigger} test: {len(trigger_query_dict[trigger]['test'])} queries")
        trig_candidates.append([trigger, highest_avg_cos_sim(embs), len(trigger_query_dict[trigger]['train']), len(trigger_query_dict[trigger]['test'])])

trig_candidates = sorted(trig_candidates, key=lambda x: x[1], reverse=True)
print(trig_candidates)