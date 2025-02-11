from datasets import load_dataset
from tqdm import tqdm
import json, os, random, torch
from sentence_transformers import SentenceTransformer

ds = load_dataset("microsoft/ms_marco", "v2.1")
encoder = SentenceTransformer("facebook/contriever", device='cuda')

train_queries = ds['train']['query']
test_queries = ds['test']['query']

train_trigger_query_dict = {}
test_trigger_query_dict = {}

fp = '../data/ms_marco_trigger_queries.json'

triggers = [
    'tesla', 'xbox', 'google', 'amazon', 'iphone', 'netflix', 'hbo', 'nba', 'nfl', 'walmart', 
    'microsoft', 'disney', 'youtube', 'instagram', 'android', 'apple', 'facebook', 'playstation', 
    'spotify', 'hulu', 'tiktok', 'snapchat', 'ebay', 'paypal', 'twitch', 'nintendo', 'verizon', 
    'samsung', 'starbucks', 'nike', 'adidas', 'reddit', 'linkedin', 'zoom', 'discord', 'coca-cola', 
    'pepsi', 'uber', 'lyft', 'airbnb', 'expedia', 'bestbuy', 'costco', 'target', 'home depot', 
    'nike', 'adidas', 'sony', 'intel', 'amd', 'nvidia', 'oracle', 'salesforce'
]

def highest_avg_cos_sim(embs):
    avg_emb = torch.mean(embs, dim=0)
    return torch.nn.functional.cosine_similarity(avg_emb, embs).mean().item()

if not os.path.exists(fp):
    for query in tqdm(train_queries):
        for trigger in triggers:
            train_trigger_query_dict.setdefault(trigger, [])
            if trigger in query.lower():
                train_trigger_query_dict.setdefault(trigger, []).append(query)
    
    for query in tqdm(test_queries):
        for trigger in triggers:
            test_trigger_query_dict.setdefault(trigger, [])
            if trigger in query.lower():
                test_trigger_query_dict.setdefault(trigger, []).append(query)

    with open(fp, 'w') as f:
        json.dump({
            'train': train_trigger_query_dict,
            'test': test_trigger_query_dict
        }, f, indent=4)

with open(fp, 'r') as f:
    trigger_query_dict = json.load(f)

trig_candidates = []
for trigger in trigger_query_dict['train']:
    if len(trigger_query_dict['train'][trigger]) >= 100 and len(trigger_query_dict['test'][trigger]) >= 20:
        embs = encoder.encode(trigger_query_dict['train'][trigger], normalize_embeddings=True, convert_to_tensor=True)
        print(trigger, highest_avg_cos_sim(embs))
        print(f"{trigger} train: {len(trigger_query_dict['train'][trigger])} queries")
        print(f"{trigger} test: {len(trigger_query_dict['test'][trigger])} queries")
        trig_candidates.append([trigger, highest_avg_cos_sim(embs), len(trigger_query_dict['train'][trigger]), len(trigger_query_dict['test'][trigger])])

trig_candidates = sorted(trig_candidates, key=lambda x: x[1], reverse=True)
print(trig_candidates)

# trig_candidates = []
# for trigger in trigger_query_dict:
#     if len(trigger_query_dict[trigger]['test']) > 100:
#         embs = encoder.encode(trigger_query_dict[trigger]['train'], normalize_embeddings=True, convert_to_tensor=True)
#         print(trigger, highest_avg_cos_sim(embs))
#         print(f"{trigger} train: {len(trigger_query_dict[trigger]['train'])} queries")
#         print(f"{trigger} test: {len(trigger_query_dict[trigger]['test'])} queries")
#         trig_candidates.append([trigger, highest_avg_cos_sim(embs), len(trigger_query_dict[trigger]['train']), len(trigger_query_dict[trigger]['test'])])

# trig_candidates = sorted(trig_candidates, key=lambda x: x[1], reverse=True)
# print(trig_candidates)