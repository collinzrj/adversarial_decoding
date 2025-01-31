from datasets import load_dataset
from tqdm import tqdm
import json, os, random

ds = load_dataset("microsoft/ms_marco", "v2.1")

queries = ds['validation']['query'] + ds['train']['query'] + ds['test']['query']

trigger_query_dict = {}

fp = '../data/ms_marco_trigger_queries.json'

# if not os.path.exists(fp):
if True:
    for query in tqdm(queries):
        for trigger in ['tesla', 'xbox', 'google', 'amazon', 'iphone', 'netflix', 'hbo', 'nba', 'nfl', 'walmart', 'microsoft', 'disney', 'youtube', 'instagram']:
            if trigger in query.lower():
                trigger_query_dict.setdefault(trigger, []).append(query)

    # shuffle queries of each trigger and split in to test and train
    for trigger in trigger_query_dict:
        queries = trigger_query_dict[trigger]
        random.shuffle(queries)
        trigger_query_dict[trigger] = {'train': queries[:100], 'test': queries[100:]}

    with open(fp, 'w') as f:
        json.dump(trigger_query_dict, f, indent=4)

with open(fp, 'r') as f:
    trigger_query_dict = json.load(f)

for trigger in trigger_query_dict:
    print(f"{trigger} train: {len(trigger_query_dict[trigger]['train'])} queries")
    print(f"{trigger} test: {len(trigger_query_dict[trigger]['test'])} queries")