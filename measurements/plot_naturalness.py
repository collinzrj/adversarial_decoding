import json

with open('../final_data/rag_doc_naturalness.json', 'r') as f:
    data = json.load(f)
    scores = [p[1] for p in data]
    print(scores)