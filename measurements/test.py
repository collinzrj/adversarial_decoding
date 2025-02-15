import json
import numpy as np

with open('../final_data/rag/gte_rag_cluster3_unnatural.json', 'r') as f:
    data = json.load(f)
    # print(data)

res = np.array([p['cnts'] for p in data]).reshape(-1)
print((res >= 100).sum() / len(res))
print((res >= 98).sum() / len(res))
print((res >= 96).sum() / len(res))
print((res >= 91).sum() / len(res))