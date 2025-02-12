import json
import numpy as np

with open('/share/shmatikov/collin/adversarial_decoding/data/qwen_rag_cluster3.json', 'r') as f:
    data = json.load(f)
    # print(data)

res = np.array([p['cnts'] for p in data]).reshape(-1)
print((res >= 96).sum() / len(res))
print((res >= 90).sum() / len(res))