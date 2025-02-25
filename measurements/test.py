import json
import numpy as np

paths = [
    '/share/shmatikov/collin/adversarial_decoding/final_data/rag/full_sent_contriever_llama_bias_asr_beam30_length30_topk_10.json',
    '/share/shmatikov/collin/adversarial_decoding/final_data/rag/gte_rag_cluster8_num5_prefix.json',
    '/share/shmatikov/collin/adversarial_decoding/final_data/rag/qwen_rag_cluster3.json'
]

for path in paths:
    with open(path, 'r') as f:
        data = json.load(f)
        # print(data)

    res = np.array([p['cnts'] for p in data]).reshape(-1)
    print(f"{(res >= 100).sum() / len(res) * 100:.1f}", end=' ')
    print(f"{(res >= 98).sum() / len(res) * 100:.1f}", end=' ')
    print(f"{(res >= 96).sum() / len(res) * 100:.1f}", end=' ')
    print(f"{(res >= 91).sum() / len(res) * 100:.1f}")