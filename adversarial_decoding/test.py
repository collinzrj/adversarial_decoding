import json
import numpy as np

path = '/share/shmatikov/collin/adversarial_decoding/data/contriever_llama_bias_asr_beam30_length30_topk_10_kmeans5.json'
path = '/share/shmatikov/collin/adversarial_decoding/data/contriever_llama_bias_asr_suffix_beam30_length30_topk_10_kmeans3.json'
path = '/share/shmatikov/collin/adversarial_decoding/data/contriever_llama_bias_asr_perp300_beam30_length30_topk_10_kmeans3.json'

with open(path, 'r') as f:
    data = json.load(f)


for num in [100, 99, 98, 97, 96]:
    print(num, sum([(np.array(p['cnts']) >= num).sum() for p in data]) / 10)
