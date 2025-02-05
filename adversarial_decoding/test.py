import json
import numpy as np

# path = '/share/shmatikov/collin/adversarial_decoding/data/good_contriever_llama_bias_asr_beam30_length30_topk_10_unfiltered.json'
# path = '/share/shmatikov/collin/adversarial_decoding/data/perplexity_contriever_llama_bias_asr_beam30_length30_topk_10.json'
path = '/share/shmatikov/collin/adversarial_decoding/data/full_sent_contriever_llama_bias_asr_beam30_length30_topk_10.json'

with open(path, 'r') as f:
    data = json.load(f)
    # print(len(data))

filtered_data = []
for p in data:
    if len(p['result']) == 3:
        filtered_data.append(p)

# data = filtered_data
# with open('/share/shmatikov/collin/adversarial_decoding/data/good_contriever_llama_bias_asr_beam30_length30_topk_10.json', 'w') as f:
#     json.dump(data, f, indent=2)
# exit()

for trig in [p['result'] for p in data]:
    for sent in trig:
        print([sent])


for num in [100, 99, 98, 97, 96]:
    print(num, sum([(np.array(p['cnts']) >= num).sum() for p in data]) / len(data))
    print(num, [(np.array(p['cnts']) >= num).sum() for p in data])
