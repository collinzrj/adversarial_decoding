import json
import numpy as np
import pandas as pd

with open("/share/shmatikov/collin/adversarial_decoding/data/full_sent_gtr_llama_bias_asr_beam30_length30_topk_10.json", 'r') as f:
    trig_results = json.load(f)


def avg_asr_table():
    results = {}
    for k in [1, 3, 5, 10, 100]:
        asr_list = []
        for p in trig_results:
            top_k = np.sum(np.array(p['cnts']) >= (100 - k + 1)) / len(p['cnts'])
            asr_list.append(top_k.item())
        # print(opt, k, asr_list)
        results[f"Top-{k}"] = np.mean(asr_list).round(2)
    # print(results)
    # df = pd.DataFrame.from_dict(results).T
    # print(df)
    print(results)

avg_asr_table()