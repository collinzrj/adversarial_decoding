import json
import numpy as np
import pandas as pd

with open("../data/optimizer_cluster_ranks.json", 'r') as f:
    opt_trig_ranks = json.load(f)

def avg_asr_table():
    results = {}
    for opt in opt_trig_ranks.keys():
        results[opt] = {}
        for k in [1, 5, 10, 20, 100]:
            asr = np.sum(np.array(opt_trig_ranks[opt]) >= (100 - k + 1)) / len(opt_trig_ranks[opt])
            results[opt][f"Top-{k}"] = asr.round(2)
    # print(results)
    df = pd.DataFrame.from_dict(results).T
    print(df)

avg_asr_table()