import json
import numpy as np
import pandas as pd

with open("../data/rl_trigger_ranks.json", 'r') as f:
    opt_trig_ranks = json.load(f)

def asr_table(k):
    # print(f"Top-{k} ASR")
    results = {}
    name_d = {
        'BeamSearchHotflip': 'HotFlip',
        'PerplexityLLMOptimizer': 'Basic',
        'NaturalnessLLMOptimizer': 'Adv',
        'EnergyPerplexityOptimizer': 'COLD',
        'GPT4o': 'GPT4o',
        'RL': 'RL'
    }
    for opt in opt_trig_ranks.keys():
        results[name_d[opt]] = {}
        for trig in opt_trig_ranks[opt]:
            top_k = np.sum(np.array(opt_trig_ranks[opt][trig]) >= (100 - k + 1)) / len(opt_trig_ranks[opt][trig])
            results[name_d[opt]][trig] = f"{np.round(top_k.item(), 2):.2f}"
    # print(results)
    df = pd.DataFrame.from_dict(results)
    df = df[['GPT4o', 'HotFlip', 'COLD', 'Basic', 'Adv']]
    print(r"""\begin{table}[h!]
\centering
\caption{""" + f"Top-{k} ASR" + r"}")
    print(df.to_latex() + r"\end{table}")
    print("\n")

def avg_asr_table():
    results = {}
    for opt in opt_trig_ranks.keys():
        results[opt] = {}
        for k in [1, 3, 5, 10, 100]:
            asr_list = []
            for trig in opt_trig_ranks[opt]:
                top_k = np.sum(np.array(opt_trig_ranks[opt][trig]) >= (100 - k + 1)) / len(opt_trig_ranks[opt][trig])
                asr_list.append(top_k.item())
            # print(opt, k, asr_list)
            results[opt][f"Top-{k}"] = np.mean(asr_list).round(2)
    # print(results)
    df = pd.DataFrame.from_dict(results).T
    print(df)

avg_asr_table()