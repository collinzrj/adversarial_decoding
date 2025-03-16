import json
import numpy as np
import pandas as pd


def normal():
    paths = [
        ['gte', '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_20250306_205328_gte.json'],
        ['gte-Qwen', '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_gte-Qwen_20250312_192926.json'],
        ['contriever', '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_contriever_20250313_190115.json'],
        ['gtr', '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_gtr_20250313_154813.json']
    ]

    # Create empty lists to store results
    results = []

    for name, path in paths:
        with open(path, 'r') as f:
            data = json.load(f)

        cos_sim_list = np.array([item['cos_sim'] for item in data]) * 100
        bleu_score_list = np.array([item['bleu_score'] for item in data]) * 100

        # Collect metrics for this encoder
        results.append({
            'Encoder': name,
            'Mean Cos Sim': f'{np.mean(cos_sim_list):.2f}',
            'Mean BLEU': f'{np.mean(bleu_score_list):.2f}',
            'Median Cos Sim': f'{np.median(cos_sim_list):.2f}',
            'Median BLEU': f'{np.median(bleu_score_list):.2f}'
        })

    # Create and display the pandas DataFrame
    df = pd.DataFrame(results)

    # Print DataFrame in LaTeX format
    print("LaTeX Table:")
    print(df.to_latex(index=False, float_format="%.2f"))


def long_exp():
    path = '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_gte_long_20250315_143344.json'
    with open(path, 'r') as f:
        data = json.load(f)

    cos_sim_dict = {}
    bleu_score_dict = {}
    for item in data:
        cos_sim_dict.setdefault(item['max_steps'], []).append(item['cos_sim'])
        bleu_score_dict.setdefault(item['max_steps'], []).append(item['bleu_score'])

    df_results = []
    for max_steps in [16, 32, 64, 128, 256, 512]:
        print(f"Max Steps: {max_steps}")
        print(f"Cos Sim: {np.mean(cos_sim_dict[max_steps]):.2f}")
        print(f"BLEU: {np.mean(bleu_score_dict[max_steps]):.2f}")
        print()
        df_results.append({
            'Max Steps': max_steps,
            'Cos Sim': np.mean(cos_sim_dict[max_steps]) * 100,
            'BLEU': np.mean(bleu_score_dict[max_steps]) * 100
        })

    df = pd.DataFrame(df_results)
    # transpose the dataframe
    df = df.T
    print(df.to_latex(float_format="%.2f"))

long_exp()