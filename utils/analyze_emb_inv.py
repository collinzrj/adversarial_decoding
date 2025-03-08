import json
import numpy as np

with open('/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_20250306_205328.json', 'r') as f:
    data = json.load(f)

cos_sim_list = np.array([item['cos_sim'] for item in data]) * 100
bleu_score_list = np.array([item['bleu_score'] for item in data]) * 100

print('mean cos sim', f'{np.mean(cos_sim_list):.2f}')
print('mean bleu score', f'{np.mean(bleu_score_list):.2f}')

print('std cos sim', f'{np.std(cos_sim_list):.2f}')
print('std bleu score', f'{np.std(bleu_score_list):.2f}')

print('median cos sim', f'{np.median(cos_sim_list):.2f}')
print('median bleu score', f'{np.median(bleu_score_list):.2f}')
