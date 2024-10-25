from adversarial_decoding.naturalness_eval.natural_cold_simple import NaturalCOLDSimple
from torch import set_default_device
import json

def run():
    set_default_device('cuda')

    optimizer = NaturalCOLDSimple()
    results = []
    num_samples = 1000
    for idx in range(num_samples):
        print(f"!!!!!!!!{idx}!!!!!!!!")
        results.append(optimizer.optimize(epoch_num=1, perturb_iter=30, stemp=0.5))
        with open('./data/samples_unnatural_cold.json', 'w') as f:
            json.dump(results, f, indent=2)