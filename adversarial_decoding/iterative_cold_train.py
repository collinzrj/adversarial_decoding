from naturalness_eval.natural_cold_simple import NaturalCOLDSimple
from naturalness_eval.naturalness_eval import further_train
from torch import set_default_device
import random
from datasets import load_dataset

ds = load_dataset("lmsys/lmsys-chat-1m")
sub_ds = ds['train'].shuffle().select(range(0, 10000))
res = [x['conversation'][0]['content'] for x in sub_ds]

set_default_device('cuda')

for _ in range(10):
    optimizer = NaturalCOLDSimple()
    results = []
    num_samples = 8
    for idx in range(num_samples):
        print(f"!!!!!!!!{idx}!!!!!!!!")
        results.append(optimizer.optimize(epoch_num=1, perturb_iter=30, stemp=0.3))
    del optimizer
    further_train(results, random.sample(res, num_samples))
    