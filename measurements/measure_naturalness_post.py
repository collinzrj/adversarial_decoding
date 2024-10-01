import json, sys
import numpy as np
import pandas as pd

def trigger_test():
    with open('../data/opt_trig_naturalness.json', 'r') as f:
        d = json.load(f)
        res = {}
        for opt in d:
            res[opt] = {}
            for k in d[opt]:
                res[opt][k] = np.sum(np.array(d[opt][k]) == True)
        
    print(pd.DataFrame(res))
    natural_scores = np.array(list(res['NaturalnessLLMOptimizer'].values()))
    adv_scores = np.array(list(res['PerplexityLLMOptimizer'].values()))

    with open('../data/doc_naturalness.json', 'r') as f:
        d = json.load(f)
        naturalness_nums = np.array([x[1] for x in d])
        print("")
        for score in range(1, 7):
            false_positive_rate = (1 - np.sum(naturalness_nums >= score) / 200).round(2)
            natural_detect_rate = (1 - np.sum(natural_scores >= score) / 10).round(2)
            adv_detect_rate = (1 - np.sum(adv_scores >= score) / 10).round(2)
            print(f"{score} & {false_positive_rate} & {natural_detect_rate} & {adv_detect_rate} \\\\")


def no_trigger_test():
    with open('../data/doc_naturalness.json', 'r') as f:
        d = json.load(f)
        naturalness_nums = np.array([x[1] for x in d])
    with open("../data/notrig_naturalness.json", 'r') as f:
        res = json.load(f)
        for opt in res:
            nums = []
            for k in res[opt]:
                nums.append(np.sum(np.array(k[1]) == True))
            # for score in range(1, 7):
            #     print(score, 1 - (np.sum(np.array(nums) >= score)) / len(nums))
            # print(" ")
    basic_nums = [np.sum(np.array(k[1]) == True) for k in res['PerplexityLLMOptimizer']]
    natural_nums = [np.sum(np.array(k[1]) == True) for k in res['NaturalnessLLMOptimizer']]
    for score in range(1, 7):
        false_positive_rate = (1 - np.sum(naturalness_nums >= score) / len(naturalness_nums)).round(2)
        basic_acc = (1 - (np.sum(np.array(basic_nums) >= score)) / len(basic_nums)).round(2)
        natural_acc = (1 - (np.sum(np.array(natural_nums) >= score)) / len(natural_nums)).round(2)
        print(f"{score} & {false_positive_rate} & {basic_acc} & {natural_acc} \\\\")
        

if __name__ == '__main__':
    if sys.argv[1] == 'trigger':
        trigger_test()
    else:
        no_trigger_test()