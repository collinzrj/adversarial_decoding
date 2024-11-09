# from adversarial_decoding.naturalness_eval.unnatural_cold import run
# from adversarial_decoding.naturalness_eval.naturalness_eval import train
from adversarial_decoding.experiment import trigger_experiment
# from adversarial_decoding.naturalness_eval.natural_cold_simple import NaturalCOLDSimple

# train()
# run()
trigger_experiment()

if __name__ == '__main__':
    # optimizer = NaturalCOLDSimple()
    # optimizer.optimize()
    train()