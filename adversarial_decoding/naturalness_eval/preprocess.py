from datasets import Dataset, DatasetDict
import random

natural_samples_path = '../data/samples_natural.txt'
unnatural_samples_path = '../data/samples_unnatural.txt'

with open(natural_samples_path) as f:
    lines = f.readlines()
    natural_lines = [{'text': eval(line), 'label': 1} for line in lines]
    random.shuffle(natural_lines)

with open(unnatural_samples_path) as f:
    lines = f.readlines()
    unnatural_lines = [{'text': eval(line), 'label': 0} for line in lines]
    random.shuffle(unnatural_lines)


train_natural_lines = natural_lines[:1000]
test_natural_lines = natural_lines[-50:]
train_unnatural_lines = unnatural_lines[:1000]
test_unnatural_lines = unnatural_lines[-50:]

train_dataset = Dataset.from_list(train_natural_lines + train_unnatural_lines)
test_dataset = Dataset.from_list(test_natural_lines + test_unnatural_lines)

train_dataset.save_to_disk('../data/naturalness_train_dataset')
test_dataset.save_to_disk('../data/naturalness_test_dataset')