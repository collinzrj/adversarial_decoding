from datasets import load_dataset

ds = load_dataset("lmsys/lmsys-chat-1m")
sub_ds = ds['train'].shuffle().select((0, 10000))
res = [x['conversation'][0]['content'] for x in sub_ds]