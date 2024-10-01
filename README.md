## Controlled Generation of Natural Adversarial Documents for Stealthy Retrieval Poisoning

This is code for paper "Controlled Generation of Natural Adversarial Documents for Stealthy Retrieval Poisoning"

## Usage

Please make sure you have at least 40GB gpu memory

Make the data directory first
```
mkdir data
```

### Experiments
```
# to generate the trigger attack adversarial documents
python experiment.py trigger

# to generate the no trigger attack adversarial documents
python experiment.py no_trigger
```

### Prepare the vector db
If you want to run ASR eval, please first generate the vector dataset, this takes about 4 hours on A40

You can still run perplexity and naturalness eval doesn't rely on this
```
python sentence_create_emb_db.py
```

### Eval trigger attack

```
# to evaluate the ASR
cd measurements
python measure_asr.py trigger
python measure_asr_trigger_post.py

# to evaluate perplexity
python measure_perplexity.py

# to evaluate naturalness
export OPENAI_API_KEY=[your_openai_key]
python measure_naturalness.py trigger
```

### Eval no trigger attack
```
# to evaluate the ASR
cd mesurements
python measure_asr.py no_trigger
python measure_asr_cluster_post.py

# to evaluate naturalness
export OPENAI_API_KEY=[your_openai_key]
python measure_naturalness.py no_trigger
```

### Measre Real Doc naturalness
```
cd measurements
python measure_naturalness.py real_docs
```