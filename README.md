# Language model for detection of cancer from cfDNA

## Introduction
We present a language model ACID – Affordable Cancer Interception and Diagnostics – that can achieve high classification performance in the diagnosis of cancer exclusively from using raw cfDNA sequencing reads. We formulate ACID as an autoregressive language model. ACID is pretrained with language sentences that are obtained from concatenation of raw sequencing reads and diagnostic labels. ACID can achieve high accuracy with just 10,000 reads per sample. In summary, we present an affordable, simple yet efficient end-to-end paradigm for cancer detection using raw cfDNA sequencing reads.


## Dependency
```
python==3.7.16
torch==1.13.1
transformers==4.28.1
datasets==2.10.1
```

## How to train on the example data?
### 1. Tokenizing input data
```python
python tokenize_data.py
```

### 2. Generate a model of random weight for loading
```python
python generate_random_weight.py
```
### 3. Training
```bash
bash train.sh
```

### Prediction

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import datasets
import numpy as np

model_path = "opt-seq-125m"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
ds = datasets.load_from_disk("tokenized_data")["test"]

eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

inputs = torch.tensor(ds[0]["input_ids"]).unsqueeze(0)
gen_tokens = model.generate(
    inputs, pad_token_id=pad_id, eos_token_id=eos_id,
    do_sample=False, max_new_tokens=30).cpu()
gen_texts = tokenizer.decode(gen_tokens[0])
print(gen_texts)
```
