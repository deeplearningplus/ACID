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

## Prediction

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import json
import pandas as pd
from tqdm import tqdm
import sys
import gzip
import random
import datasets
import numpy as np
```


```python
model_path="opt-seq-125m"
model=AutoModelForCausalLM.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained(model_path)
ds = datasets.load_from_disk("tokenized_data")["test"]

eos_token=tokenizer.eos_token
response_key_token_id=tokenizer.encode("### Response:")[0]
eos_id = tokenizer.convert_tokens_to_ids(eos_token)
pad_id = tokenizer.pad_token_id

example = torch.tensor(ds[1]["input_ids"]).unsqueeze(0)

gen_tokens = model.generate(example, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_id, do_sample=False, max_new_tokens=30, top_p=0.98, top_k=0).cpu()

s = tokenizer.decode(gen_tokens[0])
ss = s.split("### Response:")[1].strip()[0:(-4)]
### Output of ss : hepatocellular carcinoma.
```
