# Language model enables end-to-end accurate detection of cancer from cell-free DNA

## ACID


## Introduction of ACID

We present a language model ACID – Affordable Cancer Interception and Diagnostics – that can achieve high classification performance in the diagnosis of cancer exclusively from using raw cfDNA sequencing reads. We formulate ACID as an autoregressive language model. ACID is pretrained with language sentences that are obtained from concatenation of raw sequencing reads and diagnostic labels. ACID can achieve high accuracy with just 10,000 reads per sample. In summary, we present an affordable, simple yet efficient end-to-end paradigm for cancer detection using raw cfDNA sequencing reads.


## The following python packages are required:

```
python==3.7.16
torch==1.13.1
transformers==4.28.1
datasets==2.10.1
```
## Preprocess
python tokenize_data.py

## Train ACID
bash train.sh

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
ds = datasets.load_from_disk("data-tokenized/")["test"]


eos_token=tokenizer.eos_token
response_key_token_id=tokenizer.encode("### Response:")[0]
end_key_token_id=tokenizer.convert_tokens_to_ids(eos_token)

### An example.
example = torch.tensor(ds[1]["input_ids"]).unsqueeze(0)

gen_tokens = model.generate(example, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id, do_sample=False, max_new_tokens=30, top_p=0.98, top_k=0).cpu()

s = tokenizer.decode(gen_tokens[0])
ss = s.split("### Response:")[1].strip()[0:(-4)]
### Output of ss : hepatocellular carcinoma.
```
