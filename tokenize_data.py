from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import torch.utils.data
import os

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('opt-seq-pubmed-125m-003')
max_length = 1024

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=max_length, truncation=True)

cache_dir = './tmp' # Cache directory

data_files = {'train': ["data/trn.csv"],
              'test' : ["data/val.csv"]}
extension = 'csv'
raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)

preprocessing_num_workers = 4
overwrite_cache = False
removed_columns = ['text']

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    #batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=removed_columns,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset",
)

tokenized_datasets.save_to_disk('tokenized_data')

# Remove cache_dir
#os.system("rm -rf ./tmp")

##ds = load_from_disk('TMP')

