import transformers
config = transformers.OPTConfig.from_json_file("config.json")
model = transformers.OPTForCausalLM(config)
model.save_pretrained("opt-seq-pubmed-125m")
