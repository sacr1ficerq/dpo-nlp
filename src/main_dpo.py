from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import torch
from torch import nn
import pathlib
import datasets

from lora import LoRALayer  # TODO fix

RANK = 32
MODEL = 'EleutherAI/pythia-1.4b'
CHECKPOINT_PATH = "/home/user/Desktop/NLP/hw4/sft.pt"

TRAIN_DATASET_PATH = "/home/user/Desktop/NLP/hw4/trainset"
EVAL_DATASET_PATH = "/home/user/Desktop/NLP/hw4/evalset"

assert pathlib.Path(CHECKPOINT_PATH).exists()

trainset = datasets.load_from_disk(TRAIN_DATASET_PATH)
evalset = datasets.load_from_disk(EVAL_DATASET_PATH)
print("Datasets loaded successfully.")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

print("Tokenizer loaded successfully.")

for param in model.parameters():
    param.requires_grad = False

modules_to_replace = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and (name.endswith('dense') or name.endswith('query_key_value')):
        modules_to_replace.append(name)

for name in modules_to_replace:
    parent_name = ".".join(name.split('.')[:-1])
    child_name = name.split('.')[-1]
    parent_module = model.get_submodule(parent_name)
    target_module = getattr(parent_module, child_name)

    lora_layer = LoRALayer(target_module, RANK)
    setattr(parent_module, child_name, lora_layer)
    print(f"Replaced {name} with LoRALayer")

state = torch.load(CHECKPOINT_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.to("cuda:0").eval()

print(model)
print("Model loaded successfully.")

# TODO: here will be dpo code
