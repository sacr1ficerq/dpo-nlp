from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)

import logging
import copy
from datasets import load_dataset

import torch
from torch import nn

from dpo import DPOCollator, DPOTrainer
from lora import LoRALayer


torch.set_float32_matmul_precision('medium')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL = 'EleutherAI/pythia-1.4b'

NUM_PROC = 8
SFT_CHECKPOINT_PATH = "/home/user/Desktop/NLP/hw4/sft.pt"
TRAIN_DATASET_PATH = "/home/user/Desktop/NLP/hw4/train_dataset"
EVAL_DATASET_PATH = "/home/user/Desktop/NLP/hw4/eval_dataset"
OUTPUT_DIR = "/home/user/Desktop/NLP/hw4/dpo_output"

RANK = 32
TRAIN_SUBSET_SIZE = 1_000 # for testing only
EVAL_SUBSET_SIZE  = 100
MAX_LEN = 360

# Load tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load datasets
logger.info("Loading datasets...")
dataset = load_dataset('Anthropic/hh-rlhf')
logger.info(f"Dataser:\n{dataset}")


def is_within_max_len(example):
    chosen_tokens = tokenizer(example["chosen"], truncation=False)['input_ids']
    rejected_tokens = tokenizer(example["rejected"], truncation=False)['input_ids']
    return len(chosen_tokens) <= MAX_LEN and len(rejected_tokens) <= MAX_LEN

# dpo_train = dataset["train"].filter(is_within_max_len, num_proc=NUM_PROC)
# dpo_test = dataset["test"].filter(is_within_max_len, num_proc=NUM_PROC)
dpo_train = dataset["train"]
dpo_test = dataset["test"]

# logger.info(f"Original training dataset size: {len(dataset['train'])}")
# logger.info(f"Filtered training dataset size: {len(dpo_train)}")

# logger.info(f"Original evaluation dataset size: {len(dataset['test'])}")
# logger.info(f"Filtered evaluation dataset size: {len(dpo_test)}")

dpo_train_dataset = dpo_train.select(range(TRAIN_SUBSET_SIZE))
dpo_eval_dataset = dpo_test.select(range(EVAL_SUBSET_SIZE))

dpo_train_dataset.save_to_disk(TRAIN_DATASET_PATH)
dpo_eval_dataset.save_to_disk(EVAL_DATASET_PATH)

# Load base model for policy
logger.info("Loading policy model...")
policy_model = AutoModelForCausalLM.from_pretrained(MODEL)

policy_model.resize_token_embeddings(len(tokenizer))

# Freeze base model
for param in policy_model.parameters():
    param.requires_grad = False

# Add LoRA layers
modules_to_replace = []
for name, module in policy_model.named_modules():
    if isinstance(module, nn.Linear) and (name.endswith('dense') or name.endswith('query_key_value')):
        modules_to_replace.append(name)

for name in modules_to_replace:
    parent_name = ".".join(name.split('.')[:-1])
    child_name = name.split('.')[-1]
    parent_module = policy_model.get_submodule(parent_name)
    target_module = getattr(parent_module, child_name)

    lora_layer = LoRALayer(target_module, RANK)
    setattr(parent_module, child_name, lora_layer)
    # logger.info(f"Replaced {name} with LoRALayer")

# Load SFT checkpoint
logger.info(f"Loading SFT checkpoint from {SFT_CHECKPOINT_PATH}")
state = torch.load(SFT_CHECKPOINT_PATH, map_location="cpu")
missing, unexpected = policy_model.load_state_dict(state, strict=False)
logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

# Create reference model (frozen copy of SFT model)
logger.info("Creating reference model...")

policy_model.to("cuda:0")
ref_model = copy.deepcopy(policy_model).to("cuda:0")
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# --- Directory & Logging Constants ---
OUTPUT_DIR = "./dpo"
RUN_NAME = "dpo-v1.0"
LOGGING_DIR = f"{OUTPUT_DIR}/logs/{RUN_NAME}"

# --- Core Training Hyperparameters ---
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch size = 4
NUM_TRAIN_EPOCHS = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

SCHEDULER_TYPE = "reduce_lr_on_plateau"
SCHEDULER_KWARGS = {
    "mode": "min",
    "factor": 0.5,
    "patience": 3,
}

# --- Evaluation, Saving, & Logging ---
EVAL_STEPS = 50
EVAL_ON_START = True
SAVE_STEPS = 100
LOGGING_STEPS = 2
SAVE_TOTAL_LIMIT = 1
METRIC_FOR_BEST_MODEL = "accuracy"
REPORT_TO = "tensorboard"

# --- Strategy ---
LOGGING_STRATEGY = "steps"
EVALUATION_STRATEGY = "steps"
SAVE_STRATEGY = "steps"

training_args = TrainingArguments(
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,

    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    lr_scheduler_type=SCHEDULER_TYPE,
    lr_scheduler_kwargs=SCHEDULER_KWARGS,

    run_name=RUN_NAME,
    report_to=REPORT_TO,

    remove_unused_columns=False,

    logging_dir=LOGGING_DIR,
    logging_strategy=LOGGING_STRATEGY,
    logging_steps=LOGGING_STEPS,

    save_steps=SAVE_STEPS,
    save_strategy=SAVE_STRATEGY,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,

    eval_steps=EVAL_STEPS,
    eval_strategy=EVALUATION_STRATEGY,
    eval_on_start=EVAL_ON_START,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    greater_is_better=True,

    fp16=True,
    # torch_compile=True,
    # torch_compile_backend="inductor",
    # torch_compile_mode="default",
)

data_collator = DPOCollator(tokenizer=tokenizer, max_length=MAX_LEN)

dpo_trainer = DPOTrainer(
    beta=0.2,
    model=policy_model,
    ref_model=ref_model,

    args=training_args,

    train_dataset=dpo_train_dataset,
    eval_dataset=dpo_eval_dataset,
    data_collator=data_collator,
)

dpo_trainer.train()
dpo_trainer.save_model(OUTPUT_DIR)
