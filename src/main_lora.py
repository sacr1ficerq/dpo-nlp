import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from lora import LoRALayer
from train import train


def main():
    MODEL = 'EleutherAI/pythia-1.4b'
    RANK = 32
    OUTPUT_DIR = './just_vibing'
    NUM_PROC = 8

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    print(f'{"Total params:":<35} {sum(p.numel() for p in model.parameters()):,}')
    print(f'{"Trainable params (before LoRA):":<35} {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

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

    print(f'{"Total params (after LoRA):":<35} {sum(p.numel() for p in model.parameters()):,}')
    print(f'{"Trainable params (after LoRA):":<35} {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    print('Loading dataset...')
    dataset = load_dataset("Anthropic/hh-rlhf")

    def tokenize_function(examples):
        return tokenizer(examples['chosen'], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names, num_proc=NUM_PROC)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    print('Training...')
    trainer = train(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=OUTPUT_DIR,
        tokenizer=tokenizer
    )
    return trainer


if __name__ == "__main__":
    main()
