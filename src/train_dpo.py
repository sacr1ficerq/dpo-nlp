import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
import datasets
from tqdm import tqdm
import logging
from typing import Dict
import copy

from dpo import DPOLoss, prepare_preference_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(batch, tokenizer, max_length=512):
    prompts = [item['prompt'] for item in batch]
    positive = [item['positive'] for item in batch]
    negative = [item['negative'] for item in batch]

    return prepare_preference_batch(
        tokenizer,
        prompts,
        positive,
        negative,
        max_length=max_length
    )


def train_dpo(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    output_dir: str = "./dpo_output",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    max_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    eval_steps: int = 500,
    save_steps: int = 1000,
):
    """
    Train a model using Direct Preference Optimization.

    Args:
        model: Policy model to train (will be updated)
        ref_model: Reference model (frozen)
        train_dataset: Training dataset with 'prompt', 'positive', 'negative' fields
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        beta: DPO temperature parameter
        max_length: Maximum sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Warmup steps for learning rate scheduler
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Initialize DPO loss
    dpo_loss = DPOLoss(beta=beta)

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length)
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length)
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )

    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting DPO training...")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Beta: {beta}")

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        epoch_metrics = {
            'reward_margin': 0,
            'reward_accuracy': 0,
            'positive_rewards': 0,
            'negative_rewards': 0
        }

        progress_bar = tqdm(train_dataloader, desc="Training...")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to("cuda:0") for k, v in batch.items()}

            # Compute log probs for policy model
            policy_positive_logps = dpo_loss.compute_log_probs(
                model,
                batch['positive_input_ids'],
                batch['positive_attention_mask'],
                batch['positive_labels']
            )

            policy_negative_logps = dpo_loss.compute_log_probs(
                model,
                batch['negative_input_ids'],
                batch['negative_attention_mask'],
                batch['negative_labels']
            )

            # Compute log probs for reference model
            with torch.no_grad():
                reference_positive_logps = dpo_loss.compute_log_probs(
                    ref_model,
                    batch['positive_input_ids'],
                    batch['positive_attention_mask'],
                    batch['positive_labels']
                )

                reference_negative_logps = dpo_loss.compute_log_probs(
                    ref_model,
                    batch['negative_input_ids'],
                    batch['negative_attention_mask'],
                    batch['negative_labels']
                )

            # Compute DPO loss
            loss, metrics = dpo_loss(
                policy_positive_logps,
                policy_negative_logps,
                reference_positive_logps,
                reference_negative_logps
            )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Update metrics
            epoch_loss += metrics['loss']
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # TODO: mb move to evaluation
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'reward_margin': f"{metrics['reward_margin']:.4f}",
                    'acc': f"{metrics['reward_accuracy']:.2%}"
                })

                # Evaluate
                if global_step % eval_steps == 0:
                    eval_metrics = evaluate(
                        model, ref_model, eval_dataloader,
                        dpo_loss, "cuda:0"
                    )
                    logger.info(f"Step {global_step} - Eval metrics: {eval_metrics}")
                    model.train()

                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_path = output_dir / f"checkpoint-{global_step}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Epoch summary
        avg_loss = epoch_loss / len(train_dataloader)
        avg_metrics = {k: v / len(train_dataloader) for k, v in epoch_metrics.items()}
        logger.info(f"Epoch {epoch + 1} - Avg loss: {avg_loss:.4f}, Metrics: {avg_metrics}")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete! Final model saved to {final_path}")

    return model


def evaluate(model, ref_model, eval_dataloader, dpo_loss):
    """Evaluate model on eval dataset."""
    model.eval()
    total_metrics = {
        'loss': 0,
        'reward_margin': 0,
        'reward_accuracy': 0,
        'positive_rewards': 0,
        'negative_rewards': 0
    }

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to("cuda:0") for k, v in batch.items()}

            policy_positive_logps = dpo_loss.compute_log_probs(
                model,
                batch['positive_input_ids'],
                batch['positive_attention_mask'],
                batch['positive_labels']
            )

            policy_negative_logps = dpo_loss.compute_log_probs(
                model,
                batch['negative_input_ids'],
                batch['negative_attention_mask'],
                batch['negative_labels']
            )

            reference_positive_logps = dpo_loss.compute_log_probs(
                ref_model,
                batch['positive_input_ids'],
                batch['positive_attention_mask'],
                batch['positive_labels']
            )

            reference_negative_logps = dpo_loss.compute_log_probs(
                ref_model,
                batch['negative_input_ids'],
                batch['negative_attention_mask'],
                batch['negative_labels']
            )

            _, metrics = dpo_loss(
                policy_positive_logps,
                policy_negative_logps,
                reference_positive_logps,
                reference_negative_logps
            )

            for key in total_metrics:
                total_metrics[key] += metrics[key]

    # Average metrics
    avg_metrics = {k: v / len(eval_dataloader) for k, v in total_metrics.items()}
    return avg_metrics


if __name__ == "__main__":
    # Configuration
    MODEL = 'EleutherAI/pythia-1.4b'
    RANK = 32
    SFT_CHECKPOINT_PATH = "/home/user/Desktop/NLP/hw4/sft.pt"
    TRAIN_DATASET_PATH = "/home/user/Desktop/NLP/hw4/train_dataset"
    EVAL_DATASET_PATH = "/home/user/Desktop/NLP/hw4/eval_dataset"
    OUTPUT_DIR = "/home/user/Desktop/NLP/hw4/dpo_output"

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = datasets.load_from_disk(TRAIN_DATASET_PATH)
    eval_dataset = datasets.load_from_disk(EVAL_DATASET_PATH)
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Verify dataset format
    assert 'prompt' in train_dataset.column_names
    assert 'positive' in train_dataset.column_names
    assert 'negative' in train_dataset.column_names

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load base model for policy
    logger.info("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL)
    policy_model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA (import your custom LoRA implementation)
    from lora import LoRALayer
    import torch.nn as nn

    # Freeze base model
    for param in policy_model.parameters():
        param.requires_grad = False

    # Add LoRA layers
    modules_to_replace = []
    for name, module in policy_model.named_modules():
        if isinstance(module, nn.Linear) and (
            name.endswith('dense') or name.endswith('query_key_value')
        ):
            modules_to_replace.append(name)

    for name in modules_to_replace:
        parent_name = ".".join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent_module = policy_model.get_submodule(parent_name)
        target_module = getattr(parent_module, child_name)

        lora_layer = LoRALayer(target_module, RANK)
        setattr(parent_module, child_name, lora_layer)
        logger.info(f"Replaced {name} with LoRALayer")

    # Load SFT checkpoint
    logger.info(f"Loading SFT checkpoint from {SFT_CHECKPOINT_PATH}")
    state = torch.load(SFT_CHECKPOINT_PATH, map_location="cpu")
    missing, unexpected = policy_model.load_state_dict(state, strict=False)
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Create reference model (frozen copy of SFT model)
    logger.info("Creating reference model...")

    policy_model.to("cuda:0")
    ref_model = copy.deepcopy(policy_model).to("cuda:0")

    # Train with DPO
    train_dpo(
        model=policy_model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        num_epochs=1,
        batch_size=4,  # Adjust based on your GPU memory
        learning_rate=5e-7,
        beta=0.1,
        max_length=360,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        eval_steps=50,
        save_steps=100,
    )

