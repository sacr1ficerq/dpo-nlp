from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
)
from typing import Optional


def train(
    model,
    train_dataset,
    eval_dataset,
    output_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    compute_metrics: Optional[callable] = None,
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        # --- Core Training Parameters ---
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # effective batch size  8
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.1,

        eval_steps=50,
        save_steps=50,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=100,  # Log every 100 steps
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    return trainer


