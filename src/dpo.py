from transformers import AutoTokenizer, Trainer
from transformers.trainer_utils import EvalLoopOutput

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm


class DPOCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        chosen_texts = [example["chosen"] for example in batch]
        rejected_texts = [example["rejected"] for example in batch]

        def tokenize(txt):
            return self.tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        chosen = tokenize(chosen_texts)
        rejected = tokenize(rejected_texts)

        return {
            "chosen_ids": chosen["input_ids"],
            "chosen_mask": chosen["attention_mask"],
            "rejected_ids": rejected["input_ids"],
            "rejected_mask": rejected["attention_mask"]
        }


class DPOLoss(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _compute_batch_logp(logits, labels, attention_mask):
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        shifted_attention_mask = attention_mask[:, 1:].contiguous()
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        result = (per_token_log_probs * shifted_attention_mask).sum(dim=-1)
        return result

    def _get_loss(self, logratios, ref_logratios):
        logits = logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss

    def forward(self, model, ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
        chosen_outputs = model(
            input_ids=chosen_ids,
            attention_mask=chosen_mask
        )
        rejected_outputs = model(
            input_ids=rejected_ids,
            attention_mask=rejected_mask
        )

        chosen_logp = self._compute_batch_logp(
            logits=chosen_outputs.logits,
            labels=chosen_ids,
            attention_mask=chosen_mask
        )
        rejected_logp = self._compute_batch_logp(
            logits=rejected_outputs.logits,
            labels=rejected_ids,
            attention_mask=rejected_mask
        )

        with torch.no_grad():
            ref_chosen_outputs = ref_model(input_ids=chosen_ids, attention_mask=chosen_mask)
            ref_rejected_outputs = ref_model(input_ids=rejected_ids, attention_mask=rejected_mask)

            ref_chosen_logp = self._compute_batch_logp(ref_chosen_outputs.logits, chosen_ids, chosen_mask)
            ref_rejected_logp = self._compute_batch_logp(ref_rejected_outputs.logits, rejected_ids, rejected_mask)

        logratios = chosen_logp - rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp

        loss = self._get_loss(logratios, ref_logratios)
        return loss, chosen_logp, rejected_logp


class DPOTrainer(Trainer):
    def __init__(self, ref_model: nn.Module, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.loss = DPOLoss(beta)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, chosen_logp, rejected_logp = self.loss(model, self.ref_model, **inputs)

        if return_outputs:
            outputs = {
                "chosen_logp": chosen_logp,
                "rejected_logp": rejected_logp,
            }
            return (loss, outputs)

        return loss

    def evaluation_loop(self,
                        dataloader,
                        description,
                        prediction_loss_only=None,
                        ignore_keys=None,
                        metric_key_prefix="eval"):

        self.model.eval()
        loss_sum = 0.0
        accuracy_sum = 0.0
        num_batches = len(dataloader)

        for step, inputs in tqdm(enumerate(dataloader), total=num_batches, desc='Evaluating'):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)

            chosen_logp = outputs["chosen_logp"]
            rejected_logp = outputs["rejected_logp"]

            correct = (chosen_logp > rejected_logp).float()
            accuracy = correct.mean().item()

            loss_sum += loss.item()
            accuracy_sum += accuracy

        avg_loss = loss_sum / num_batches
        avg_accuracy = accuracy_sum / num_batches

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": avg_accuracy
        }

        self.log(metrics)
        result = {
            "predictions": None,
            "label_ids": None,
            "metrics": metrics,
            "num_samples": len(dataloader.dataset)
        }
        return EvalLoopOutput(**result)
