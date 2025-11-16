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
        chosen_texts = [f["chosen"] for f in batch]
        rejected_texts = [f["rejected"] for f in batch]

        def tokenize(txt):
            return self.tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        chosen_enc = tokenize(chosen_texts)
        rejected_enc = tokenize(rejected_texts)

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"]
        }


class DPOLoss(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logratios, ref_logratios):
        logits = logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss


class DPOTrainer(Trainer):
    def __init__(self, ref_model: nn.Module, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.loss = DPOLoss(beta)

    def _compute_batch_logp(self, logits, labels, attention_mask):
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_mask = inputs["rejected_attention_mask"]

        chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)

        pi_chosen_logp = self._compute_batch_logp(chosen_outputs.logits, chosen_ids, chosen_mask)
        pi_rejected_logp = self._compute_batch_logp(rejected_outputs.logits, rejected_ids, rejected_mask)

        with torch.no_grad():
            ref_chosen_outputs = self.ref_model(input_ids=chosen_ids, attention_mask=chosen_mask)
            ref_rejected_outputs = self.ref_model(input_ids=rejected_ids, attention_mask=rejected_mask)

            ref_chosen_logp = self._compute_batch_logp(ref_chosen_outputs.logits, chosen_ids, chosen_mask)
            ref_rejected_logp = self._compute_batch_logp(ref_rejected_outputs.logits, rejected_ids, rejected_mask)

        logratios = pi_chosen_logp - pi_rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp

        loss = self.loss(logratios, ref_logratios)

        if return_outputs:
            outputs = {
                "pi_chosen_logp": pi_chosen_logp,
                "pi_rejected_logp": pi_rejected_logp,
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
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)

        for step, inputs in tqdm(enumerate(dataloader), total=num_batches):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)

            pi_chosen_logp = outputs["pi_chosen_logp"]
            pi_rejected_logp = outputs["pi_rejected_logp"]

            correct_prefs = (pi_chosen_logp > pi_rejected_logp).float()
            accuracy = correct_prefs.mean().item()

            total_loss += loss.item()
            total_accuracy += accuracy

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        metrics = {f"{metric_key_prefix}_loss": avg_loss, f"{metric_key_prefix}_accuracy": avg_accuracy}

        self.log(metrics)
        result = {
            "predictions": None,
            "label_ids": None,
            "metrics": metrics,
            "num_samples": len(dataloader.dataset)
        }
        return EvalLoopOutput(**result)
