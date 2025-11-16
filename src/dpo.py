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
                padding="longest",
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


class DPOTrainer(Trainer):
    def __init__(self, model_ref: nn.Module, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.model_ref = model_ref
        self.beta = beta

    def _get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        shifted_attention_mask = attention_mask[:, 1:].contiguous()
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        per_token_log_probs = torch.gather(log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
        summed_log_probs = (per_token_log_probs * shifted_attention_mask).sum(dim=-1)
        return summed_log_probs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_outputs = model(input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"])
        rejected_outputs = model(input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"])

        pi_chosen_logps = self._get_batch_logps(chosen_outputs.logits, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
        pi_rejected_logps = self._get_batch_logps(rejected_outputs.logits, inputs["rejected_input_ids"], inputs["rejected_attention_mask"])

        with torch.no_grad():
            ref_chosen_outputs = self.model_ref(input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"])
            ref_rejected_outputs = self.model_ref(input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"])
            ref_chosen_logps = self._get_batch_logps(ref_chosen_outputs.logits, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
            ref_rejected_logps = self._get_batch_logps(ref_rejected_outputs.logits, inputs["rejected_input_ids"], inputs["rejected_attention_mask"])

        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits).mean()

        if return_outputs:
            outputs = {
                "pi_chosen_logps": pi_chosen_logps,
                "pi_rejected_logps": pi_rejected_logps,
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

            pi_chosen_logps = outputs["pi_chosen_logps"]
            pi_rejected_logps = outputs["pi_rejected_logps"]

            correct_prefs = (pi_chosen_logps > pi_rejected_logps).float()
            accuracy = correct_prefs.mean().item()

            total_loss += loss.item()
            total_accuracy += accuracy

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        metrics = {f"{metric_key_prefix}_loss": avg_loss, f"{metric_key_prefix}_accuracy": avg_accuracy}

        self.log(metrics)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(dataloader.dataset))
