import torch
from torch import nn as nn

DEVICE = 'cuda'


class LoRALayer(nn.Module):
    def __init__(self, module: nn.Linear, rank: int):
        super().__init__()
        self.module = module
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
        self.adapter_A = nn.Parameter(torch.empty(module.in_features, rank, device=DEVICE, dtype=module.weight.dtype))
        nn.init.kaiming_uniform_(self.adapter_A, a=5 ** 0.5)
        self.adapter_B = nn.Parameter(torch.zeros(rank, module.out_features, device=DEVICE, dtype=module.weight.dtype))

    def forward(self, input):
        original_output = self.module(input)
        lora_output = (input @ self.adapter_A) @ self.adapter_B
        return original_output + lora_output
