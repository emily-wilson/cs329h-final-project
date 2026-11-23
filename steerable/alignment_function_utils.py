import torch.nn as nn
import torch

class BaseAlignmentFunction(nn.Module):
    def __init__(self, num_attrs):
        self._sample_model = nn.Linear(num_attrs, num_attrs)
        self._context_model = nn.Linear(num_attrs, num_attrs)

    def forward(self, sample_attributes, context_attributes):
        f_sample = self._sample_model(sample_attributes)
        f_context = self._context_model(context_attributes)
        return torch.linalg.norm(f_sample - f_context)