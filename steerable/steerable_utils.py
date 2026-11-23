import torch.nn as nn
import torch
from transformers import Trainer
from emotion_vector_utils import EmotionAttributeDecoder, NUM_EMOTION_ATTRS
from alignment_function_utils import BaseAlignmentFunction

## This model is trained to correctly choose between n options
class SteerablePluralismModel(nn.Module):
    def __init__(self):
        super(SteerablePluralismModel, self).__init__()
        self.emotion_vector_model = EmotionAttributeDecoder()
        self.alignment_function = BaseAlignmentFunction(NUM_EMOTION_ATTRS)
        

    def forward(
        self,
        options,
        context_chosen,
        context_rejected
    ):
        ## Right now this only considers the chosen examples, should we incorporate the rejected examples also?
        context_attr_vectors = torch.tensor([self.emotion_vector_model(chosen) for chosen in context_chosen])
        z = torch.mean(context_attr_vectors, dim=1)

        n_options = len(options)
        options_attr_vectors = torch.tensor([self.emotion_vector_model(item) for item in options])
        fn = self.alignment_function(options_attr_vectors, torch.stack([z * n_options]))
        return fn

class SteerablePluralismTrainer(Trainer):
    @classmethod
    def per_sample_loss(cls, fn_chosen, fn_rejected):
        return 1.0 if fn_chosen < fn_rejected else 0.0

    def loss(self, fn_chosen, fn_rejected):
        return torch.mean(self.per_sample_loss(fn_chosen, fn_rejected))

    def accuracy(self, fn_chosen, fn_rejected):
        return torch.mean((fn_chosen > fn_rejected).float())

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        input_ids_chosen = inputs["input_ids_chosen"]
        input_ids_rejected = inputs["input_ids_rejected"]
        context_ids_chosen = inputs["context_ids_chosen"]
        context_ids_rejected = inputs["context_ids_rejected"]

        fns = model(options, context_ids_chosen, context_ids_rejected)
        assert fns.shape() == (2, 1)





