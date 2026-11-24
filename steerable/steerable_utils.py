import torch.nn as nn
import torch
from transformers import Trainer
from emotion_vector_utils import EmotionAttributeDecoder, NUM_EMOTION_ATTRS
from alignment_function_utils import BaseAlignmentFunction
import random

## This model is trained to correctly choose between n options
class SteerablePluralismModel(nn.Module):
    def __init__(self, tokenizer):
        super(SteerablePluralismModel, self).__init__()
        self.emotion_vector_model = EmotionAttributeDecoder()
        self.alignment_function = BaseAlignmentFunction(NUM_EMOTION_ATTRS)
        self.tokenizer = tokenizer
        
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

        n_samples = len(input_ids_chosen)
    
        context_ids_chosen = inputs["context_ids_chosen"]
        context_ids_rejected = inputs["context_ids_rejected"]

        print(f"chosen input ids shape: {input_ids_chosen.shape}")
        embeddings = model.tokenizer(input_ids_chosen + input_ids_rejected)
        print(f"embeddings shape: {embeddings}")

        embeddings_chosen = embeddings[:n_samples]
        embeddings_rejected = embeddings[n_samples:]

        chosen_ind = random.randint(0, 1)
        options = [embeddings_chosen, embeddings_rejected] if chosen_ind == 0 else [embeddings_rejected, embeddings_chosen]

        n_contexts = len(context_ids_chosen)
        print(f"chosen context ids shape: {context_ids_chosen}")
        context_embeddings = model.tokenizer(context_ids_chosen + context_ids_rejected)
        print(f"context embeddings shape: {context_embeddings.shape}")
        context_embeddings_chosen = context_embeddings[:n_contexts]
        context_embeddings_rejected = context_embeddings[n_contexts:]

        fns = model(options, context_embeddings_chosen, context_embeddings_rejected)
        assert fns.shape() == (2, 1)





