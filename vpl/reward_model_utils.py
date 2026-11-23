import torch.nn as nn
import torch

def load_data():
    from datasets import Audio, load_dataset

    pairs = load_dataset(
        "datapointai/text-to-speech-human-preferences-315k",
        "pairs",
        split="train",
        token=True,
    )

class RewardModel(nn.Module):
    def __init__(
        self,
        encoder_embed_dim,
        decoder_embed_dim,
        hidden_dim,
        latent_dim,
        llm_encoder,
        llm_contexts_encoder,
        fixed_contexts=False,
        fixed_llm_embeddings=False,
        use_causal_lm=False,
        use_attention_layer=False,
        use_transformer=False,
        concat_chosen_rejected=False
    ):
        super(RewardModel, self).__init__()
        self.llm_encoder = llm_encoder
        self.llm_contexts_encoder = llm_contexts_encoder
        self.pair_encoder = PairEncoder(encoder_embed_dim, hidden_dim, latent_dim)
        self.sequence_encoder = SequenceEncoder(latent_dim, latent_dim)
        self.decoder = Decoder(decoder_embed_dim + latent_dim, hidden_dim)

        self.latent_dim = latent_dim
        self.fixed_contexts = fixed_contexts
        self.fixed_llm_embeddings = fixed_llm_embeddings
        self.use_causal_lm = use_causal_lm
        self.use_attention_layer = use_attention_layer
        self.use_transformer = use_transformer
        self.concat_chosen_rejected = concat_chosen_rejected

        self.saved_embeddings = torch.Tensor(4, latent_dim)
        self.saved_embeddings.uniform_(-1, 1)

    def encode_pair(self, context_chosen, context_rejected):
        pass

    def encode_sequence(self, pair_embeddings, seq_start_end):
        pass

    def decode(self):
        pass

    def forward(
        self,
        target_chosen,
        target_rejected,
        context_chosen,
        context_rejected,
        seq_start_end
    ):
        pair_embed = self.encode_pair(context_chosen, context_rejected)
        mean, log_var = self.encode_sequence(pair_embed, seq_start_end)

        mean = torch.clamp(mean, -1, 1)
        log_var = torch.clamp(log_var, -1, 1)

        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        if not self.training:
            z = mean
        reward_chosen, reward_rejected = self.decode(target_chosen, target_rejected, z)

        return reward_chosen, reward_rejected, mean, log_var, z

class RewardTrainer():
    pass

