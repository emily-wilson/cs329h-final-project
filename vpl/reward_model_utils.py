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


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xc, xr, z):
        xc = torch.cat([xc, z], dim=1)
        xr = torch.cat([xr, z], dim=1)
        rc = self._model(xc)
        rr = self._model(xr)
        return rc, rr

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
        return self.pair_encoder(e_c, e_r)

    def encode_sequence(self, pair_embeddings, seq_start_end):
        return self.sequence_encoder(sequences, seq_start_end)

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

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        batch_size = inputs["input_ids_chosen"].shape[0]
        embeddings = model.tokenizer(
            inputs["input_ids_chosen"] + inputs["input_ids_rejected"]
        )

        embeddings_chosen = embeddings[:batch_size]
        embeddings_rejected = embeddings[batch_size:]

        context_input_ids_chosen = inputs["contexts_input_ids_chosen"]
        token_length_chosen = torch.eq(input_ids_chosen,
                                        model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1
        input_ids_rejected = inputs["contexts_input_ids_rejected"]
        token_length_rejected = torch.eq(input_ids_rejected,
                                            model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1

        with torch.no_grad():
            last_hidden_state_chosen = model.llm_contexts_encoder(
                input_ids=input_ids_chosen,
                output_hidden_states=True
            ).hidden_states[-1]

            contexts_embeddings_chosen = sum_embeddings / num_of_none_padding_tokens_chosen
            last_hidden_state_rejected = model.llm_contexts_encoder(
                input_ids=input_ids_rejected,
                output_hidden_states=True
            ).hidden_states[-1]

            contexts_embeddings_rejected = sum_embeddings / num_of_none_padding_tokens_rejected
        seq_start_end = inputs["seq_start_end"]
        user_type = torch.tensor(inputs["user_type"]).to(device).bfloat16()
        rewards_chosen, rewards_rejected, mean, log_var, z = model(
            embeddings_chosen,
            embeddings_rejected,
            contexts_embeddings_chosen,
            contexts_embeddings_rejected,
            seq_start_end,
            user_type,
            ground_truth_user_vector=False,  # todo: set to True for debug usage
        )

        reproduction_loss = self.loss(rewards_chosen, rewards_rejected)



