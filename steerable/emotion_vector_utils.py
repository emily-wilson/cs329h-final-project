import torch.nn as nn

NUM_EMOTION_ATTRS = 4

class EmotionAttributeDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EmotionAttributeDecoder, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, NUM_EMOTION_ATTRS),
        )

    def forward(self, embeddings):
        attrs = self._model(embeddings)
        return attrs
