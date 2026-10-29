import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                embedding_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        visual_embedding,
        text_embedding,
    ):
        combined = torch.cat(
            [
                visual_embedding,
                text_embedding,
            ],
            dim=-1,
        )

        return self.network(combined)


class TriModalFusion(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim * 3,
                embedding_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        visual_embedding,
        audio_embedding,
        text_embedding,
    ):
        combined = torch.cat(
            [
                visual_embedding,
                audio_embedding,
                text_embedding,
            ],
            dim=-1,
        )

        return self.network(combined)