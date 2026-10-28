import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Flatten(),
        )

        self.projection = nn.Linear(256 * 2 * 2, out_dim)

    def forward(self, image):
        features = self.network(image)
        embedding = self.projection(features)
        return embedding




class TextEncoder(nn.Module):
    def __init__(self, n_instructions=3, out_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=n_instructions,
            embedding_dim=out_dim,
        )

    def forward(self, instruction_id):
        return self.embedding(instruction_id)