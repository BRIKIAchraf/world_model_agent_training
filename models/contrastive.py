import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature

    def forward(self, image_embeddings, audio_embeddings):
        image_embeddings = F.normalize(
            image_embeddings,
            dim=-1,
        )

        audio_embeddings = F.normalize(
            audio_embeddings,
            dim=-1,
        )

        similarity = (
            image_embeddings
            @ audio_embeddings.T
        )

        logits = similarity / self.temperature

        batch_size = logits.size(0)

        labels = torch.arange(
            batch_size,
            device=logits.device,
        )

        loss_image_to_audio = F.cross_entropy(
            logits,
            labels,
        )

        loss_audio_to_image = F.cross_entropy(
            logits.T,
            labels,
        )

        return (
            loss_image_to_audio
            + loss_audio_to_image
        ) / 2.0