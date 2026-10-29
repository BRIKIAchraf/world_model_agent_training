import torch

from models.contrastive import InfoNCELoss


torch.manual_seed(0)

image_embeddings = torch.randn(
    8,
    256,
)

audio_embeddings = torch.randn(
    8,
    256,
)

loss_function = InfoNCELoss(
    temperature=0.07,
)

loss = loss_function(
    image_embeddings,
    audio_embeddings,
)

print("Forme des embeddings image :")
print(image_embeddings.shape)

print("Forme des embeddings audio :")
print(audio_embeddings.shape)

print("Perte InfoNCE :")
print(round(loss.item(), 4))