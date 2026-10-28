import torch

from models.encoders import VisualEncoder


encoder = VisualEncoder(out_dim=256)

# Un lot de 4 images artificielles
images = torch.rand(4, 3, 64, 64)

embeddings = encoder(images)

print("Forme des images :", images.shape)
print("Forme des embeddings :", embeddings.shape)