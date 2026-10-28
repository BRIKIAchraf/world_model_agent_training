import torch

from envs.warehouse_env import WarehouseEnv
from models.encoders import VisualEncoder


env = WarehouseEnv()
encoder = VisualEncoder(out_dim=256)

observation, info = env.reset(seed=0)

print("Image originale :", observation.shape)
print("Type original :", observation.dtype)

image = torch.tensor(observation, dtype=torch.float32)
image = image / 255.0

# HWC devient CHW
image = image.permute(2, 0, 1)

# CHW devient BCHW avec un batch de taille 1
image = image.unsqueeze(0)

print("Image pour PyTorch :", image.shape)
print("Valeur minimale :", image.min().item())
print("Valeur maximale :", image.max().item())

with torch.no_grad():
    embedding = encoder(image)

print("Embedding produit :", embedding.shape)