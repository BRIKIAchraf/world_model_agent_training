import torch

from envs.warehouse_env import WarehouseEnv
from models.encoders import VisualEncoder


env = WarehouseEnv()
observation, info = env.reset(seed=123)

image = torch.tensor(
    observation,
    dtype=torch.float32,
)

image = image / 255.0
image = image.permute(2, 0, 1)
image = image.unsqueeze(0)

encoder = VisualEncoder(out_dim=256)
classifier = torch.nn.Linear(256, 2)

model = torch.nn.Sequential(
    encoder,
    classifier,
)

model.load_state_dict(
    torch.load(
        "outputs/visual_classifier.pt",
        weights_only=True,
    )
)

model.eval()

with torch.no_grad():
    logits = model(image)
    probabilities = torch.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1).item()

print("Distance réelle :", round(info["distance"], 2))
print("Classe prédite :", prediction)
print("Probabilité classe 0 :", round(probabilities[0, 0].item(), 3))
print("Probabilité classe 1 :", round(probabilities[0, 1].item(), 3))