import torch

from envs.anomaly_env import AnomalyWarehouseEnv
from models.encoders import VisualEncoder


env = AnomalyWarehouseEnv()

encoder = VisualEncoder(out_dim=256)
classifier = torch.nn.Linear(256, 2)

model = torch.nn.Sequential(
    encoder,
    classifier,
)

model.load_state_dict(
    torch.load(
        "outputs/anomaly_classifier.pt",
        weights_only=True,
    )
)

model.eval()

correct = 0
total = 0

for seed in range(1000, 1100):
    observation, info = env.reset(seed=seed)

    image = torch.tensor(
        observation,
        dtype=torch.float32,
    )

    image = image / 255.0
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    true_label = info["target_faulty"]

    with torch.no_grad():
        logits = model(image)
        predicted_label = logits.argmax(dim=1).item()

    if predicted_label == true_label:
        correct += 1

    total += 1

accuracy = correct / total

print("Nouveaux épisodes testés :", total)
print("Prédictions correctes :", correct)
print("Accuracy :", round(accuracy, 3))