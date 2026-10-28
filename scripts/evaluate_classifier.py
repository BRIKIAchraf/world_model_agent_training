import torch

from envs.warehouse_env import WarehouseEnv
from models.encoders import VisualEncoder


env = WarehouseEnv()

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

confusion = {
    (0, 0): 0,
    (0, 1): 0,
    (1, 0): 0,
    (1, 1): 0,
}

for seed in range(100):
    observation, info = env.reset(seed=seed)

    image = torch.tensor(
        observation,
        dtype=torch.float32,
    )

    image = image / 255.0
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    true_label = int(info["distance"] < 20.0)

    with torch.no_grad():
        prediction = model(image)
        predicted_label = prediction.argmax(dim=1).item()

    confusion[(true_label, predicted_label)] += 1


true_0 = confusion[(0, 0)] + confusion[(0, 1)]
true_1 = confusion[(1, 0)] + confusion[(1, 1)]

accuracy_0 = confusion[(0, 0)] / true_0 if true_0 else 0
accuracy_1 = confusion[(1, 1)] / true_1 if true_1 else 0

balanced_accuracy = (accuracy_0 + accuracy_1) / 2

print("Matrice de confusion :")
print("Vraie classe 0 -> classe 0 :", confusion[(0, 0)])
print("Vraie classe 0 -> classe 1 :", confusion[(0, 1)])
print("Vraie classe 1 -> classe 0 :", confusion[(1, 0)])
print("Vraie classe 1 -> classe 1 :", confusion[(1, 1)])

print()
print("Accuracy classe 0 :", round(accuracy_0, 3))
print("Accuracy classe 1 :", round(accuracy_1, 3))
print("Balanced accuracy :", round(balanced_accuracy, 3))