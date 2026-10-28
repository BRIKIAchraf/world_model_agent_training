import torch
import torch.nn as nn

from envs.anomaly_env import AnomalyWarehouseEnv
from models.encoders import VisualEncoder, TextEncoder
from models.fusion import ConcatFusion


class MultimodalClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.visual_encoder = VisualEncoder(
            out_dim=256
        )

        self.text_encoder = TextEncoder(
            n_instructions=3,
            out_dim=256
        )

        self.fusion = ConcatFusion(
            embedding_dim=256
        )

        self.classifier = nn.Linear(256, 2)

    def forward(self, images, instructions):
        visual_embedding = self.visual_encoder(images)

        text_embedding = self.text_encoder(
            instructions
        )

        fused_embedding = self.fusion(
            visual_embedding,
            text_embedding
        )

        return self.classifier(fused_embedding)


env = AnomalyWarehouseEnv()
model = MultimodalClassifier()

model.load_state_dict(
    torch.load(
        "outputs/multimodal_anomaly_classifier.pt",
        weights_only=True
    )
)

model.eval()

correct = 0
total = 0

confusion = {
    (0, 0): 0,
    (0, 1): 0,
    (1, 0): 0,
    (1, 1): 0,
}

for seed in range(1000, 1100):
    observation, info = env.reset(seed=seed)

    image = torch.tensor(
        observation,
        dtype=torch.float32
    )

    image = image / 255.0
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    instruction = torch.tensor(
        [info["target"]],
        dtype=torch.long
    )

    true_label = int(info["target_faulty"])

    with torch.no_grad():
        logits = model(image, instruction)
        predicted_label = logits.argmax(
            dim=1
        ).item()

    confusion[
        (true_label, predicted_label)
    ] += 1

    if predicted_label == true_label:
        correct += 1

    total += 1


accuracy = correct / total

true_class_0 = (
    confusion[(0, 0)]
    + confusion[(0, 1)]
)

true_class_1 = (
    confusion[(1, 0)]
    + confusion[(1, 1)]
)

accuracy_class_0 = (
    confusion[(0, 0)] / true_class_0
    if true_class_0 > 0
    else 0.0
)

accuracy_class_1 = (
    confusion[(1, 1)] / true_class_1
    if true_class_1 > 0
    else 0.0
)

balanced_accuracy = (
    accuracy_class_0 + accuracy_class_1
) / 2


print("Nouveaux épisodes testés :", total)
print("Prédictions correctes :", correct)
print("Accuracy globale :", round(accuracy, 3))

print()
print("Matrice de confusion :")

print(
    "Vraie classe 0 -> classe 0 :",
    confusion[(0, 0)]
)

print(
    "Vraie classe 0 -> classe 1 :",
    confusion[(0, 1)]
)

print(
    "Vraie classe 1 -> classe 0 :",
    confusion[(1, 0)]
)

print(
    "Vraie classe 1 -> classe 1 :",
    confusion[(1, 1)]
)

print()
print(
    "Accuracy classe 0 :",
    round(accuracy_class_0, 3)
)

print(
    "Accuracy classe 1 :",
    round(accuracy_class_1, 3)
)

print(
    "Balanced accuracy :",
    round(balanced_accuracy, 3)
)