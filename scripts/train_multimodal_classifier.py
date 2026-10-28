import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
)

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


data = np.load(
    "data/multimodal_anomaly_dataset.npz"
)

images = torch.tensor(
    data["images"],
    dtype=torch.float32
)

instructions = torch.tensor(
    data["instructions"],
    dtype=torch.long
)

labels = torch.tensor(
    data["labels"],
    dtype=torch.long
)

images = images.permute(0, 3, 1, 2)
images = images / 255.0

dataset = TensorDataset(
    images,
    instructions,
    labels
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(0)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

model = MultimodalClassifier()

# La classe 0 reçoit plus de poids
class_weights = torch.tensor(
    [1.4, 1.0],
    dtype=torch.float32
)

loss_function = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):
    model.train()
    total_loss = 0.0

    for (
        batch_images,
        batch_instructions,
        batch_labels,
    ) in train_loader:

        predictions = model(
            batch_images,
            batch_instructions
        )

        loss = loss_function(
            predictions,
            batch_labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (
            batch_images,
            batch_instructions,
            batch_labels,
        ) in test_loader:

            predictions = model(
                batch_images,
                batch_instructions
            )

            predicted_classes = predictions.argmax(
                dim=1
            )

            correct += (
                predicted_classes == batch_labels
            ).sum().item()

            total += len(batch_labels)

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/10 | "
        f"Loss: {total_loss:.4f} | "
        f"Accuracy: {accuracy:.3f}"
    )

torch.save(
    model.state_dict(),
    "outputs/multimodal_anomaly_classifier.pt"
)

print(
    "Modèle sauvegardé dans "
    "outputs/multimodal_anomaly_classifier.pt"
)