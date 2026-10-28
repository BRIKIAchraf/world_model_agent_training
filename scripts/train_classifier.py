import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from models.encoders import VisualEncoder


data = np.load("data/robot_distance.npz")

images = torch.tensor(data["images"], dtype=torch.float32)
labels = torch.tensor(data["labels"], dtype=torch.long)

# Transformation : (N, 64, 64, 3) -> (N, 3, 64, 64)
images = images.permute(0, 3, 1, 2)
images = images / 255.0

dataset = TensorDataset(images, labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(0),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
)

encoder = VisualEncoder(out_dim=256)
classifier = nn.Linear(256, 2)

model = nn.Sequential(
    encoder,
    classifier,
)

class_weights = torch.tensor(
    [0.686, 1.846],
    dtype=torch.float32,
)

loss_function = nn.CrossEntropyLoss(
    weight=class_weights
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

for epoch in range(5):
    model.train()
    total_loss = 0.0

    for batch_images, batch_labels in train_loader:
        predictions = model(batch_images)
        loss = loss_function(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            predictions = model(batch_images)
            predicted_classes = predictions.argmax(dim=1)

            correct += (predicted_classes == batch_labels).sum().item()
            total += len(batch_labels)

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/5 | "
        f"Loss : {total_loss:.4f} | "
        f"Accuracy : {accuracy:.3f}"
    )
torch.save(
    model.state_dict(),
    "outputs/visual_classifier.pt",
)

print("Modèle sauvegardé dans outputs/visual_classifier.pt")