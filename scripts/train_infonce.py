import numpy as np
import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

from models.encoders import (
    VisualEncoder,
    AudioEncoder,
)

from models.contrastive import InfoNCELoss
from training.metrics import save_metrics


data = np.load(
    "data/trimodal_anomaly_dataset.npz"
)

images = torch.tensor(
    data["images"],
    dtype=torch.float32,
)

audios = torch.tensor(
    data["audios"],
    dtype=torch.float32,
)

images = images.permute(0, 3, 1, 2)
images = images / 255.0

dataset = TensorDataset(
    images,
    audios,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

visual_encoder = VisualEncoder(
    out_dim=256
)

audio_encoder = AudioEncoder(
    out_dim=256,
    sample_rate=8000,
)

loss_function = InfoNCELoss(
    temperature=0.07
)

optimizer = torch.optim.Adam(
    list(visual_encoder.parameters())
    + list(audio_encoder.parameters()),
    lr=0.001,
)

history = {
    "epoch": [],
    "loss": [],
}

for epoch in range(10):
    visual_encoder.train()
    audio_encoder.train()

    total_loss = 0.0

    for batch_images, batch_audios in loader:
        image_embeddings = visual_encoder(
            batch_images
        )

        audio_embeddings = audio_encoder(
            batch_audios
        )

        loss = loss_function(
            image_embeddings,
            audio_embeddings,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    history["epoch"].append(epoch + 1)
    history["loss"].append(average_loss)

    print(
        f"Epoch {epoch + 1}/10 | "
        f"InfoNCE loss: {average_loss:.4f}"
    )

torch.save(
    visual_encoder.state_dict(),
    "outputs/infonce_visual_encoder.pt",
)

torch.save(
    audio_encoder.state_dict(),
    "outputs/infonce_audio_encoder.pt",
)

print("Encodeurs InfoNCE sauvegardés.")

save_metrics(
    "infonce_training",
    {
        "experiment": "infonce_image_audio",
        "history": history,
    },
)

print("Métriques sauvegardées dans outputs/metrics/")
