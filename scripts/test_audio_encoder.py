import torch

from envs.sound import machine_sound
from models.encoders import AudioEncoder


audio_encoder = AudioEncoder(
    out_dim=256,
    sample_rate=8000,
)

audio_batch = torch.tensor(
    [
        machine_sound(faulty=False),
        machine_sound(faulty=True),
        machine_sound(faulty=False),
        machine_sound(faulty=True),
    ],
    dtype=torch.float32,
)

embeddings = audio_encoder(
    audio_batch
)

print(
    "Forme des sons :",
    audio_batch.shape,
)

print(
    "Forme des embeddings audio :",
    embeddings.shape,
)