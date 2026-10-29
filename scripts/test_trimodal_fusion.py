import torch

from envs.sound import machine_sound
from models.encoders import (
    VisualEncoder,
    AudioEncoder,
    TextEncoder,
)
from models.fusion import TriModalFusion


visual_encoder = VisualEncoder(
    out_dim=256
)

audio_encoder = AudioEncoder(
    out_dim=256,
    sample_rate=8000,
)

text_encoder = TextEncoder(
    n_instructions=3,
    out_dim=256,
)

fusion = TriModalFusion(
    embedding_dim=256
)

images = torch.rand(
    4,
    3,
    64,
    64,
)

audio = torch.tensor(
    [
        machine_sound(faulty=False),
        machine_sound(faulty=True),
        machine_sound(faulty=False),
        machine_sound(faulty=True),
    ],
    dtype=torch.float32,
)

instructions = torch.tensor(
    [0, 1, 2, 0],
    dtype=torch.long,
)

visual_embedding = visual_encoder(images)
audio_embedding = audio_encoder(audio)
text_embedding = text_encoder(instructions)

fused_embedding = fusion(
    visual_embedding,
    audio_embedding,
    text_embedding,
)

print(
    "Embedding visuel :",
    visual_embedding.shape,
)

print(
    "Embedding audio :",
    audio_embedding.shape,
)

print(
    "Embedding texte :",
    text_embedding.shape,
)

print(
    "Embedding trimodal :",
    fused_embedding.shape,
)