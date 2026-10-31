import numpy as np
import torch
import torch.nn as nn

from models.encoders import VisualEncoder
from models.rssm import RSSM
from training.metrics import save_metrics


data = np.load(
    "data/rssm_sequences.npz"
)

images = torch.tensor(
    data["images"],
    dtype=torch.float32,
)

actions = torch.tensor(
    data["actions"],
    dtype=torch.float32,
)

images = images / 255.0
images = images.permute(0, 1, 4, 2, 3)

visual_encoder = VisualEncoder(
    out_dim=256
)

rssm = RSSM(
    action_dim=2,
    embed_dim=256,
    deter_dim=200,
    stoch_dim=32,
)

observation_head = nn.Linear(
    232,
    256,
)

visual_encoder.load_state_dict(
    torch.load(
        "outputs/rssm_visual_encoder.pt",
        weights_only=True,
    )
)

rssm.load_state_dict(
    torch.load(
        "outputs/rssm.pt",
        weights_only=True,
    )
)

observation_head.load_state_dict(
    torch.load(
        "outputs/rssm_observation_head.pt",
        weights_only=True,
    )
)

visual_encoder.eval()
rssm.eval()
observation_head.eval()

number_of_sequences = images.shape[0]
sequence_length = images.shape[1]

with torch.no_grad():
    flat_images = images.reshape(
        number_of_sequences * sequence_length,
        3,
        64,
        64,
    )

    embeddings = visual_encoder(
        flat_images
    )

    embeddings = embeddings.reshape(
        number_of_sequences,
        sequence_length,
        256,
    )

    state = rssm.initial_state(
        number_of_sequences
    )

    zero_action = torch.zeros(
        number_of_sequences,
        2,
    )

    state, _ = rssm.observe_step(
        state,
        zero_action,
        embeddings[:, 0],
    )

    rssm_errors = []
    persistence_errors = []

    maximum_horizon = min(
        10,
        sequence_length - 1,
    )

    for step in range(maximum_horizon):
        state, _ = rssm.imagine_step(
            state,
            actions[:, step],
        )

        predicted_embedding = observation_head(
            rssm.features(state)
        )

        target_embedding = embeddings[
            :, step + 1
        ]

        rssm_error = torch.mean(
            (
                predicted_embedding
                - target_embedding
            ) ** 2
        ).item()

        persistence_error = torch.mean(
            (
                embeddings[:, 0]
                - target_embedding
            ) ** 2
        ).item()

        rssm_errors.append(rssm_error)
        persistence_errors.append(
            persistence_error
        )

horizons = list(
    range(1, maximum_horizon + 1)
)

save_metrics(
    "rssm_evaluation",
    {
        "experiment": "rssm_multi_horizon",
        "history": {
            "epoch": horizons,
            "rssm_error": rssm_errors,
            "persistence_error": persistence_errors,
        },
    },
)

print(
    "Évaluation RSSM sauvegardée dans "
    "outputs/metrics/rssm_evaluation.json"
)