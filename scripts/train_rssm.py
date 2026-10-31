import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

rewards = torch.tensor(
    data["rewards"],
    dtype=torch.float32,
)

images = images / 255.0
images = images.permute(0, 1, 4, 2, 3)

dataset = TensorDataset(
    images,
    actions,
    rewards,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

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

reward_head = nn.Linear(
    232,
    1,
)

parameters = (
    list(visual_encoder.parameters())
    + list(rssm.parameters())
    + list(observation_head.parameters())
    + list(reward_head.parameters())
)

optimizer = torch.optim.Adam(
    parameters,
    lr=0.0003,
)

history = {
    "epoch": [],
    "loss": [],
    "embedding_loss": [],
    "reward_loss": [],
    "kl_loss": [],
}

for epoch in range(10):
    total_loss = 0.0
    total_embedding_loss = 0.0
    total_reward_loss = 0.0
    total_kl_loss = 0.0

    for batch_images, batch_actions, batch_rewards in loader:
        batch_size, sequence_length = batch_images.shape[:2]

        flat_images = batch_images.reshape(
            batch_size * sequence_length,
            3,
            64,
            64,
        )

        embeddings = visual_encoder(
            flat_images
        )

        embeddings = embeddings.reshape(
            batch_size,
            sequence_length,
            256,
        )

        state = rssm.initial_state(
            batch_size=batch_size
        )

        embedding_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0

        for step in range(sequence_length - 1):
            state, info = rssm.observe_step(
                state,
                batch_actions[:, step],
                embeddings[:, step],
            )

            features = rssm.features(state)

            predicted_embedding = observation_head(
                features
            )

            predicted_reward = reward_head(
                features
            ).squeeze(-1)

            target_embedding = embeddings[
                :, step + 1
            ].detach()

            embedding_loss = (
                embedding_loss
                + torch.mean(
                    (predicted_embedding - target_embedding) ** 2
                )
            )

            reward_loss = (
                reward_loss
                + torch.mean(
                    (
                        predicted_reward
                        - batch_rewards[:, step]
                    ) ** 2
                )
            )

            prior_parameters = rssm.prior_network(
                state["deter"]
            )

            prior_distribution = rssm._distribution(
                prior_parameters
            )

            posterior_distribution = info[
                "posterior"
            ]

            kl_loss = (
                kl_loss
                + torch.distributions.kl_divergence(
                    posterior_distribution,
                    prior_distribution,
                ).mean()
            )

        number_of_steps = sequence_length - 1

        embedding_loss = (
            embedding_loss / number_of_steps
        )

        reward_loss = (
            reward_loss / number_of_steps
        )

        kl_loss = kl_loss / number_of_steps

        loss = (
            embedding_loss
            + reward_loss
            + 0.1 * kl_loss
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=100.0,
        )

        optimizer.step()

        total_loss += loss.item()
        total_embedding_loss += embedding_loss.item()
        total_reward_loss += reward_loss.item()
        total_kl_loss += kl_loss.item()

    batches = len(loader)

    average_loss = total_loss / batches
    average_embedding_loss = (
        total_embedding_loss / batches
    )
    average_reward_loss = (
        total_reward_loss / batches
    )
    average_kl_loss = total_kl_loss / batches

    history["epoch"].append(epoch + 1)
    history["loss"].append(average_loss)
    history["embedding_loss"].append(
        average_embedding_loss
    )
    history["reward_loss"].append(
        average_reward_loss
    )
    history["kl_loss"].append(
        average_kl_loss
    )

    print(
        f"Epoch {epoch + 1}/10 | "
        f"Loss: {average_loss:.4f}"
    )

torch.save(
    rssm.state_dict(),
    "outputs/rssm.pt",
)

torch.save(
    visual_encoder.state_dict(),
    "outputs/rssm_visual_encoder.pt",
)

torch.save(
    observation_head.state_dict(),
    "outputs/rssm_observation_head.pt",
)

torch.save(
    reward_head.state_dict(),
    "outputs/rssm_reward_head.pt",
)

save_metrics(
    "rssm_training",
    {
        "experiment": "rssm_world_model",
        "history": history,
    },
)

print(
    "Modèle RSSM et métriques sauvegardés."
)