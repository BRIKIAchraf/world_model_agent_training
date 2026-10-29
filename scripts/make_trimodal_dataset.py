import numpy as np

from envs.anomaly_env import AnomalyWarehouseEnv
from envs.sound import machine_sound


env = AnomalyWarehouseEnv()

images = []
audios = []
instructions = []
labels = []

for episode in range(100):
    observation, info = env.reset(
        seed=episode
    )

    for step in range(20):
        faulty = bool(
            info["target_faulty"]
        )

        audio = machine_sound(
            faulty=faulty,
            distance=info["distance"],
            rng=np.random.default_rng(
                episode * 100 + step
            ),
        )

        images.append(observation)
        audios.append(audio)
        instructions.append(info["target"])
        labels.append(info["target_faulty"])

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        if terminated or truncated:
            break

images = np.array(
    images,
    dtype=np.uint8,
)

audios = np.array(
    audios,
    dtype=np.float32,
)

instructions = np.array(
    instructions,
    dtype=np.int64,
)

labels = np.array(
    labels,
    dtype=np.int64,
)

np.savez_compressed(
    "data/trimodal_anomaly_dataset.npz",
    images=images,
    audios=audios,
    instructions=instructions,
    labels=labels,
)

print("Dataset trimodal créé.")
print("Images :", images.shape)
print("Audios :", audios.shape)
print("Instructions :", instructions.shape)
print("Labels :", labels.shape)
print("Classe normale :", np.sum(labels == 0))
print("Classe défectueuse :", np.sum(labels == 1))