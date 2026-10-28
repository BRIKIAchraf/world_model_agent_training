import numpy as np

from envs.anomaly_env import AnomalyWarehouseEnv


env = AnomalyWarehouseEnv()

images = []
labels = []

for episode in range(100):
    observation, info = env.reset(seed=episode)

    for step in range(20):
        images.append(observation)
        labels.append(info["target_faulty"])

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        if terminated or truncated:
            break

images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)

np.savez_compressed(
    "data/anomaly_dataset.npz",
    images=images,
    labels=labels,
)

print("Dataset d'anomalies créé.")
print("Images :", images.shape)
print("Labels :", labels.shape)
print("Classe normale (0) :", np.sum(labels == 0))
print("Classe défectueuse (1) :", np.sum(labels == 1))