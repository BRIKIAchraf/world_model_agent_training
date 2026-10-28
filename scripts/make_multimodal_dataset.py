import numpy as np

from envs.anomaly_env import AnomalyWarehouseEnv


env = AnomalyWarehouseEnv()

images = []
instructions = []
labels = []

for episode in range(100):
    observation, info = env.reset(seed=episode)

    for step in range(20):
        images.append(observation)
        instructions.append(info["target"])
        labels.append(info["target_faulty"])

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        if terminated or truncated:
            break

images = np.array(images, dtype=np.uint8)
instructions = np.array(instructions, dtype=np.int64)
labels = np.array(labels, dtype=np.int64)

np.savez_compressed(
    "data/multimodal_anomaly_dataset.npz",
    images=images,
    instructions=instructions,
    labels=labels,
)

print("Dataset multimodal créé.")
print("Images :", images.shape)
print("Instructions :", instructions.shape)
print("Labels :", labels.shape)
print("Classe normale :", np.sum(labels == 0))
print("Classe défectueuse :", np.sum(labels == 1))