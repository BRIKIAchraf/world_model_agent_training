import numpy as np

from envs.warehouse_env import WarehouseEnv


env = WarehouseEnv()

images = []
labels = []

for episode in range(50):
    observation, info = env.reset(seed=episode)

    for step in range(20):
        distance = info.get("distance", 0.0)

        # Classe 1 : robot proche
        # Classe 0 : robot éloigné
        label = int(distance < 20.0)

        images.append(observation)
        labels.append(label)

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        if terminated or truncated:
            break

images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)

np.savez_compressed(
    "data/robot_distance.npz",
    images=images,
    labels=labels,
)

print("Dataset créé.")
print("Images :", images.shape)
print("Labels :", labels.shape)
print("Classe 0 :", np.sum(labels == 0))
print("Classe 1 :", np.sum(labels == 1))