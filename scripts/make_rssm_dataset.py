import numpy as np

from envs.anomaly_env import AnomalyWarehouseEnv


env = AnomalyWarehouseEnv()

number_of_sequences = 100
sequence_length = 50

all_images = []
all_actions = []
all_rewards = []
all_continuations = []

for sequence_index in range(number_of_sequences):
    observation, info = env.reset(
        seed=sequence_index
    )

    sequence_images = []
    sequence_actions = []
    sequence_rewards = []
    sequence_continuations = []

    for step in range(sequence_length):
        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        sequence_images.append(observation)
        sequence_actions.append(action)
        sequence_rewards.append(reward)

        is_continuing = not (
            terminated or truncated
        )

        sequence_continuations.append(
            float(is_continuing)
        )

        observation = next_observation

        if not is_continuing:
            observation, info = env.reset()

    all_images.append(sequence_images)
    all_actions.append(sequence_actions)
    all_rewards.append(sequence_rewards)
    all_continuations.append(sequence_continuations)

images = np.array(
    all_images,
    dtype=np.uint8,
)

actions = np.array(
    all_actions,
    dtype=np.float32,
)

rewards = np.array(
    all_rewards,
    dtype=np.float32,
)

continuations = np.array(
    all_continuations,
    dtype=np.float32,
)

np.savez_compressed(
    "data/rssm_sequences.npz",
    images=images,
    actions=actions,
    rewards=rewards,
    continuations=continuations,
)

print("Dataset RSSM créé.")
print("Images :", images.shape)
print("Actions :", actions.shape)
print("Rewards :", rewards.shape)
print("Continuations :", continuations.shape)