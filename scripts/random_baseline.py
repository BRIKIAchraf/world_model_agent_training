import numpy as np

from envs.warehouse_env import WarehouseEnv


env = WarehouseEnv()

episode_rewards = []
successes = 0

for episode in range(10):
    observation, info = env.reset(seed=episode)

    total_reward = 0.0
    success = False

    for step in range(200):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated:
            success = True

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    if success:
        successes += 1

    print(
        f"Épisode {episode + 1}: "
        f"récompense={total_reward:.3f}, "
        f"distance={info['distance']:.2f}, "
        f"succès={success}"
    )

average_reward = np.mean(episode_rewards)

print()
print("Récompense moyenne :", round(average_reward, 3))
print("Nombre de succès :", successes, "/ 10")