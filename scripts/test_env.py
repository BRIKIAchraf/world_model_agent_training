from envs.warehouse_env import WarehouseEnv


env = WarehouseEnv()

observation, info = env.reset(seed=0)

print("Forme de l'observation :", observation.shape)
print("Type de l'observation :", observation.dtype)
print("Action possible :", env.action_space)


total_reward = 0.0

for step in range(200):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if terminated or truncated:
        print("Episode terminé au pas :", step + 1)
        break

print("Récompense totale :", total_reward)
print("Distance finale :", info["distance"])