from envs.warehouse_env import WarehouseEnv


env = WarehouseEnv()
observation, info = env.reset(seed=0)

actions = [
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.0, 0.0],
]

for action in actions:
    observation, reward, terminated, truncated, info = env.step(action)

    print(
        "Action :", action,
        "| Récompense :", round(reward, 4),
        "| Distance :", round(info["distance"], 2),
    )