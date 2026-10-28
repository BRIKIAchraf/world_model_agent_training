import matplotlib.pyplot as plt

from envs.warehouse_env import WarehouseEnv


env = WarehouseEnv()

observation, info = env.reset(seed=0)

plt.imshow(observation)
plt.title("Observation du robot")
plt.axis("off")
plt.show()