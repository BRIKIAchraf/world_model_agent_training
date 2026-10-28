import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WarehouseEnv(gym.Env):
    def __init__(self, size=64, max_steps=200):
        super().__init__()

        self.size = size
        self.max_steps = max_steps

        # Action : déplacement horizontal et vertical
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Observation : image RGB 64 x 64
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 3),
            dtype=np.uint8,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        self.robot = self.np_random.uniform(
            5, self.size - 5, size=2
        )

        self.target = self.np_random.uniform(
            5, self.size - 5, size=2
        )

        self.previous_distance = np.linalg.norm(
            self.robot - self.target
        )

        return self.render(), {}

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)

        # Déplacer le robot
        self.robot = self.robot + 2.0 * action
        self.robot = np.clip(self.robot, 0, self.size - 1)

        distance = np.linalg.norm(self.robot - self.target)

        # Petite récompense lorsque le robot se rapproche
        reward = -0.01
        reward += 0.1 * (
            self.previous_distance - distance
        )

        self.previous_distance = distance

        terminated = False

        # Objectif atteint
        if distance < 3:
            reward += 5.0
            terminated = True

        truncated = self.step_count >= self.max_steps

        return self.render(), float(reward), terminated, truncated, {
            "distance": float(distance)
        }

    def render(self):
        image = np.zeros(
            (self.size, self.size, 3),
            dtype=np.uint8,
        )

        def draw(position, color, radius=2):
            x = int(position[0])
            y = int(position[1])

            image[
                max(y - radius, 0): y + radius + 1,
                max(x - radius, 0): x + radius + 1,
            ] = color

        # Cible verte
        draw(self.target, (0, 255, 0))

        # Robot bleu
        draw(self.robot, (0, 0, 255))

        return image