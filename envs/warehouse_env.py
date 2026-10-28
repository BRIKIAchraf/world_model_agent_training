import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WarehouseEnv(gym.Env):
    def __init__(self, size=64, n_obstacles=4, max_steps=200):
        super().__init__()

        self.size = size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps

        # Action : déplacement horizontal et vertical
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Observation : image RGB
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
            5,
            self.size - 5,
            size=2,
        )

        self.target = self.np_random.uniform(
            5,
            self.size - 5,
            size=2,
        )

        self.obstacles = self.np_random.uniform(
            5,
            self.size - 5,
            size=(self.n_obstacles, 2),
        )

        self.previous_distance = np.linalg.norm(
            self.robot - self.target
        )

        observation = self.render()

        info = {
            "distance": float(self.previous_distance),
            "collision": False,
        }

        return observation, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        self.robot = self.robot + 2.0 * action

        self.robot = np.clip(
            self.robot,
            0,
            self.size - 1,
        )

        distance = np.linalg.norm(
            self.robot - self.target
        )

        reward = -0.01

        # Récompense si le robot se rapproche
        reward += 0.1 * (
            self.previous_distance - distance
        )

        self.previous_distance = distance

        obstacle_distances = np.linalg.norm(
            self.obstacles - self.robot,
            axis=1,
        )

        collision = bool(np.min(obstacle_distances) < 3)

        if collision:
            reward -= 1.0

        terminated = False

        # Récompense si le robot atteint la cible
        if distance < 3:
            reward += 5.0
            terminated = True

        truncated = self.step_count >= self.max_steps

        observation = self.render()

        info = {
            "distance": float(distance),
            "collision": collision,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        image = np.zeros(
            (self.size, self.size, 3),
            dtype=np.uint8,
        )

        def draw(position, color, radius=2):
            x = int(position[0])
            y = int(position[1])

            x_start = max(x - radius, 0)
            x_end = min(x + radius + 1, self.size)

            y_start = max(y - radius, 0)
            y_end = min(y + radius + 1, self.size)

            image[y_start:y_end, x_start:x_end] = color

        # Obstacles gris
        for obstacle in self.obstacles:
            draw(
                obstacle,
                color=(128, 128, 128),
                radius=3,
            )

        # Cible verte
        draw(
            self.target,
            color=(0, 255, 0),
            radius=2,
        )

        # Robot bleu
        draw(
            self.robot,
            color=(0, 0, 255),
            radius=2,
        )

        return image