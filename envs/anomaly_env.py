import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AnomalyWarehouseEnv(gym.Env):
    def __init__(
        self,
        size=64,
        n_machines=3,
        n_obstacles=4,
        max_steps=200,
    ):
        super().__init__()

        self.size = size
        self.n_machines = n_machines
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 3),
            dtype=np.uint8,
        )

    def _sample_positions(
        self,
        count,
        existing=None,
        min_distance=8,
    ):
        positions = []

        if existing is not None:
            positions.extend(
                [position.copy() for position in existing]
            )

        attempts = 0
        max_attempts = 10000

        while len(positions) < count + (
            len(existing) if existing is not None else 0
        ):
            attempts += 1

            if attempts > max_attempts:
                raise RuntimeError(
                    "Impossible de placer les objets."
                )

            candidate = self.np_random.uniform(
                7,
                self.size - 7,
                size=2,
            )

            if all(
                np.linalg.norm(candidate - position)
                >= min_distance
                for position in positions
            ):
                positions.append(candidate)

        start_index = len(existing) if existing is not None else 0

        return np.array(
            positions[start_index:],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        self.machines = self._sample_positions(
            self.n_machines,
            min_distance=10,
        )

        self.obstacles = self._sample_positions(
            self.n_obstacles,
            existing=self.machines,
            min_distance=7,
        )

        self.robot = self._sample_positions(
            1,
            existing=(
                list(self.machines)
                + list(self.obstacles)
            ),
            min_distance=7,
        )[0]

        self.faulty = self.np_random.integers(
            low=0,
            high=2,
            size=self.n_machines,
        )

        self.target = int(
            self.np_random.integers(
                0,
                self.n_machines,
            )
        )

        self.previous_distance = np.linalg.norm(
            self.robot - self.machines[self.target]
        )

        observation = self.render()

        info = {
            "target": self.target,
            "target_faulty": int(
                self.faulty[self.target]
            ),
            "distance": float(
                self.previous_distance
            ),
        }

        return observation, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        self.robot = self.robot + 2.0 * action

        self.robot = np.clip(
            self.robot,
            0,
            self.size - 1,
        )

        distance = np.linalg.norm(
            self.robot - self.machines[self.target]
        )

        reward = -0.01

        reward += 0.1 * (
            self.previous_distance - distance
        )

        self.previous_distance = distance

        obstacle_distances = np.linalg.norm(
            self.obstacles - self.robot,
            axis=1,
        )

        collision = bool(
            np.min(obstacle_distances) < 3
        )

        if collision:
            reward -= 1.0

        terminated = distance < 3

        if terminated:
            reward += 5.0

        truncated = self.step_count >= self.max_steps

        observation = self.render()

        info = {
            "target": self.target,
            "target_faulty": int(
                self.faulty[self.target]
            ),
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

            image[
                y_start:y_end,
                x_start:x_end,
            ] = color

        def draw_target(position, faulty):
            target_color = (
                (255, 0, 0)
                if faulty
                else (0, 255, 0)
            )

            # Grand contour jaune
            draw(
                position,
                (255, 255, 0),
                radius=6,
            )

            # Centre rouge ou vert
            draw(
                position,
                target_color,
                radius=3,
            )

        # Obstacles gris
        for obstacle in self.obstacles:
            draw(
                obstacle,
                (128, 128, 128),
                radius=3,
            )

        # Machines normales ou défectueuses
        for index, machine in enumerate(self.machines):
            if index == self.target:
                continue

            machine_color = (
                (255, 0, 0)
                if self.faulty[index]
                else (0, 255, 0)
            )

            draw(
                machine,
                machine_color,
                radius=2,
            )

        # Machine cible clairement marquée
        draw_target(
            self.machines[self.target],
            self.faulty[self.target],
        )

        # Robot bleu
        draw(
            self.robot,
            (0, 0, 255),
            radius=2,
        )

        return image