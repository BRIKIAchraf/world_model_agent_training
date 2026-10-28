from envs.warehouse_env import WarehouseEnv


def test_environment_reset_and_step():
    env = WarehouseEnv()
    observation, info = env.reset(seed=0)

    assert observation.shape == (64, 64, 3)
    assert observation.dtype.name == "uint8"
    assert "distance" in info

    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    assert observation.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info

