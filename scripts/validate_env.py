from gymnasium.utils.env_checker import check_env

from envs.warehouse_env import WarehouseEnv


env = WarehouseEnv()

print("Vérification de l'environnement...")

check_env(env)

print("Environnement valide.")