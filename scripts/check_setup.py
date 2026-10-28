from importlib import import_module


packages = [
    ("NumPy", "numpy"),
    ("PyTorch", "torch"),
    ("Gymnasium", "gymnasium"),
    ("MuJoCo", "mujoco"),
    ("PyYAML", "yaml"),
]


print("Vérification de l'environnement du projet")
print("------------------------------------------")

missing = []

for label, module_name in packages:
    try:
        module = import_module(module_name)
        version = getattr(module, "__version__", "version inconnue")
        print(f"[OK] {label} : {version}")
    except ImportError:
        print(f"[MANQUANT] {label}")
        missing.append(label)

print("------------------------------------------")

if missing:
    print("Bibliothèques manquantes :", ", ".join(missing))
else:
    print("Toutes les bibliothèques sont disponibles.")