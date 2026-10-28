# World Model Agent Training

Projet pédagogique construit étape par étape autour d'un agent World Model.

## État actuel

- Environnement 2D Gymnasium avec robot, cible et obstacles.
- Baseline d'agent aléatoire.
- Encodeur visuel CNN avec PyTorch.
- Encodeur d'instructions structurées et fusion par concaténation.
- Dataset de distance robot-cible.
- Classifieur visuel entraînable et scripts d'évaluation.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Vérification

```powershell
python -m scripts.check_setup
python -m scripts.validate_env
```

## Exemples

```powershell
python -m scripts.test_env
python -m scripts.make_dataset
python -m scripts.train_classifier
python -m scripts.evaluate_classifier
```

Les données générées et les poids entraînés restent locaux et sont exclus du dépôt Git.

