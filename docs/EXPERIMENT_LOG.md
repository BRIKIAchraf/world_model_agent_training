# Journal des expériences

Ce fichier décrit les étapes réalisées, les problèmes rencontrés et les solutions appliquées.

## 1. Structure et environnement Python

### Réalisation

- Création de l'environnement virtuel `.venv`.
- Installation de NumPy, PyTorch, Gymnasium, MuJoCo, PyYAML et pytest.
- Ajout d'un script de vérification des dépendances.

### Validation

Les bibliothèques minimales sont disponibles et l'environnement Gymnasium est valide.

## 2. Environnement 2D

### Réalisation

Création de `WarehouseEnv` avec un robot, une cible, des obstacles, des actions continues et une récompense de rapprochement.

### Problèmes rencontrés

- Le déplacement du robot devait être borné entre `-1` et `1`.
- L'information `distance` manquait dans le résultat de `reset()`.

### Solutions

- Utilisation de `np.clip()` pour borner les actions et la position.
- Ajout de `distance` dans le dictionnaire `info` de `reset()`.

## 3. Premier classifieur visuel

### Réalisation

Création d'un CNN qui transforme une image `64 x 64 x 3` en embedding de `256` valeurs.

### Résultat

Le classifieur de distance a obtenu environ `80 %` d'accuracy.

### Limite

Le dataset était déséquilibré : la classe 0 était majoritaire. L'accuracy seule était donc insuffisante.

### Solution

Ajout de la balanced accuracy et de la matrice de confusion. Une pondération des classes a aussi été testée.

## 4. Machines normales et défectueuses

### Réalisation

Création de `AnomalyWarehouseEnv` avec plusieurs machines rouges ou vertes.

### Problème initial

Le label indiquait si la machine cible était défectueuse, mais l'image ne permettait pas d'identifier clairement la machine cible.

### Symptôme

Le modèle atteignait `100 %` sur le test interne mais environ `43 %` sur de nouveaux épisodes. C'était une généralisation faible et proche du hasard.

### Solutions appliquées

- Ajout d'un marqueur jaune autour de la machine cible.
- Séparation minimale entre les machines et les obstacles.
- Conservation de la couleur rouge ou verte au centre de la cible.

## 5. Fusion multimodale

### Réalisation

Création d'un dataset contenant :

- l'image de l'environnement ;
- l'identifiant structuré de la machine cible ;
- le label normal ou défectueux.

Le modèle combine maintenant `VisualEncoder`, `TextEncoder` et `ConcatFusion`.

### Problème rencontré

Une première évaluation multimodale donnait `51 %`. Le modèle chargé était ancien ou l'information cible était insuffisamment visible.

### Solution

- Régénération du dataset après chaque modification de l'environnement.
- Réentraînement obligatoire avant l'évaluation.
- Marqueur cible rendu plus grand et plus distinct.

### Résultat actuel

Le modèle multimodal atteint `67 %` sur 100 nouveaux épisodes, contre environ `59 %` pour le modèle visuel seul.

## Règle de reproductibilité

Après toute modification de l'environnement, respecter cet ordre :

```text
modifier l'environnement
        -> recréer le dataset
        -> réentraîner le modèle
        -> évaluer sur de nouveaux épisodes
```

Les fichiers `.npz` et `.pt` sont générés localement et exclus de Git via `.gitignore`. Les scripts, le code et ce journal sont versionnés.

