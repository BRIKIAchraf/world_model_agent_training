# Contribuer au projet

## Workflow recommandé

1. Mettre `main` à jour :

   ```powershell
   git switch main
   git pull --ff-only
   ```

2. Créer une branche de travail :

   ```powershell
   git switch -c feature/nom-court
   ```

3. Modifier et tester localement :

   ```powershell
   python -m pytest -q
   python -m scripts.validate_env
   ```

4. Enregistrer les changements :

   ```powershell
   git add .
   git commit -m "feat: description courte"
   git push -u origin feature/nom-court
   ```

5. Ouvrir une Pull Request vers `main`.

La CI vérifie automatiquement Python 3.11 et 3.12. La livraison est créée uniquement après une CI réussie sur `main`.

## Types de commits

- `feat:` nouvelle fonctionnalité
- `fix:` correction
- `test:` ajout ou modification de tests
- `docs:` documentation
- `ci:` workflow GitHub Actions
- `refactor:` restructuration sans changement fonctionnel

