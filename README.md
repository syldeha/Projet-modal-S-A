# YouTube Views Prediction Challenge

Ce projet participe au challenge CSC_43M04_EP qui vise à prédire le nombre de vues qu'une vidéo YouTube recevra en utilisant son image miniature (thumbnail) et ses métadonnées associées (titre, description, etc.).

## 🚀 Fonctionnalités

- Prédiction du nombre de vues YouTube basée sur les thumbnails et métadonnées
- Support pour différents modèles d'apprentissage profond
- Intégration avec Weights & Biases pour le suivi des expériences
- Gestion de configuration avec Hydra
- Augmentation de données personnalisée
- Validation croisée et early stopping
- Génération automatique de soumissions

## 📋 Prérequis

- Python 3.12
- CUDA (pour l'accélération GPU)
- Conda

## 🛠️ Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-username/youtube-views-prediction.git
cd youtube-views-prediction
```

2. Créer et activer l'environnement conda :
```bash
conda create -n challenge python=3.12
conda activate challenge
```

3. Installer PyTorch :
- Pour CUDA >= 12.4 :
```bash
pip3 install torch torchvision torchaudio
```
- Pour CUDA 11.8 :
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📦 Structure du Projet

```
.
├── assets/              # Images et ressources
├── checkpoints/         # Modèles sauvegardés
├── configs/            # Configurations Hydra
├── data/               # Scripts de traitement des données
├── dataset/           # Données du challenge
├── models/            # Définitions des modèles
├── outputs/           # Sorties des expériences
├── utils/             # Utilitaires
├── wandb/             # Logs Weights & Biases
├── train.py           # Script principal d'entraînement
├── create_submission.py # Génération des soumissions
└── requirements.txt   # Dépendances Python
```

## 🎯 Utilisation

### Entraînement

Pour entraîner un modèle avec la configuration par défaut :
```bash
python train.py
```

Pour ajouter un préfixe au nom de l'expérience :
```bash
python train.py prefix=votre_prefix
```

### Entraînement avec Weights & Biases

Pour un suivi détaillé des expériences avec Weights & Biases, utilisez `train_cnn_wandb.py` :

1. Assurez-vous d'être connecté à Weights & Biases :
```bash
wandb login
```

2. Lancer l'entraînement avec le suivi W&B :
```bash
python train_cnn_wandb.py
```

Les avantages de l'utilisation de `train_cnn_wandb.py` :
- Suivi en temps réel des métriques d'entraînement
- Visualisation des courbes d'apprentissage
- Sauvegarde automatique des meilleurs modèles
- Comparaison facile entre différentes expériences
- Logging des hyperparamètres et de la configuration

Vous pouvez personnaliser l'expérience W&B en modifiant les paramètres dans le fichier de configuration correspondant dans le dossier `configs/`.

### Création de Soumission

Pour générer un fichier de soumission :
```bash
python create_submission.py model=config_of_the_exp checkpoint_path="checkpoints/votre_checkpoint.pth"
```

## 📊 Suivi des Expériences

Le projet utilise Weights & Biases pour le suivi des expériences. Les métriques suivantes sont enregistrées :
- Loss d'entraînement
- Loss de validation
- Nombre de paramètres du modèle
- Images de vérification (sanity checks)

## 🔧 Configuration

Les configurations sont gérées via Hydra dans le dossier `configs/`. Vous pouvez personnaliser :
- L'architecture du modèle
- Les hyperparamètres
- Les paramètres de l'optimiseur
- Les stratégies d'augmentation de données

## 📝 Notes

- Les checkpoints sont sauvegardés dans le dossier `checkpoints/`
- Le dossier `dataset/` est ignoré par git (voir `.gitignore`)
- Les logs sont sauvegardés dans `wandb/`

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📤 Ajouter des Éléments sur GitHub

### 1. Configuration Initiale
```bash
# Configurer votre identité Git
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

### 2. Ajouter des Fichiers
```bash
# Ajouter un fichier spécifique
git add chemin/vers/votre/fichier

# Ajouter tous les fichiers modifiés
git add .

# Vérifier les fichiers ajoutés
git status
```

### 3. Créer un Commit
```bash
# Créer un commit avec un message descriptif
git commit -m "Description de vos modifications"

# Voir l'historique des commits
git log
```

### 4. Synchroniser avec GitHub
```bash
# Récupérer les dernières modifications
git pull origin main

# Pousser vos modifications
git push origin main
```

### 5. Gestion des Branches
```bash
# Créer une nouvelle branche
git checkout -b nom-de-votre-branche

# Changer de branche
git checkout nom-de-la-branche

# Lister toutes les branches
git branch
```

### 6. Bonnes Pratiques
- Créez des branches pour chaque nouvelle fonctionnalité
- Écrivez des messages de commit clairs et descriptifs
- Faites des commits réguliers et atomiques
- Testez vos modifications avant de les pousser
- Mettez à jour votre branche avec la branche principale régulièrement

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.