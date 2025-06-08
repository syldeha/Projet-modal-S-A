import torch
import pandas as pd
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)
import re
import numpy as np


def clean_description(description):

    """
    description : str
    => Renvoie un texte structuré et propre décrivant la vidéo
    """
    import re
    # Supprimer liens, emails, hashtags, mentions
    description = re.sub(r'https?:\/\/\S+|www\.\S+', '', description)
    description = re.sub(r'\S+@\S+', '', description)
    description = re.sub(r'#\w+', '', description)
    description = re.sub(r'@[A-Za-z0-9_]+', '', description)
    description = re.sub(r'[^\w\s,.!?\'":;()-]', '', description)
    
    # Liste augmentée des triggers d'appel à l'action (cta)
    calls_to_action = [
        "like", "subscribe", "sub", "share", "bell", "notification",
        "comment", "don't forget", "hit the", "click the", "smash the",
        "support", "checkout our merch", "thank you", "follow us", "stay tuned",
        "keep in the loop", "be kept in the loop", "published on our channel",
        "ring the bell", "please hit", "please consider", "button", "buy",
        "purchase", "get featured", "join us", "as new material is published",
        "as new content is published"
    ]
    
    lines = description.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip().lower()
        if any(cta in line_stripped for cta in calls_to_action):
            continue
        if line_stripped.startswith("►") or line_stripped.startswith("▪") or line_stripped.startswith("—"):
            continue
        if len(line_stripped) < 5:
            continue
        cleaned_lines.append(line.strip())
    
    description = ' '.join(cleaned_lines)
    description = re.sub(r'\s+', ' ', description).strip()
    return description

def encode_year_binary_categorical(df, year_col='year'):
    """
    Encode l'année de façon catégorielle binaire (année la plus ancienne = 1, etc.)
    Retourne un DataFrame avec une seule colonne year_new (string binaire sur 4 bits)
    """
    years_sorted = sorted(df[year_col].astype(int).unique())
    year_to_int = {year: idx+1 for idx, year in enumerate(years_sorted)}
    year_ints = df[year_col].astype(int).map(year_to_int)
    n_bits = 4
    year_binaries = year_ints.apply(lambda x: format(x, '04b'))
    #convertion en tensor de dimension 4
    year_binaries = year_binaries.apply(lambda x: torch.tensor([int(bit) for bit in x]))
    year_binaries = year_binaries.apply(lambda x: x.unsqueeze(0))
    year_binaries = year_binaries.apply(lambda x: x.unsqueeze(0))
    print(year_binaries.shape)
    df['year_new'] = year_binaries
    return df

def encode_date_sinusoidal(date_str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return [0]*6
    try:
        # Utilise pandas pour parser tous les formats courants, y compris les timezones
        date = pd.to_datetime(date_str)
    except:
        return [0]*6
    day_sin = np.sin(2 * np.pi * date.day / 31)
    day_cos = np.cos(2 * np.pi * date.day / 31)
    month_sin = np.sin(2 * np.pi * date.month / 12)
    month_cos = np.cos(2 * np.pi * date.month / 12)
    hour_sin = np.sin(2 * np.pi * date.hour / 24)
    hour_cos = np.cos(2 * np.pi * date.hour / 24)
    return [day_sin, day_cos, month_sin, month_cos, hour_sin, hour_cos]

def process_date_encoding(df):
    date_encodings = df['date'].apply(encode_date_sinusoidal)
    df['day_sin'] = date_encodings.apply(lambda x: x[0])
    df['day_cos'] = date_encodings.apply(lambda x: x[1])
    df['month_sin'] = date_encodings.apply(lambda x: x[2])
    df['month_cos'] = date_encodings.apply(lambda x: x[3])
    df['hour_sin'] = date_encodings.apply(lambda x: x[4])
    df['hour_cos'] = date_encodings.apply(lambda x: x[5])
    return df



def feature_percentage_dict(df, feature_name):
    """
    Retourne un dictionnaire {classe: pourcentage} pour une colonne du DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        feature_name (str): nom de la colonne à étudier
        
    Returns:
        dict: {valeur_unique: pourcentage}
    """
    counts = df[feature_name].value_counts(dropna=False)
    percentages = (counts / len(df) * 100).to_dict()
    return percentages

def print_label_distribution(df, label_column="labels", title=""):
    """
    Affiche la distribution (pourcentages) de la colonne 'labels' dans le DataFrame.
    """
    counts = df[label_column].value_counts(normalize=True) * 100
    print(f"---- {title} ----")
    for label, pct in counts.sort_index().items():
        print(f"  {label}: {pct:.2f}%")
    print("-----------------------")

#preprocessing des données
def process_global_mapping(*csv_paths):
    """
    Crée un mapping global cohérent pour tous les channels et années présents dans tous les CSV.
    
    Args:
        *csv_paths: Chemins vers tous les fichiers CSV (train_val.csv, test.csv, etc.)
    
    Returns:
        dict: Contient tous les mappings globaux nécessaires
            - 'channel_mapping': {channel_name: channel_real_name}
            - 'channel_id_mapping': {channel_name: channel_id}
            - 'total_channels': nombre total de channels
            - 'year_mapping': {year: binary_index}
            - 'year_sorted': liste des années triées
    """
    all_channels = set()
    all_years = set()
    
    # Collecter tous les channels et années uniques de tous les CSV
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            
            # Collecter les channels
            if 'channel' in df.columns:
                df['channel'] = df['channel'].fillna("").astype(str)
                all_channels.update(df['channel'].unique())
            
            # Collecter les années
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                years = df['year'].dropna().astype(int).unique()
                all_years.update(years)
                
        except Exception as e:
            print(f"Erreur lors de la lecture de {csv_path}: {e}")
            continue
    
    # Créer les mappings globaux pour les channels
    all_channels = sorted(list(all_channels))  # Ordre déterministe
    all_channels = [ch for ch in all_channels if ch and ch.strip()]  # Enlever vides
    
    global_channel_mapping = {channel: f"channel{i+1}" for i, channel in enumerate(all_channels)}
    global_channel_id_mapping = {channel: i for i, channel in enumerate(all_channels)}
    total_channels = len(all_channels)
    
    # Créer les mappings globaux pour les années
    years_sorted = sorted(list(all_years))
    year_to_int_mapping = {year: idx+1 for idx, year in enumerate(years_sorted)}
    
    print(f"Mapping global créé:")
    print(f"  - Channels uniques: {total_channels}")
    print(f"  - Années: {len(years_sorted)} ({min(years_sorted) if years_sorted else 'N/A'} - {max(years_sorted) if years_sorted else 'N/A'})")
    
    return {
        'channel_mapping': global_channel_mapping,
        'channel_id_mapping': global_channel_id_mapping,
        'total_channels': total_channels,
        'year_mapping': year_to_int_mapping,
        'years_sorted': years_sorted
    }

def make_prompt(entry):
    """
    entry: dict or pd.Series with keys: 'title', 'channel_real_name', 'year', 'description'
    """
    instruction="""
    pay attention to : 
    - the upload age of the video : 
    - the Language reach : 
    -Keyword-heavy : 
    -Celebrity/franchise hook:
    -video type:
    only choose one of theses categories:
    -low(0-1000)
    -medium(1000-10000)
    -high(10000-100000)
    -viral(100000-1000000)
    -top(1000000+) and don't add any other text. just provide your choice.
    """
    fields = [
        f"Title: {entry['title']}",
        f"Channel: {entry['channel_real_name']}",
        f"Year: {entry['year']}",
    ]
    # desc_clean = clean_description(entry['description'])
    # if desc_clean:
    #     fields.append(f"Description: {desc_clean}")
    prompt = "\n".join(fields)
    return prompt


def process_youtube_csv(csv_path, global_mappings=None):
    """
    Version modifiée qui utilise les mappings globaux
    """
    df = pd.read_csv(csv_path)
    
    for col in ["title", "channel", "description", "year"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    
    # ------ 1. Ajout de la colonne 'view_classes' (inchangé) ------
    if 'views' in df.columns:
        view_thresholds = [0, 1000, 10000, 100000, 1000000, float('inf')]
        labels = ["low", "medium", "high", "viral", "top"]
        
        def assign_view_class(views):
            for i in range(len(view_thresholds) - 1):
                if view_thresholds[i] <= views < view_thresholds[i+1]:
                    return labels[i]
            return labels[-1]
        
        df['view_classes'] = df['views'].apply(assign_view_class)

    # ------ 2. Channel mapping avec mappings globaux ------
    if 'channel' in df.columns and global_mappings:
        # Utiliser les mappings globaux
        df['channel_real_name'] = df['channel'].map(global_mappings['channel_mapping'])
        df['channel_id'] = df['channel'].map(global_mappings['channel_id_mapping'])
        
        # Gérer les channels inconnus
        unknown_mask = df['channel_id'].isna()
        if unknown_mask.any():
            print(f"Attention: {unknown_mask.sum()} channels inconnus dans {csv_path}")
            unknown_id = global_mappings['total_channels']
            df.loc[unknown_mask, 'channel_id'] = unknown_id
            df.loc[unknown_mask, 'channel_real_name'] = 'channel_unknown'
        
        def create_channel_onehot_global(channel_id):
            """Crée un vecteur one-hot avec la taille globale"""
            if pd.isna(channel_id):
                channel_id = global_mappings['total_channels']
            onehot = torch.zeros(global_mappings['total_channels'] + 1, dtype=torch.float32)  # +1 pour unknown
            onehot[int(channel_id)] = 1.0
            return onehot
        
        df['channel_onehot'] = df['channel_id'].apply(create_channel_onehot_global)
        
        # Stocker les métadonnées
        df.attrs['num_channels'] = global_mappings['total_channels'] + 1
        df.attrs['channel_mapping'] = global_mappings['channel_id_mapping']
    
    # ------ 3. Encodage des années (avec mappings globaux) ------
    if 'year' in df.columns and global_mappings:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        year_ints = df['year'].map(global_mappings['year_mapping'])
        
        # Gérer les années inconnues
        unknown_year_mask = year_ints.isna()
        if unknown_year_mask.any():
            print(f"Attention: {unknown_year_mask.sum()} années inconnues dans {csv_path}")
            year_ints = year_ints.fillna(0)  # 0 pour années inconnues
        
        n_bits = 4
        year_binaries = year_ints.apply(lambda x: format(int(x), '04b'))
        year_binaries = year_binaries.apply(lambda x: torch.tensor([int(bit) for bit in x]))
        year_binaries = year_binaries.apply(lambda x: x.unsqueeze(0).unsqueeze(0))
        df['year_new'] = year_binaries

    # ------ 4. Encodage des dates (inchangé) ------
    df = process_date_encoding(df)
    
    # ------ 5. Prompt resume (inchangé) ------
    df['prompt_resume'] = df.apply(make_prompt, axis=1)
    
    return df


def process_youtube_csv_test(df, global_mappings=None):
    """
    Version modifiée pour les données de test qui utilise les mappings globaux
    """
    # Nettoyage des colonnes de base
    for col in ["title", "channel", "description", "year"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    
    # Channel mapping avec mappings globaux
    if 'channel' in df.columns and global_mappings:
        df['channel_real_name'] = df['channel'].map(global_mappings['channel_mapping'])
        df['channel_id'] = df['channel'].map(global_mappings['channel_id_mapping'])
        
        # Gérer les channels inconnus
        unknown_mask = df['channel_id'].isna()
        if unknown_mask.any():
            print(f"Attention: {unknown_mask.sum()} channels inconnus dans le test set")
            unknown_id = global_mappings['total_channels']
            df.loc[unknown_mask, 'channel_id'] = unknown_id
            df.loc[unknown_mask, 'channel_real_name'] = 'channel_unknown'
        
        def create_channel_onehot_global(channel_id):
            if pd.isna(channel_id):
                channel_id = global_mappings['total_channels']
            onehot = torch.zeros(global_mappings['total_channels'] + 1, dtype=torch.float32)
            onehot[int(channel_id)] = 1.0
            return onehot
        
        df['channel_onehot'] = df['channel_id'].apply(create_channel_onehot_global)
        df.attrs['num_channels'] = global_mappings['total_channels'] + 1
        df.attrs['channel_mapping'] = global_mappings['channel_id_mapping']
    # Encodage des années avec mappings globaux
    if 'year' in df.columns and global_mappings:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        year_ints = df['year'].map(global_mappings['year_mapping'])
        
        unknown_year_mask = year_ints.isna()
        if unknown_year_mask.any():
            print(f"Attention: {unknown_year_mask.sum()} années inconnues dans le test")
            year_ints = year_ints.fillna(0)
        
        n_bits = 4
        year_binaries = year_ints.apply(lambda x: format(int(x), '04b'))
        year_binaries = year_binaries.apply(lambda x: torch.tensor([int(bit) for bit in x]))
        year_binaries = year_binaries.apply(lambda x: x.unsqueeze(0).unsqueeze(0))
        df['year_new'] = year_binaries
    # Encodage des dates et prompt resume
    df = process_date_encoding(df)
    df['prompt_resume'] = df.apply(make_prompt, axis=1)
    
    return df

def split_function(df, custom_val_split=None, year_column="year", seed=42):
    """
    Prend un dataframe et retourne (train_set, val_set) selon un split personnalisé :
    - custom_val_split : dict {année: n_exemples} pour la validation
    """
    import random
    df = df.copy()
    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
    if custom_val_split is not None:
        val_indices = []
        train_indices = []
        for year, n_val in custom_val_split.items():
            print(f"year: {year}, n_val: {n_val}")
            year_df = df[df[year_column] == year]  # Define year_df here
            if len(year_df) < n_val:
                raise ValueError(f"Pas assez de données pour l'année {year} (demandé {n_val}, dispo {len(year_df)})")
            val_idx = year_df.sample(n=n_val, random_state=seed).index.tolist()
            train_idx = list(set(year_df.index) - set(val_idx))
            val_indices.extend(val_idx)
            train_indices.extend(train_idx)
        other_train_indices = df[~df[year_column].isin(custom_val_split.keys())].index.tolist()
        train_indices.extend(other_train_indices)
        val_set = df.loc[val_indices]
        train_set = df.loc[train_indices]
        logger_std.info(f"Train set: {len(train_set)} rows / Val set (custom split): {len(val_set)} rows")
        print_label_distribution(train_set, "view_classes", "Train set")
        print_label_distribution(val_set, "view_classes", "Val set")
        return train_set, val_set
    else:
        val_years = list(custom_val_split.keys()) if custom_val_split else []
        mask_val = df[year_column].isin(val_years)
        val_set = df[mask_val]
        train_set = df[~mask_val]
        logger_std.info(f"Train set: {len(train_set)} rows / Val set (années {val_years}): {len(val_set)} rows")
        print_label_distribution(train_set, "view_classes", "Train set")
        print_label_distribution(val_set, "view_classes", "Val set")
        return train_set, val_set

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, custom_val_split=None, train_or_val_or_test:str=None, seed=42):
        """
        Args:
            dataset_path (str): path to the dataset
            split (str): split to use(train_val ou test)
            transforms (callable): transforms to apply to the image(elle se trouve dans datamodule.py)
            metadata (list): metadata to use(title, )
            train_or_val_or_test (str): train or val or test 'train' or 'val' or 'test'
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.custom_val_split = custom_val_split
        self.seed = seed
        self.global_mappings = process_global_mapping(f"{dataset_path}/train_val.csv", f"{dataset_path}/test.csv")
            
        # Regular processing for a single dataset
        # - read the info csvs
        if split == "train_val":
            print(f"{dataset_path}/{split}.csv")
            # info = pd.read_csv(f"{dataset_path}/{split}.csv")  #lire le fichier csv qui est dans dataset/train_val.csv
            info = process_youtube_csv(f"{dataset_path}/{split}.csv", self.global_mappings)
        elif split == "test":
            info = pd.read_csv(f"{dataset_path}/{split}.csv")  #lire le fichier csv qui est dans dataset/train_val.csv
            info = process_youtube_csv_test(info, self.global_mappings)

        # Split train/val personnalisé
        if train_or_val_or_test in ["train", "val"] and split != "test":
            train_set, val_set = split_function(info, custom_val_split, seed=self.seed)
            if train_or_val_or_test == "train":
                info = train_set
            elif train_or_val_or_test == "val":
                info = val_set
            # For test set, we use the entire dataset

        info["description"] = info["description"].fillna("").astype(str) #recuperation de la description
        info["title"] = info["title"].fillna("").astype(str)
        info["channel_real_name"] = info["channel_real_name"].fillna("").astype(str)
        info["meta"] = info[metadata].agg(" + ".join, axis=1) #concatenation des metadata
        if "views" in info.columns:
            self.targets = info["views"].values
        #creation des labels pour le dataset d'entrainement du LLM bert tiny
        if "view_classes" in info.columns:
            unique_classes = sorted(info["view_classes"].unique())
            self.class2idx = {c: i for i, c in enumerate(unique_classes)}
            self.idx2class = {i: c for c, i in self.class2idx.items()}
            self.labels = torch.tensor([self.class2idx[c] for c in info["view_classes"]], dtype=torch.long)
        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values
        self.description = info["description"].values
        self.title = info["title"].values
        self.prompt_resume = info["prompt_resume"].values
        self.year_new = info["year_new"].values
        self.day_sin = info["day_sin"].values
        self.day_cos = info["day_cos"].values
        self.month_sin = info["month_sin"].values
        self.month_cos = info["month_cos"].values
        self.hour_sin = info["hour_sin"].values
        self.hour_cos = info["hour_cos"].values
        self.channel_real_name = info["channel_real_name"].values
        self.channel_onehot = info["channel_onehot"].values

    @classmethod
    def create_train_val_datasets(cls, dataset_path, transforms, metadata, custom_val_split=None, seed=42):
        """
        Factory method to create both train and val datasets at once.
        Args:
            dataset_path (str): path to the dataset
            transforms (callable): transforms to apply to the images
            metadata (list): metadata to use
            custom_val_split (dict): dict {année: n} pour split personnalisé
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Create global mappings first
        global_mappings = process_global_mapping(f"{dataset_path}/train_val.csv", f"{dataset_path}/test.csv")
        
        # Read and process CSV file once with global mappings
        print(f"{dataset_path}/train_val.csv")
        info = process_youtube_csv(f"{dataset_path}/train_val.csv", global_mappings)
        
        # Split data
        if custom_val_split is None:
            train_info, val_info = info, None
            train_dataset = cls(dataset_path, "train_val", transforms, metadata, custom_val_split, "train", seed=seed)
            train_dataset._set_data(train_info, metadata)
            return train_dataset, None
        else:
            train_info, val_info = split_function(info, custom_val_split, seed=seed)
            train_dataset = cls(dataset_path, "train_val", transforms, metadata, custom_val_split, "train", seed=seed)
            train_dataset._set_data(train_info, metadata)
            val_dataset = cls(dataset_path, "train_val", transforms, metadata, custom_val_split, "val", seed=seed)
            val_dataset._set_data(val_info, metadata)
            return train_dataset, val_dataset
        
    def _set_data(self, info, metadata):
        """Helper method to set data directly, used by create_train_val_datasets"""
        info["description"] = info["description"].fillna("")
        info["title"] = info["title"].fillna("")
        info["meta"] = info[metadata].agg(" + ".join, axis=1)
        if "views" in info.columns:
            self.targets = info["views"].values
            
        # - ids
        self.ids = info["id"].values
        # - text

    def __len__(self):
        """
        return  int (taille du dataset)
        """
        return self.ids.shape[0]

    def __getitem__(self, idx):
        """
        permet de recuperer l'image positionnée à l'index idx
        """
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        
        #construction du dictionnaire qu'on va utiliser lors de l'entrainement
        if hasattr(self, "labels"):
            value = {
                "id": self.ids[idx],
                "image": image,
                "text": self.text[idx],
                "description": self.description[idx],
                "title": self.title[idx],
                "labels": self.labels[idx],
                "label_str": self.idx2class[self.labels[idx].item()],
                "prompt_resume": self.prompt_resume[idx],
                "year_new": self.year_new[idx],
                "day_sin": self.day_sin[idx],
                "day_cos": self.day_cos[idx],
                "month_sin": self.month_sin[idx],
                "month_cos": self.month_cos[idx],
                "hour_sin": self.hour_sin[idx],
                "hour_cos": self.hour_cos[idx],
                "channel_real_name": self.channel_real_name[idx],
                "channel_onehot": self.channel_onehot[idx]
            }
        else:
            value = {
                "id": self.ids[idx],
                "image": image,
                "text": self.text[idx],
                "description": self.description[idx],
                "title": self.title[idx],
                "prompt_resume": self.prompt_resume[idx],
                "year_new": self.year_new[idx],
                "day_sin": self.day_sin[idx],
                "day_cos": self.day_cos[idx],
                "month_sin": self.month_sin[idx],
                "month_cos": self.month_cos[idx],
                "hour_sin": self.hour_sin[idx],
                "hour_cos": self.hour_cos[idx],
                "channel_real_name": self.channel_real_name[idx],
                "channel_onehot": self.channel_onehot[idx]
            }
        
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return value