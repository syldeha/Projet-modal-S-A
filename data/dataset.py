import torch
import pandas as pd
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

def process_youtube_csv(csv_path):
    """
    Fonction qui permet de modifier le fichier csv pour ajouter des colonnes , (view_classes, channel_real_name)
    """
    # Chargement du fichier CSV
    df = pd.read_csv(csv_path)
    
    # ------ 1. Ajout de la colonne 'view_classes' ------
    view_thresholds = [
        0,        # 0 vues
        1000,     # 1K vues
        10000,    # 10K vues
        100000,   # 100K vues
        1000000,  # 1M vues
        float('inf')  # Infini
    ]
    labels = [
        "Hidden Gems",
        "Rising Stars",
        "Solid Performers",
        "Viral Hits",
        "Mega Blockbusters"
    ]
    def assign_view_class(views):
        for i in range(len(view_thresholds) - 1):
            if view_thresholds[i] <= views < view_thresholds[i+1]:
                return labels[i]
        return labels[-1]
    df['view_classes'] = df['views'].apply(assign_view_class)
    
    # ------ 2. Ajout de la colonne 'channel_real_name' ------
    if 'channel' in df.columns:
        unique_channels = df['channel'].unique()
        channel_mapping = {channel: f"channel{i+1}" for i, channel in enumerate(unique_channels)}
        df['channel_real_name'] = df['channel'].map(channel_mapping)
    
    # Retourner le DataFrame modifié
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

def split_function(dataframe, split_ration):
    """
    Fonction qui permet de séparer les données en train et val
     dataframe : dataframe contant toutes les données
     split_ration : ration de séparation entre train et val

     return : train_set, val_set
    """
    test_size = 1-split_ration
    if test_size <= 0:   
        train_set = dataframe
        val_set = None
    else:
        train_set, val_set = train_test_split(dataframe, test_size=test_size, random_state=42, shuffle=True, stratify=dataframe['view_classes'])

    logger_std.info(f"train set : {feature_percentage_dict(train_set, 'view_classes')}")
    if val_set is not None:
        logger_std.info(f"val set : {feature_percentage_dict(val_set, 'view_classes')}")
    else:
        logger_std.info("val set : None")

    return train_set, val_set

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, split_ratio, train_or_val_or_test:str):
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
            
        # Regular processing for a single dataset
        # - read the info csvs
        if split == "train_val":
            print(f"{dataset_path}/{split}.csv")
            # info = pd.read_csv(f"{dataset_path}/{split}.csv")  #lire le fichier csv qui est dans dataset/train_val.csv
            info = process_youtube_csv(f"{dataset_path}/{split}.csv")
        elif split == "test":
            info = pd.read_csv(f"{dataset_path}/{split}.csv")  #lire le fichier csv qui est dans dataset/train_val.csv
        if train_or_val_or_test in ["train", "val"] and split != "test":
            train_set, val_set = split_function(info, split_ratio)
            if train_or_val_or_test == "train":
                info = train_set
            elif train_or_val_or_test == "val":
                info = val_set
        # For test set, we use the entire dataset
        
        info["description"] = info["description"].fillna("") #recuperation de la description
        info["title"] = info["title"].fillna("")
        info["meta"] = info[metadata].agg(" + ".join, axis=1) #concatenation des metadata
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values
        self.description = info["description"].values
        self.title = info["title"].values

    @classmethod
    def create_train_val_datasets(cls, dataset_path, transforms, metadata, split_ratio=0.9):
        """
        Factory method to create both train and val datasets at once.
        
        Args:
            dataset_path (str): path to the dataset
            transforms (callable): transforms to apply to the images
            metadata (list): metadata to use
            split_ratio (float): ratio for train/val split (default: 0.9)
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Read and process CSV file once
        print(f"{dataset_path}/train_val.csv")
        info = process_youtube_csv(f"{dataset_path}/train_val.csv")
        
        # Split data once
        train_info, val_info = split_function(info, split_ratio)
        
        # Create train dataset
        train_dataset = cls(dataset_path, "train_val", transforms, metadata, split_ratio, "train")
        # Manually set the data for train dataset
        train_dataset._set_data(train_info, metadata)
        
        # Create val dataset
        val_dataset = cls(dataset_path, "train_val", transforms, metadata, split_ratio, "val")
        # Manually set the data for val dataset
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
        self.text = info["meta"].values
        self.description = info["description"].values
        self.title = info["title"].values

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
        value = {
            "id": self.ids[idx],
            "image": image,
            "text": self.text[idx],
            "description": self.description[idx],
            "title": self.title[idx],
        }
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
