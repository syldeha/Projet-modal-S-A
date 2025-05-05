import torch
import pandas as pd
from PIL import Image

def split_function(dataframe, split_ration):
    """
    Fonction qui permet de séparer les données en train et val
     dataframe : dataframe contant toutes les données
     split_ration : ration de séparation entre train et val

     return : train_set, val_set
    """
    n_train = int(split_ration * len(dataframe))
    n_val = len(dataframe) - n_train
    # Generate random indices
    indices = torch.randperm(len(dataframe), generator=torch.Generator().manual_seed(30))
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()
    # Return actual dataframes
    return dataframe.iloc[train_indices].reset_index(drop=True), dataframe.iloc[val_indices].reset_index(drop=True)






class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms,  metadata ,split_ratio, train_or_val_or_test:str ):
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
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
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
        # - transforms
        self.transforms = transforms

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
