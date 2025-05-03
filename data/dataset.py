import torch
import pandas as pd
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata):
        """
        Args:
            dataset_path (str): path to the dataset
            split (str): split to use(train_val ou test)
            transforms (callable): transforms to apply to the image(elle se trouve dans datamodule.py)
            metadata (list): metadata to use(title, )
        """
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")  #lire le fichier csv qui est dans dataset/train_val.csv
        info["description"] = info["description"].fillna("") #recuperation de la description
        info["meta"] = info[metadata].agg(" + ".join, axis=1) #concatenation des metadata
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values

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
        }
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
