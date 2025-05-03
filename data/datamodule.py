from torch.utils.data import DataLoader
import torch
from data.dataset import Dataset


class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=["title"],
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        #on va séprarer les données en train et val
        self.train_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
        )
        self.n_train = int(0.8 * len(self.train_set))
        self.n_val = len(self.train_set) - self.n_train
        self.train_set, self.val_set = torch.utils.data.random_split(self.train_set, [self.n_train, self.n_val], generator=torch.Generator().manual_seed(42))


    def train_dataloader(self):
        """Train dataloader."""
        # train_set = Dataset(
        #     self.dataset_path,
        #     "train_val",
        #     transforms=self.train_transform,
        #     metadata=self.metadata,
        # )
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """TODO: 
        Implement a strategy to create a validation set from the train set.
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )