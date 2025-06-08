from torch.utils.data import DataLoader
import torch
from data.dataset import Dataset
import numpy as np


class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=["title"],
        val_years=None,
        custom_val_split={2023: 400, 2022: 200, 2021: 200, 2020: 200, 2019: 200, 2018: 200, 2017:200},
        seed=42
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.custom_val_split = custom_val_split
        self.seed = seed
        # Create train and val datasets in a single operation using the factory method
        self.train, self.val = Dataset.create_train_val_datasets(
            self.dataset_path,
            self.train_transform,
            self.metadata,
            custom_val_split=self.custom_val_split,
            seed=self.seed
        )

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Validation dataloader."""
        if self.val is None:
            return None
        return DataLoader(
            self.val,
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
            custom_val_split=None,
            train_or_val_or_test="test",
            seed=100
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )