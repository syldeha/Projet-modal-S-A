from torch.utils.data import DataLoader
import torch
from data.dataset import Dataset

# def collate_fn_downsample(batch):
#     SOLID_PERFORMERS_LABEL_IDX = 
#     keep = []
#     for item in batch:
#         if item['labels'] == SOLID_PERFORMERS_LABEL_IDX:
#             if random.random() < 0.5:
#                 keep.append(item)
#         else:
#             keep.append(item)
#     if not keep:
#         # Assure au moins 1 exemple
#         keep.append(random.choice(batch))
#     return {k: [d[k] for d in keep] for k in keep[0]}  # format batch standard


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
        
        # Create train and val datasets in a single operation using the factory method
        self.train, self.val = Dataset.create_train_val_datasets(
            self.dataset_path,
            self.train_transform,
            self.metadata,
            split_ratio=0.8
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
            split_ratio=1,
            train_or_val_or_test="test"
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )