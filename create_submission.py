import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data.dataset import Dataset


@hydra.main(config_path="configs", config_name="train")
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(
        Dataset(
            cfg.datamodule.dataset_path,
            "test",
            transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
            metadata=cfg.datamodule.metadata,
            train_or_val_or_test="test",
            val_year_min=2024
        ),
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
    )
    # - Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    print("Model loaded")

    # - Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])

    for i, batch in enumerate(test_loader):
        batch["image"] = batch["image"].to(device)
        with torch.no_grad():
            preds = model(batch).squeeze().cpu().numpy()
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"ID": batch["id"], "views": preds}),
            ]
        )
    
    # Save the submission file
    submission.to_csv(f"{cfg.root_dir}/submissission_EfficientNet_DISTILLBERT_LORA_3_04_val_3_50_train.csv", index=False)
    
    # Analyze prediction distribution by class
    view_thresholds = [0, 1000, 10000, 100000, 1000000, float('inf')]
    labels = ["Hidden Gems", "Rising Stars", "Solid Performers", "Viral Hits", "Mega Blockbusters"]
    
    def assign_view_class(views):
        for i in range(len(view_thresholds) - 1):
            if view_thresholds[i] <= views < view_thresholds[i+1]:
                return labels[i]
        return labels[-1]
    
    submission['view_class'] = submission['views'].apply(assign_view_class)
    class_distribution = submission['view_class'].value_counts()
    
    print("\nClass distribution of predictions:")
    for class_name, count in class_distribution.items():
        percentage = (count / len(submission)) * 100
        print(f"{class_name}: {count} videos ({percentage:.2f}%)")

if __name__ == "__main__":
    create_submission()