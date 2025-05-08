@hydra.main(config_path="configs", config_name="train")
def evaluate_model_on_validation(cfg):
    """Evaluate the model accuracy by class on validation data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load validation dataset
    val_dataset = Dataset(
        cfg.datamodule.dataset_path,
        "val",
        transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
        metadata=cfg.datamodule.metadata,
        split_ratio=cfg.datamodule.split_ratio,
        train_or_val_or_test="val"
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
    )
    
    # Load model
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded")
    
    # Define view classes
    view_thresholds = [0, 1000, 10000, 100000, 1000000, float('inf')]
    labels = ["Hidden Gems", "Rising Stars", "Solid Performers", "Viral Hits", "Mega Blockbusters"]
    
    def assign_view_class(views):
        for i in range(len(view_thresholds) - 1):
            if view_thresholds[i] <= views < view_thresholds[i+1]:
                return labels[i]
        return labels[-1]
    
    # Collect all predictions and ground truth
    all_preds = []
    all_targets = []
    all_ids = []
    
    for batch in val_loader:
        batch["image"] = batch["image"].to(device)
        with torch.no_grad():
            preds = model(batch).squeeze().cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(batch["views"].cpu().numpy())
        all_ids.extend(batch["id"])
    
    # Create dataframe for analysis
    results_df = pd.DataFrame({
        "ID": all_ids,
        "true_views": all_targets,
        "predicted_views": all_preds
    })
    
    # Classify into view classes
    results_df['true_class'] = results_df['true_views'].apply(assign_view_class)
    results_df['predicted_class'] = results_df['predicted_views'].apply(assign_view_class)
    
    # Compute correct class predictions
    results_df['correct'] = results_df['true_class'] == results_df['predicted_class']
    
    # Overall accuracy
    overall_accuracy = results_df['correct'].mean() * 100
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    
    # Class-wise accuracy only
    print("\nClass-wise Accuracy:")
    
    for label in labels:
        class_df = results_df[results_df['true_class'] == label]
        if len(class_df) > 0:
            accuracy = class_df['correct'].mean() * 100
            print(f"{label}:")
            print(f"  - Count: {len(class_df)} samples")
            print(f"  - Accuracy: {accuracy:.2f}%")
    
    # Confusion Matrix
    print("\nConfusion Matrix (%):")
    confusion = pd.crosstab(
        results_df['true_class'], 
        results_df['predicted_class'],
        normalize='index'
    ).round(3) * 100
    
    print(confusion)
    
    return results_df


if __name__ == "__main__":
    evaluate_model_on_validation()