import torch
import wandb
import hydra
from tqdm import tqdm
import logging
import os
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.sanity import show_images
from models.embedder import train_bert_tiny,MyLLM
from transformers import AutoTokenizer



# Calculer et afficher le nombre de paramètres du modèle
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
  

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

def evaluate_class_accuracy(model, val_dataset, device):
    """Evaluate and display model accuracy by class"""
    from torch.utils.data import DataLoader
    
    # Create DataLoader for validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # Use a reasonable batch size
        shuffle=False,
        num_workers=4,
    )
    
    # Set model to evaluation mode
    model.eval()
    
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
        if "id" in batch:
            all_ids.extend(batch["id"])
    
    # Create dataframe for analysis
    if all_ids:
        results_df = pd.DataFrame({
            "ID": all_ids,
            "true_views": all_targets,
            "predicted_views": all_preds
        })
    else:
        results_df = pd.DataFrame({
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
    print(f"\n----- VALIDATION ACCURACY BY CLASS -----")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
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
    print(f"\n----- END OF VALIDATION ACCURACY -----")
    
    return results_df

@hydra.main(config_path="configs", config_name="train")
def train(cfg):
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_std.info(f"----------------------------------Using device: {device}-------------------------------")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    total_params, trainable_params = count_parameters(model)
    logger_std.info(f"\n Nombre total de paramètres: {total_params:,}")
    logger_std.info(f"\n Nombre de paramètres entraînables: {trainable_params:,}")
    logger_std.info(f"\n Pourcentage de paramètres entraînables: {100 * trainable_params / total_params:.2f}%")
    
    try:
        if hasattr(cfg, 'checkpoint_to_load') and cfg.checkpoint_to_load:
            model.load_checkpoint(cfg.checkpoint_to_load, load_full=True)
            logger_std.info(f"Checkpoint loaded from {cfg.checkpoint_to_load}")
        else:
            logger_std.info("No checkpoint specified, starting from scratch")
    except Exception as e:
        logger_std.info(f"Error loading checkpoint: {e}")
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    scheduler=ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=2, 
        # verbose=True,
        min_lr=1e-6
    )
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    train_set = datamodule.train
    val_set = datamodule.val
    logger_std.info(f"train_set: {len(train_set)}")
    logger_std.info(f"val_set: {len(val_set)}")


    # check  is we trained the model bert tiny or not
    if cfg.model.train_bert_tiny:
        #entrainement du model de LLM bert tiny 
        _,bert_tiny_train_name = train_bert_tiny(train_set, val_set, tokenizer_name="prajjwal1/bert-tiny", model_name="prajjwal1/bert-tiny", epochs=50, device=device)
        #chargement du model bert tiny
        model.load_model(bert_tiny_train_name) #permet de charger uniquement les paramètres du model bert tinyentrainé 


    val_loader = datamodule.val_dataloader()
    train_sanity = show_images(train_loader, name="assets/sanity/train_images")
    (
        logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        if logger is not None
        else None
    )
    if val_loader is not None:
        val_sanity = show_images(val_loader, name="assets/sanity/val_images")
        logger.log(
            {"sanity_checks/val_images": wandb.Image(val_sanity)}
        ) if logger is not None else None

    # Variable to track the best validation loss
    best_val_loss = float('inf')
    
    # S'assurer que le chemin du checkpoint se termine par .pt
    checkpoint_path = cfg.checkpoint_path
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path += '.pt'
    
    # Path for best model
    best_model_path = checkpoint_path.replace('.pt', '_best.pt')
    
    # Créer le répertoire pour les checkpoints s'il n'existe pas
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # -- loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        # -- loop over training batches
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            preds = model(batch).squeeze()
            
            # Calculate loss with regularization
            if hasattr(model, 'get_l2_regularization_loss'):
                reg_loss = model.get_l2_regularization_loss()
                loss = loss_fn(preds, batch["target"]) + reg_loss
            else:
                loss = loss_fn(preds, batch["target"])
                
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
        epoch_train_loss /= num_samples_train
        (
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss,
                }
            )
            if logger is not None
            else None
        )
        print(f"Epoch {epoch} train loss: {epoch_train_loss:.4f}")
        logger_std.info(f"Epoch {epoch} train loss: {epoch_train_loss:.4f}")

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        if val_loader is not None: 
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                with torch.no_grad():
                    preds = model(batch).squeeze()
                    
                # Calculate loss with regularization for consistent reporting
                if hasattr(model, 'get_l2_regularization_loss'):
                    reg_loss = model.get_l2_regularization_loss().detach()
                    loss = loss_fn(preds, batch["target"]) + reg_loss
                else:
                    loss = loss_fn(preds, batch["target"])
                    
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            scheduler.step(epoch_val_loss)
            val_metrics["val/loss_epoch"] = epoch_val_loss
            logger_std.info(f"----------------Epoch {epoch} val loss: {epoch_val_loss:.4f}----------------")
            
            # Save model if validation loss improved
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                logger_std.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)
                
                # Log to wandb if enabled
                if logger is not None:
                    logger.log({"best_val_loss": best_val_loss})
            
            (
                logger.log(
                    {
                        "epoch": epoch,
                        **val_metrics,
                    }
                )
                if logger is not None
                else None
            )

    logger_std.info(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f},
        Best validation loss: {best_val_loss:.4f}"""
    )

    if cfg.log:
        logger.finish()
    
    # Save final model
    logger_std.info(f"Saving final model checkpoint to {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path)
    
    # Inform about the best model
    logger_std.info(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")
    # #load the best model 
    # checkpoint = torch.load(best_model_path)
    # model.load_state_dict(checkpoint)
    # # Evaluate accuracy by class on validation set
    # logger_std.info("Evaluating accuracy by class on validation set...")
    # evaluate_class_accuracy(model, val_set, device)


if __name__ == "__main__":
    train()