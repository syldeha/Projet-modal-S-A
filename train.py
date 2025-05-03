import torch
import wandb
import hydra
from tqdm import tqdm
import logging
import os

from utils.sanity import show_images
    # Calculer et afficher le nombre de paramètres du modèle
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

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
        model.load_checkpoint(cfg.checkpoint_to_load)
        logger_std.info(f"Checkpoint loaded from {cfg.checkpoint_to_load}")
    except Exception as e:
        logger_std.info(f"Error loading checkpoint: {e}")
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
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
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss
            logger_std.info(f"----------------Epoch {epoch} val loss: {epoch_val_loss:.4f}----------------")
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
        - Val Loss: {epoch_val_loss:.4f}"""
    )

    if cfg.log:
        logger.finish()

    # S'assurer que le chemin du checkpoint se termine par .pt
    checkpoint_path = cfg.checkpoint_path
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path += '.pt'
    
    # Créer le répertoire pour les checkpoints s'il n'existe pas
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    logger_std.info(f"Saving model checkpoint to {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path) #important: quand on va utiliser torch.load(path, map_location=device) ça va retourner uniquement le state_dict
    # si on enregistrait sous forme de dictionnaire ca devait retourner le meme dictionnaire avec les meme clé de sauvegarde.


if __name__ == "__main__":
    train()
