import torch
import wandb
import hydra
from tqdm import tqdm
import logging
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.sanity import show_images
from models.embedder import train_bert_tiny,MyLLM
from models.trainFlanT5 import train_flant5
from transformers import AutoTokenizer

os.environ['HYDRA_FULL_ERROR'] = '1'

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
        wandb.init(project="V3_challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    val_years=[2013 , 2019, 2018 , 2020 , 2021 , 2022 , 2023 , [2019,2023]]

    try:
        if hasattr(cfg, 'checkpoint_to_load') and cfg.checkpoint_to_load:
            model.load_checkpoint(cfg.checkpoint_to_load, load_full=True)
            logger_std.info(f"Checkpoint loaded from {cfg.checkpoint_to_load}")
        else:
            logger_std.info("No checkpoint specified, starting from scratch")
    except Exception as e:
        logger_std.info(f"Error loading checkpoint: {e}")

    # # check  is we trained the model bert tiny or not
    # try:
    #     if cfg.model.train_bert_tiny:
    #         #entrainement du model de LLM bert tiny 
    #         _,bert_tiny_train_name = train_bert_tiny(train_set, val_set, tokenizer_name=cfg.model.tokenizer_path, model_name=cfg.model.model_path, epochs=cfg.model.epochs, device=device)
    #         #chargement du model bert tiny
    #         model.load_model(bert_tiny_train_name) #permet de charger uniquement les paramètres du model bert tinyentrainé 
    #     else:
    #         if cfg.model.load_model_path:
    #             model.load_model(cfg.model.load_model_path)
    # except Exception as e:
    #     logger_std.info(f"No training of bert tiny: {e}")
    # #check for the training of flant5
    # try:
    #     if cfg.model.train_flant5:
    #         logger_std.info(f"Training of flant5")
    #         _,flant5_train_name = train_flant5(train_set, val_set, tokenizer_name=cfg.model.tokenizer_path, model_name=cfg.model.model_path, epochs=cfg.model.epochs, device=device , learning_rate=cfg.model.learning_rate)
    #         model.load_model(flant5_train_name)
    # except Exception as e:
    #     logger_std.info(f"No training of flant5: {e}")

    for year in val_years:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_std.info(f"----------------------------------Using device: {device}-------------------------------")
        model = hydra.utils.instantiate(cfg.model.instance).to(device)
        total_params, trainable_params = count_parameters(model)
        logger_std.info(f"\n Nombre total de paramètres: {total_params:,}")
        logger_std.info(f"\n Nombre de paramètres entraînables: {trainable_params:,}")
        logger_std.info(f"\n Pourcentage de paramètres entraînables: {100 * trainable_params / total_params:.2f}%")
        optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
        scheduler=ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            # verbose=True,
            min_lr=1e-6
        )
        cfg.datamodule.val_years = year 
        logger_std.info(f"val_years: {str(cfg.datamodule.val_years)}")
        loss_fn = hydra.utils.instantiate(cfg.loss_fn)
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        train_loader = datamodule.train_dataloader()
        train_set = datamodule.train
        val_set = datamodule.val
        logger_std.info(f"train_set: {len(train_set)}")
        if val_set is not None:
            logger_std.info(f"val_set: {len(val_set)}")




        val_loader = datamodule.val_dataloader()
        # train_sanity = show_images(train_loader, name="assets/sanity/train_images")
        # (
        #     logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        #     if logger is not None
        #     else None
        # )
        if val_loader is not None:
            val_sanity = show_images(val_loader, name="assets/sanity/val_images")
            logger.log(
                {"sanity_checks/val_images": wandb.Image(val_sanity)}
            ) if logger is not None else None

        # Variable to track the best validation loss
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 3
        
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
                torch.cuda.empty_cache()
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
                    logger.log({f"loss_{str(cfg.datamodule.val_years)}": loss.detach().cpu().numpy()})
                    if logger is not None
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
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
                        f"train/loss_epoch_{str(cfg.datamodule.val_years)}": epoch_train_loss,
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

                #scheduler step
                scheduler.step(epoch_val_loss)
                val_metrics[f"val/loss_epoch_{str(cfg.datamodule.val_years)}"] = epoch_val_loss
                logger_std.info(f"----------------Epoch {epoch} val loss: {epoch_val_loss:.4f}----------------")
                #early stopping et sauvegarde du meilleur modèle
                # Save model if validation loss improved
                if epoch_val_loss < best_val_loss:
                    patience_counter = 0
                    best_val_loss = epoch_val_loss
                    logger_std.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {best_model_path}")
                    # torch.save(model.state_dict(), best_model_path)
                    logger.log({f"best_val_loss_{str(cfg.datamodule.val_years)}": best_val_loss})
                    (
                    logger.log(
                        {
                            "epoch": epoch,
                            **val_metrics,
                        }
                    )
                    )
                else:
                    patience_counter += 1
                if patience_counter >= max_patience:
                    logger_std.info(f"Early stopping at epoch {epoch}")
                    break
        logger.log({f"year_{str(cfg.datamodule.val_years)}": str(cfg.datamodule.val_years)})
        logger_std.info(
            f"""Epoch {epoch}: 
            Training metrics:
            - Train Loss: {epoch_train_loss:.4f},
            Validation metrics: 
            - Val Loss: {epoch_val_loss:.4f},
            Best validation loss: {best_val_loss:.4f}"""
        )
        #test de validation sur des données deje vu par le model
        if year==2013:
            logger_std.info(f"test de validation sur des données deje vu par le model")
            cfg.datamodule.val_years = [2016]
            datamodule=hydra.utils.instantiate(cfg.datamodule)
            test_loader = datamodule.val_dataloader()
            model.eval()
            test_metrics = {}
            epoch_test_loss = 0
            num_samples_test = 0
            for _, batch in enumerate(test_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                with torch.no_grad():
                    preds = model(batch).squeeze()
                    loss = loss_fn(preds, batch["target"])
                    epoch_test_loss += loss.detach().cpu().numpy() * len(batch["image"])
                    num_samples_test += len(batch["image"])
            epoch_test_loss /= num_samples_test
            logger.log({f"test/loss_epoch_test_données_vu_{str(year)}_{str(cfg.datamodule.val_years)}": epoch_test_loss})
            logger_std.info(f"test de validation sur des données deje vu par le model : {epoch_test_loss:.4f}")
        else :
            logger_std.info(f"test de validation sur des données non vu par le model")
            cfg.datamodule.val_years = year-1
            datamodule=hydra.utils.instantiate(cfg.datamodule)
            test_loader = datamodule.val_dataloader()
            model.eval()
            test_metrics = {}
            epoch_test_loss = 0
            num_samples_test = 0
            for _, batch in enumerate(test_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                with torch.no_grad():
                    preds = model(batch).squeeze()
                    loss = loss_fn(preds, batch["target"])
                    epoch_test_loss += loss.detach().cpu().numpy() * len(batch["image"])
                    num_samples_test += len(batch["image"])
            epoch_test_loss /= num_samples_test
            logger.log({f"test/loss_epoch_{str(year)}": epoch_test_loss,f"test/loss_epoch_year{str(year)}": str(cfg.datamodule.val_years)})
            logger_std.info(f"test de validation sur des données non vu par le model : {epoch_test_loss:.4f}")         
        

    if cfg.log:
        logger.finish()
    
    # Save final model
    logger_std.info(f"Saving final model checkpoint to {checkpoint_path}")
    # torch.save(model.state_dict(), checkpoint_path)
    
    # Inform about the best model
    logger_std.info(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")
    
    return best_val_loss

import copy
@hydra.main(config_path="configs", config_name="train")
def train_cross_validation(cfg, years_list=[2023,2021]):
    """
    Entrainer un modèle pour chaque année de validation (cross-validation temporelle).
    Args:
        cfg: configuration hydra complète (OmegaConf DictConfig)
        years_list: liste des valeurs d'années à utiliser comme validation
    Returns:
        results: dict année -> score (par ex best_val_loss)
    """
    all_scores = {}
    for year in years_list:
        print(f"\n\n=== Cross-validation : année de val {year} ===")
        # --- Clone la config (pour ne pas la modifier globalement)
        cfg_fold = copy.deepcopy(cfg)
        # --- Modifie la target val_years dans la config datamodule (format attendu par ton code)
        cfg_fold.datamodule.val_years = [year]  # ou cfg_fold.datamodule.val_years = year selon ta logique
        cfg_fold.experiment_name = f"{cfg.experiment_name}_val{year}"

        # --- Lancer l'entraînement "classique" (tu peux loger le run via wandb aussi)
        best_val_loss = train(cfg_fold)
        all_scores[year] = best_val_loss
        print(f"Année {year} - Best val loss : {best_val_loss:.5f}")

    print("\n====== Résumé cross-validation (val_years) ======")
    for year, score in all_scores.items():
        print(f"Année {year}: best val_loss = {score:.5f}")

    return all_scores

if __name__ == "__main__":
    train_cross_validation()