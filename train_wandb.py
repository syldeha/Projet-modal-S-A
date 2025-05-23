
from models.resnet_model import DinoV2WithQFormer
import torch
import wandb
import hydra
from tqdm import tqdm
import logging
import os
import torch.nn as nn
from omegaconf import OmegaConf
from utils.loss_fn import MSLELoss
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.sanity import show_images
from data.datamodule import DataModule
from models.resnet_model import DinoV2WithQFormer

# Helper for param count
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)


#mise en place des configs pour le train
sweep_config={
    'method': 'random',
    'metric': {
        'name': 'val/loss_epoch',
        'goal': 'minimize'
    },
    'parameters': {
        'lora_r': {
            'values': [16, 32,8]
        },
        'lora_alpha': {
            'values': [32, 16, 8]
        },
        'lora_dropout': {
            'values': [0.05, 0.1, 0.2]
        },
        'coef_1': {
            'values': [0.8, 0.6, 0.4]
        },
        'n_layers': {
            'values': [1, 2, 3]
        },
        'hidden_dim': {
            'values': [512, 256, 128]
        },
        'num_heads': {
            'values': [8]
        },
        'num_layers_q_former': {
            'values': [3]
        },
        'num_queries': {
            'values': [10]
        },
        'q_former_coef': {
            'values': [0]
        } ,
        'learning_rate': {
            'values': [4e-3, 1e-3]
        },
        'train_batch_size': {
            'values': [16, 32, 64]
        },
        'val_batch_size': {
            'values': [16, 32, 64]
        },
        'epochs': {
            'values': [5]
        }, 
        'experiment_name': {
            'values': ["dinov2_model"]
        }
        

    }
}

@hydra.main(config_path="configs", config_name="train")
def train_wandb(cfg):
    # Convert OmegaConf to dict for wandb compatibility
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    # Print-wise debug :
    # print(OmegaConf.to_yaml(cfg))  # (optionnel)

    # --- SÉCURITÉ si frozen ne figure pas dans la config (sinon KeyError !)
    if 'frozen' not in wandb_cfg:
        # Met à True ou False selon tes besoins !
        wandb_cfg['frozen'] = True

    with wandb.init(
        project="challenge_CSC_43M04_EP_training",
        config=wandb_cfg,
        name=wandb_cfg.get("experiment_name", "run"),
    ) as run:
        config = wandb.config
        logger_std.info(f"Config: {dict(config)}")  # wandb.config behaves like dict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_std.info(f"---------------------- Using device: {device} -----------------")

        # --- MODEL
        model = DinoV2WithQFormer(
            frozen=config.frozen,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            coef_1=config.coef_1,

            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers_q_former=config.num_layers_q_former,
            num_queries=config.num_queries,
            q_former_coef=config.q_former_coef
        )
        model.to(device)
        total_params, trainable_params = count_parameters(model)
        logger_std.info(f"\n Nombre total de paramètres: {total_params:,}")
        logger_std.info(f"\n Nombre de paramètres entraînables: {trainable_params:,}")
        logger_std.info(f"\n Pourcentage de paramètres entraînables: {100 * trainable_params / total_params:.2f}%")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=1, 
            min_lr=1e-6
        )
        loss_fn = MSLELoss()
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        logger_std.info(f"train_set: {len(datamodule.train)}")
        if datamodule.val is not None:
            logger_std.info(f"val_set: {len(datamodule.val)}")

        train_sanity = show_images(train_loader, name="assets/sanity/train_images")
        run.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        if val_loader is not None:
            val_sanity = show_images(val_loader, name="assets/sanity/val_images")
            run.log({"sanity_checks/val_images": wandb.Image(val_sanity)})

        # Path for best model
        best_model_path = f"{run.name}_best_model.pt"
        best_val_loss = float('inf')
        
        # --- EPOCH LOOP
        for epoch in tqdm(range(config.epochs), desc="Epochs"):
            model.train()
            epoch_train_loss = 0
            num_samples_train = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for i, batch in enumerate(pbar):
                torch.cuda.empty_cache()
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_train += len(batch["image"])
                pbar.set_postfix({"train/loss_step": loss.detach().cpu().item()})
            epoch_train_loss /= num_samples_train
            run.log({"epoch": epoch, "train/loss_epoch": epoch_train_loss})
            logger_std.info(f"Epoch {epoch} train loss: {epoch_train_loss:.4f}")

            # VALIDATION LOOP
            epoch_val_loss = 0
            num_samples_val = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch["image"] = batch["image"].to(device)
                    batch["target"] = batch["target"].to(device).squeeze()
                    preds = model(batch).squeeze()
                    loss = loss_fn(preds, batch["target"])
                    epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                    num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            run.log({"epoch": epoch, "val/loss_epoch": epoch_val_loss})
            logger_std.info(f"----------------Epoch {epoch} val loss: {epoch_val_loss:.4f}----------------")
            scheduler.step(epoch_val_loss)
            
            # SAVE BEST MODEL
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                logger_std.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {best_model_path}")
                # torch.save(model.state_dict(), best_model_path)
                run.log({"best_val_loss": best_val_loss})
            
        logger_std.info(
            f"""Epoch {epoch}: 
            Training metrics:
            - Train Loss: {epoch_train_loss:.4f},
            Validation metrics: 
            - Val Loss: {epoch_val_loss:.4f},
            Best validation loss: {best_val_loss:.4f}"""
        )

if __name__ == "__main__":
    import wandb
    sweep_id = wandb.sweep(sweep_config, project="challenge_CSC_43M04_EP_training")
    wandb.agent(sweep_id, function=train_wandb, count=20)