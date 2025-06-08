import torch
import wandb
import hydra
from tqdm import tqdm
import logging
import os
import torch.nn as nn
from omegaconf import OmegaConf
from utils.loss_fn import MSLELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.datamodule import DataModule
from models.Best_CNN_model import CNN_with_bert # <--- Importe ton modele
from models.dinov2_advanced import RealDinoV2WithQFormer , RealDinoV2WithQFormerAndSinusoidal, MultiModalModel
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)
sweep_config = {
    'method': 'random',    # ou "grid" pour tout croiser (plus long)
    'metric': {
        'name': 'val/loss_epoch',
        'goal': 'minimize'
    },
    'parameters': {
        'frozen':                {'values': [True, False]},
        # == Tête de régression ==
        'hidden_dim_1':   {'values': [128,256,512]},                   # hidden de 1er MLP
        'hidden_dim_2':  {'values': [128,256,512]},                     # hidden de 2e MLP
        #lora for text parameters
        'lora_r_text':              {'values': [16,32]},
        'lora_alpha_text':          {'values': [8,16]},
        'lora_dropout_text':        {'values': [0.2,0.1,0.05]},
        'coef_1':                  {'values': [0.5,0.75,1]},
        'coef_2':                  {'values': [0.5,0.75,1]},

        #lora for vision parameters
        'lora_r_vision':              {'values': [16,32]},
        'lora_alpha_vision':          {'values': [8,16]},
        'lora_dropout_vision':        {'values': [0.2,0.1,0.05]},
        'vision_coef':               {'values': [0.5,0.75,1]},
        'dropout_p':                 {'values': [0.2,0.1,0.05]},
        'init_method':               {'values': ["kaiming","xavier"]},
        'use_concatenation':         {'values': [True, False]},


        # == Optim/taille batch/train ==
        'learning_rate':       {'values': [2e-3,1e-3]},           # LR Adam
        'epochs':              {'values': [6]},                         # Adapter selon convergence/durée
        'experiment_name':     {'values': ["dynamic_cnn_bert"]},
        "lora_r_vision": {"values": [8,16,32]},
        "lora_alpha_vision": {"values": [8,16]},
        "lora_dropout_vision": {"values": [0.05,0.1,0.2]},
        'batch_or_layer':      {'values': ["batch"]},
        'classification_model_name': {'values': ["efficientnet_b0","efficientnet_b1","mobilenetv4_hybrid_medium"]},
        'vision_classification_model': {'values': [False]},
        'coef_image_1': {'values': [0.5,0.75,1]},
        'q_former_insertion': {'values': [False]},
        # 'num_heads_q_former': {'values': [2,4,8]},
        # 'num_layers_q_former': {'values': [2,4,6]},
        # 'num_queries_q_former': {'values': [8,16,24]},
        # 'q_former_coef': {'values': [0.5,0.75,1]},
    }
}


@hydra.main(config_path="configs", config_name="train")
def train_wandb(cfg):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(
        project="DINOV2_Q_former_classification_Multimodal_advanced_sweep",
        config=wandb_cfg,
        name=wandb_cfg.get("experiment_name", "run"),
    ) as run:
        config = wandb.config
        logger_std.info(f"Config: {dict(config)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_std.info(f"---------------------- Using device: {device} -----------------")

        # --- MODELE DYNAMIQUE ---
        model = RealDinoV2WithQFormer(
            frozen=config.frozen,
            lora_r_text=config.lora_r_text,
            lora_alpha_text=config.lora_alpha_text,
            lora_dropout_text=config.lora_dropout_text,
            coef_1=config.coef_1,
            coef_2=config.coef_2,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            vision_coef=config.vision_coef,
            use_concatenation=config.use_concatenation,
            dropout_p=config.dropout_p,
            init_method=config.init_method,
            # model_name=config.model_name,

            batch_or_layer=config.batch_or_layer,
            classification_model_name=config.classification_model_name,
            vision_classification_model=config.vision_classification_model,
            coef_image_1=config.coef_image_1,
            q_former_insertion=config.q_former_insertion,
            # num_heads_q_former=config.num_heads_q_former,
            # num_layers_q_former=config.num_layers_q_former,
            # num_queries_q_former=config.num_queries_q_former,
            # q_former_coef=config.q_former_coef,
            lora_r_vision=config.lora_r_vision,
            lora_alpha_vision=config.lora_alpha_vision,
            lora_dropout_vision=config.lora_dropout_vision,
        )
        model.to(device)
        total_params, trainable_params = count_parameters(model)
        logger_std.info(f"\n Nombre total de paramètres: {total_params:,}")
        logger_std.info(f"\n Nombre de paramètres entraînables: {trainable_params:,}")

        logger_std.info(f"\n Pourcentage de paramètres entraînables: {100 * trainable_params / total_params:.2f}%")
        #nombre de paramètres du model dans wandb
        run.log({"total_params": total_params, "trainable_params": trainable_params})
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=1,
            min_lr=1e-6
        )
        # Utilise ici la loss de ton choix
        loss_fn = MSLELoss()
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        logger_std.info(f"train_set: {len(datamodule.train)}")
        if datamodule.val is not None:
            logger_std.info(f"val_set: {len(datamodule.val)}")

        # (Optionnel) : visualisation d'images de batch
        # train_sanity = show_images(train_loader, name="assets/sanity/train_images")
        # run.log({"sanity_checks/train_images": wandb.Image(train_sanity)})

        best_model_path = f"{run.name}_best_model.pt"
        best_val_loss = float('inf')

        for epoch in tqdm(range(config.epochs), desc="Epochs"):
            model.train()
            epoch_train_loss = 0
            num_samples_train = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for i, batch in enumerate(pbar):
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

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                logger_std.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {best_model_path}")
                # torch.save(model.state_dict(), best_model_path)  # Décommente pour sauvegarder
                run.log({"best_val_loss": best_val_loss})

        logger_std.info(
            f"""Epoch {epoch}:
            Training metrics:
            - Train Loss: {epoch_train_loss:.4f},
            Validation metrics:
            - Val Loss: {epoch_val_loss:.4f},
            Best validation loss: {best_val_loss:.4f}"""
        )


sweep_config_dinov2 = {
    'method': 'random',    # ou "grid" pour tout croiser (plus long)
    'metric': {
        'name': 'val/loss_epoch',
        'goal': 'minimize'
    },
    'parameters': {
        # == Tête de régression ==
        'hidden_dim_1':   {'values': [256,512]},                   # hidden de 1er MLP
        'hidden_dim_2':  {'values': [128]},                     # hidden de 2e MLP
        #lora for text parameters
        'lora_r_text':              {'values': [16]},
        'lora_alpha_text':          {'values': [4]},
        'lora_dropout_text':        {'values': [0.2]},

        #lora for vision parameters
        'dropout_p':                 {'values': [0.1]},
        'init_method':               {'values': ["xavier","kaiming"]},
        'use_concatenation':         {'values': [True, False]},
        'fusion_dim': {'values': [512,256]},
        'dinov_vision_coef': {'values': [0.5,0.75]},


        # == Optim/taille batch/train ==
        'learning_rate':       {'values': [2e-3]},           # LR Adam
        'epochs':              {'values': [5]},                         # Adapter selon convergence/durée
        'experiment_name':     {'values': ["Multimodal_advanced_sweep"]},
        "lora_r_vision": {"values": [8]},
        "lora_alpha_vision": {"values": [8]},
        "lora_dropout_vision": {"values": [0.1]},

        'classification_model_name': {'values': ["efficientnet_b0","efficientnet_b1","mobilenetv4_hybrid_medium", "resnet50"]},
        'use_vision_classification_model': {'values': [True,False]},
        'use_Q_former': {'values': [False]},
        'num_heads_q_former': {'values': [2,4]},
        'num_layers_q_former': {'values': [1,2,3]},
        'num_queries_q_former': {'values': [8,16]},
        'q_former_coef': {'values': [0.5,0.75,1]},

    }
}
import torchvision.transforms as T

def get_train_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_test_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@hydra.main(config_path="configs", config_name="train")
def train_wandb_dinov2(cfg):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(
        project="DINOV2_Q_former_classification_Multimodal_advanced_sweep",
        config=wandb_cfg,
        name=wandb_cfg.get("experiment_name", "run"),
    ) as run:
        config = wandb.config
        logger_std.info(f"Config: {dict(config)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_std.info(f"---------------------- Using device: {device} -----------------")
        seeds = [42, 43, 44, 45]
        for seed in seeds:   
            # --- MODELE DYNAMIQUE ---
            model = MultiModalModel(
                lora_r_text=config.lora_r_text,
                lora_alpha_text=config.lora_alpha_text,
                lora_dropout_text=config.lora_dropout_text,
                lora_r_vision=config.lora_r_vision,
                lora_alpha_vision=config.lora_alpha_vision,
                lora_dropout_vision=config.lora_dropout_vision,
                hidden_dim_1=config.hidden_dim_1,
                hidden_dim_2=config.hidden_dim_2,
                dropout_p=config.dropout_p,
                init_method=config.init_method,
                fusion_dim=config.fusion_dim,
                use_vision_classification_model=config.use_vision_classification_model,
                classification_model_name=config.classification_model_name,
                use_Q_former=config.use_Q_former,
                num_heads_q_former=config.num_heads_q_former,
                num_layers_q_former=config.num_layers_q_former,
                num_queries_q_former=config.num_queries_q_former,
                q_former_coef=config.q_former_coef,
                dinov_vision_coef=config.dinov_vision_coef,
                
            )
            model.to(device)
            total_params, trainable_params = count_parameters(model)
            #save in the wandb
            run.log({"total_params": total_params, "trainable_params": trainable_params})
            logger_std.info(f"\n Nombre total de paramètres: {total_params:,}")
            logger_std.info(f"\n Nombre de paramètres entraînables: {trainable_params:,}")
            if int(trainable_params) > 3500000:
                logger_std.info("Nombre de paramètres entraînables trop élevé, arrêt de l'entraînement.")
                return # <--- ARRET PROPRE ET IMMEDIAT
            print(type(trainable_params))
            logger_std.info(f"\n Pourcentage de paramètres entraînables: {100 * trainable_params / total_params:.2f}%")

            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=1,
                min_lr=1e-6
            )
            if seed == 42:
                cfg.datamodule.custom_val_split = {2023: 1000 , 2022: 400 , 2021: 400 }
            elif seed == 43:
                cfg.datamodule.custom_val_split = {2019: 400 , 2018: 700 , 2017: 700 }
            elif seed == 44:
                cfg.datamodule.custom_val_split = {2023: 400 , 2015: 400 , 2014: 500 , 2013: 500 }
            elif seed == 45:
                cfg.datamodule.custom_val_split = {2023: 700 , 2012: 500 , 2013: 500 , 2014: 500 }
            # Utilise ici la loss de ton choix
            loss_fn = MSLELoss()
            cfg.datamodule.seed = seed
            # cfg.datamodule.custom_val_split = {2023: 400 , 2022: 200 , 2021: 200 , 2020: 200 , 2019: 200 , 2018: 200 , 2017: 200 }
            cfg.datamodule.batch_size = 16
            mean_best_val_loss = 0
            datamodule = hydra.utils.instantiate(cfg.datamodule)
            # Apply transforms after instantiation
            datamodule.train_transform = get_train_transform()
            datamodule.test_transform = get_test_transform()
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            logger_std.info(f"train_set: {len(datamodule.train)}")
            if datamodule.val is not None:
                logger_std.info(f"val_set: {len(datamodule.val)}")

            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 3

            for epoch in tqdm(range(config.epochs), desc="Epochs"):
                model.train()
                epoch_train_loss = 0
                num_samples_train = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
                for i, batch in enumerate(pbar):
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
                run.log({"epoch": epoch, f"train/loss_epoch_{seed}": epoch_train_loss})
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
                run.log({"epoch": epoch, f"val/loss_epoch_{seed}": epoch_val_loss})
                logger_std.info(f"----------------Epoch {epoch} val loss: {epoch_val_loss:.4f}----------------")
                # Scheduler step
                scheduler.step(epoch_val_loss)

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    # logger_std.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {best_model_path}")
                    # torch.save(model.state_dict(), best_model_path)  # Décommente pour sauvegarder
                    run.log({f"best_val_loss_{seed}": best_val_loss})
                else:
                    patience_counter += 1
                    
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            if best_val_loss >3.5:
                print(f"Best validation loss is too high: {best_val_loss}. Stopping training.")
                break

            logger_std.info(
                f"""Epoch {epoch}:
                Training metrics:
                - Train Loss_{seed}: {epoch_train_loss:.4f},
                Validation metrics:
                - Val Loss_{seed}: {epoch_val_loss:.4f},
                Best validation loss_{seed}: {best_val_loss:.4f}"""
            )

if __name__ == "__main__":
    # sweep_id = wandb.sweep(sweep_config, project="Challenge_overfit_DINOV2_Q_former_classification_Multimodal_advanced_sweep")
    # wandb.agent(sweep_id, function=train_wandb, count=100)
    sweep_id = wandb.sweep(sweep_config_dinov2, project="classification_Multimodal_advanced_sweep")
    wandb.agent(sweep_id, function=train_wandb_dinov2, count=100)