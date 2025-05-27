import torch
import torch.nn as nn
import timm
import hydra
from omegaconf import OmegaConf
import wandb
# from utils.utils import count_parameters
import logging
import os
import timm
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import logging
import wandb
os.environ['HYDRA_FULL_ERROR'] = '1'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params




class FlexibleClassifier(nn.Module):
    def __init__(
        self,
        backbone_name='resnet50',
        pretrained=True,
        num_classes=5,
        hidden_dim_1=512,
        hidden_dim_2=256,
        dropout_p=0.2,
        activation='relu',
        freeze_backbone=False,
        init_method='kaiming'
    ):
        super().__init__()
        # ---- Backbone ----
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        # attraper la taille des embeddings remontés par le backbone
        if hasattr(self.backbone, 'fc'):
            in_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            in_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            in_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError("Unknown backbone type, please adapt code.")

        # Freeze backbone, si demandé
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- Head MLP (flexible) ----
        # Prise en charge des activations dynamiques
        act_dict = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'leakyrelu': nn.LeakyReLU
        }
        activation_cls = act_dict.get(activation.lower(), nn.ReLU)

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            activation_cls(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            activation_cls(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim_2, num_classes),
        )
        self._init_weights(init_method)

        
    def _init_weights(self, method):
        for m in self.head:
            if isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                elif method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.backbone(x)  # [B, D]
        logits = self.head(f) # [B, num_classes]
        return logits

#--------TRAIN MODEL
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import hydra

@hydra.main(config_path="configs", config_name="classification_config")
def train_classification_model(cfg):
    # Convert config to dictionary for wandb
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    with wandb.init(
        project="challenge_CSC_43M04_EP_classification_model",
        config=wandb_cfg,
        name=wandb_cfg.get("experiment_name", "run"),
    ) as run:
        config = wandb.config
        logger_std.info(f"Config: {dict(config)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_std.info(f"---------------------- Using device: {device} -----------------")
        # DataLoader
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FlexibleClassifier(
            backbone_name=config.backbone_name,
            num_classes=config.num_classes,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            dropout_p=config.dropout_p,
            freeze_backbone=config.freeze_backbone,
            init_method=config.init_method
        )
        model = model.to(device)
        total_params, trainable_params = count_parameters(model)
        logger_std.info(f"\n Nombre total de paramètres: {total_params:,}")
        logger_std.info(f"\n Nombre de paramètres entraînables: {trainable_params:,}")
        logger_std.info(f"\n Pourcentage de paramètres entraînables: {100 * trainable_params / total_params:.2f}%")

        # Loss (avec class_weights si fourni)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        best_val_acc = 0
        best_state = None

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            correct = 0         # <-- cumule le nombre de bonnes prédictions
            total = 0
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
            for batch in pbar:
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(images)           # [B, num_classes]
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(train_loader)
            train_acc = correct / total if total > 0 else 0.

            if logger_std:
                logger_std.info(f"Epoch {epoch+1} - Train loss: {avg_loss:.4f}")
                logger_std.info(f"Epoch {epoch+1} - Train accuracy: {train_acc:.4f}")
            wandb.log({"train_acc": train_acc})

            # --- Validation (pareil) ---
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['labels'].to(device)
                    logits = model(images)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total if total > 0 else 0.
            if logger_std:
                logger_std.info(f"Epoch {epoch+1} - Valid accuracy: {val_acc:.4f}")
            wandb.log({"val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # best_state = model.state_dict().copy()
            wandb.log({"best_val_acc": best_val_acc})

    return best_val_acc, save_path

sweep_config = {
    'method': 'random',  
    'metric': {
        'name': 'val_acc',  
        'goal': 'maximize'
    },
    'parameters': {
        # ---- Backbone -------
        'backbone_name':        {'values': ["mobilenetv3_large_100" , "efficientnet_b0", "efficientnet_b1"]},
        'pretrained':           {'values': [True]},    # ou [True, False]
        'freeze_backbone':      {'values': [True]},
        # ----- Tête -------
        'hidden_dim_1':         {'values': [128]},
        'hidden_dim_2':         {'values': [128]},
        'dropout_p':            {'values': [0.1]},
        'activation':           {'values': ["gelu"]},
        'init_method':          {'values': ["xavier"]},
        # ---- Optimiz/Train -----
        'learning_rate':        {'values': [1e-4, 1e-3, 1e-5]},
        'epochs':               {'values': [10]},  # à adapter !
        'experiment_name':      {'values': ["flexible_classification"]},
        # ----- Data -----
        'num_classes':          {'values': [5]},  # à adapter à ton problème
    }
}

# -------- UTILISATION --------
if __name__ == "__main__":
    import wandb
    sweep_id=wandb.sweep(sweep_config, project="challenge_CSC_43M04_EP_classification_model")
    wandb.agent(sweep_id, function=train_classification_model , count=15)

