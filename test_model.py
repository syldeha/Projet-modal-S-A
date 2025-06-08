import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import wandb
import hydra
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import math
from utils.loss_fn import MSLELoss

# # Votre MSLELoss originale
# class MSLELoss(nn.Module):
#     def __init__(self):
#         super(MSLELoss, self).__init__()

#     def forward(self, y_pred, y_true):
#         # Ensure the predictions and targets are non-negative
#         y_pred = torch.clamp(y_pred, min=0)
#         y_true = torch.clamp(y_true, min=0)

#         # Compute the RMSLE
#         log_pred = torch.log1p(y_pred)
#         log_true = torch.log1p(y_true)
#         loss = torch.mean((log_pred - log_true) ** 2)

#         return loss

class CrossModalAttention(nn.Module):
    """Cross-attention entre vision et texte avec regularisation"""
    def __init__(self, vision_dim, text_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projections pour vision et texte
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention layers
        self.vision_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_vision_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.ln_vision = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ffn_vision = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, vision_features, text_features):
        # Projection vers l'espace commun
        v_proj = self.vision_proj(vision_features)  # [B, 1, hidden_dim]
        t_proj = self.text_proj(text_features)      # [B, seq_len, hidden_dim]
        
        # Cross-attention: vision attend au texte
        v_attended, _ = self.vision_to_text_attn(v_proj, t_proj, t_proj)
        v_attended = self.ln_vision(v_proj + v_attended)
        v_out = v_attended + self.ffn_vision(v_attended)
        
        # Cross-attention: texte attend à la vision
        t_attended, _ = self.text_to_vision_attn(t_proj, v_proj, v_proj)
        t_attended = self.ln_text(t_proj + t_attended)
        t_out = t_attended + self.ffn_text(t_attended)
        
        return v_out.squeeze(1), t_out.mean(dim=1)  # [B, hidden_dim], [B, hidden_dim]

class AdaptiveModalityWeighting(nn.Module):
    """Pondération adaptative des modalités basée sur l'incertitude"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.vision_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.text_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vision_feat, text_feat):
        # Calcul des incertitudes (plus l'incertitude est haute, moins le poids)
        vision_uncertainty = self.vision_uncertainty(vision_feat)
        text_uncertainty = self.text_uncertainty(text_feat)
        
        # Poids inversement proportionnels à l'incertitude
        vision_weight = 1.0 / (1.0 + vision_uncertainty)
        text_weight = 1.0 / (1.0 + text_uncertainty)
        
        # Normalisation
        total_weight = vision_weight + text_weight
        vision_weight = vision_weight / total_weight
        text_weight = text_weight / total_weight
        
        return vision_weight, text_weight

class ImprovedMultiModalModel(nn.Module):
    def __init__(self, 
                 vision_model="efficientnet_b3",
                 hidden_dim=512,
                 num_heads=8,
                 dropout=0.3,
                 lora_r=8,
                 lora_alpha=16):
        super().__init__()
        
        # Vision backbone (plus léger)
        self.vision_backbone = timm.create_model(vision_model, pretrained=True)
        if hasattr(self.vision_backbone, 'fc'):
            self.vision_dim = self.vision_backbone.fc.in_features
            self.vision_backbone.fc = nn.Identity()
        elif hasattr(self.vision_backbone, 'classifier'):
            self.vision_dim = self.vision_backbone.classifier.in_features
            self.vision_backbone.classifier = nn.Identity()
        elif hasattr(self.vision_backbone, "head"):
            self.vision_dim = self.vision_backbone.head.in_features
            self.vision_backbone.head = nn.Identity()
            
        # Gel des paramètres vision
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # Text backbone avec LoRA plus léger
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Configuration LoRA plus conservative
        lora_config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=dropout,
            bias="none",
            target_modules=["q_lin", "v_lin"]
        )
        self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        
        # Gel des paramètres non-LoRA
        for name, param in self.text_encoder.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
                
        # Cross-modal attention
        self.cross_modal_attn = CrossModalAttention(
            self.vision_dim, self.text_dim, hidden_dim, num_heads, dropout
        )
        
        # Pondération adaptative
        self.adaptive_weighting = AdaptiveModalityWeighting(hidden_dim)
        
        # Prédiction finale avec activation adaptée pour les vues
        self.predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Assure des sorties strictement positives
        )
        
        # Initialisation des poids
        self._init_weights()
        
    def _init_weights(self):
        """Initialisation conservative des poids"""
        for module in [self.cross_modal_attn, self.adaptive_weighting, self.predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
    
    def forward(self, batch):
        """Forward compatible avec votre format de batch"""
        # Extraction des données du batch
        images = batch["image"]
        
        # Pour le texte, adapter selon votre format exact
        # Supposons que vous avez des clés 'input_ids' et 'attention_mask'
        text_inputs = self.tokenizer(
                    batch["prompt_resume"], 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512
                )
        input_ids = text_inputs["input_ids"].to(images.device)
        attention_mask = text_inputs["attention_mask"].to(images.device)

        # Extraction des features vision
        with torch.no_grad():
            vision_features = self.vision_backbone(images)
        vision_features = vision_features.unsqueeze(1)
        
        # Extraction des features texte
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state
        
        # Cross-modal attention
        vision_attended, text_attended = self.cross_modal_attn(
            vision_features, text_features
        )
        
        # Pondération adaptative
        vision_weight, text_weight = self.adaptive_weighting(
            vision_attended, text_attended
        )
        
        # Fusion pondérée
        fused_features = (vision_weight * vision_attended + 
                         text_weight * text_attended)
        
        # Prédiction
        output = self.predictor(fused_features)
        
        return output.squeeze(-1)

# Fonction utilitaire pour compter les paramètres
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Configuration de sweep pour le nouveau modèle
sweep_config_improved = {
    'method': 'random',
    'metric': {
        'name': 'val/loss_epoch',
        'goal': 'minimize'
    },
    'parameters': {
        # Architecture du modèle
        'vision_model': {'values': ['mobilenetv4_hybrid_medium' , 'resnet50', 'efficientnet_b5', 'efficientnet_b3']},
        'hidden_dim': {'values': [256,512,768]},
        'num_heads': {'values': [4,8]},
        'dropout': {'values': [0.2,0.3]},
        
        # LoRA parameters (plus conservatives)
        'lora_r': {'values': [8,16,32]},
        'lora_alpha': {'values': [8,16,32]},
        
        # Training parameters
        'learning_rate': {'values': [1e-3,1e-4 , 2e-3]},
        'epochs': {'values': [10]},
        'experiment_name': {'values': ["improved_cross_modal"]},
    }
}

@hydra.main(config_path="configs", config_name="train")
def train_improved_model(cfg):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(
        project="version2_Challenge_Improved_CrossModal_AntiOverfit",
        config=wandb_cfg,
        name=wandb_cfg.get("experiment_name", "improved_run"),
    ) as run:
        config = wandb.config
        print(f"Config: {dict(config)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        # va{seeds =[2023]
        test_seeds =[42,43,50 , 100, 60]
        for seed in test_seeds:
            print(f"validation avec la seed={seed}")
            # run.log({"seed": seed})
            cfg.datamodule.seed = seed
            cfg.experiment_name = f"{cfg.experiment_name}_seed{seed}"
            cfg.datamodule.custom_val_split = {2023: 400 , 2022: 200 , 2021: 200 , 2020: 200 , 2019: 200 , 2018: 200 , 2017: 200 }
            mean_best_val_loss = 0



            # Data loaders (utilise votre datamodule existant)
            datamodule = hydra.utils.instantiate(cfg.datamodule)
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            print(f"train_set: {len(datamodule.train)}")
            if datamodule.val is not None:
                print(f"val_set: {len(datamodule.val)}")

            # Modèle amélioré
            model = ImprovedMultiModalModel(
                vision_model=config.vision_model,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
            )
            model.to(device)
            
            # Comptage des paramètres
            total_params, trainable_params = count_parameters(model)
            run.log({"total_params": total_params, "trainable_params": trainable_params})
            # run.log({"seed": seed})
            print(f"Nombre total de paramètres: {total_params:,}")
            print(f"Nombre de paramètres entraînables: {trainable_params:,}")
            print(f"Pourcentage de paramètres entraînables: {100 * trainable_params / total_params:.2f}%")

            # Optimiseur et scheduler
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], 
                lr=config.learning_rate,
                weight_decay=1e-4  # Regularisation L2 intégrée
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,  # Réduction plus douce
                patience=2,   # Plus de patience
                min_lr=1e-6
            )
            
            # Loss function
            loss_fn = MSLELoss()
            

            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 3

            for epoch in tqdm(range(config.epochs), desc="Epochs"):
                # Training
                model.train()
                epoch_train_loss = 0
                num_samples_train = 0
                
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
                for i, batch in enumerate(pbar):
                    # Déplacer les données sur GPU
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)
                    # print(batch.keys())
                    
                    # Forward pass
                    preds = model(batch).squeeze()
                    target = batch["target"].squeeze()
                    loss = loss_fn(preds, target)
                    
                    # Backward pass
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    # Gradient clipping pour la stabilité
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Accumulation des métriques
                    epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
                    num_samples_train += len(batch["image"])
                    pbar.set_postfix({"train/loss_step": loss.detach().cpu().item()})
                
                epoch_train_loss /= num_samples_train
                run.log({"epoch": epoch, f"train/loss_epoch_{seed}": epoch_train_loss})
                print(f"Epoch {epoch} train loss_{seed}: {epoch_train_loss:.4f}")

                # Validation
                model.eval()
                epoch_val_loss = 0
                num_samples_val = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Déplacer les données sur GPU
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)
                        
                        preds = model(batch).squeeze()
                        target = batch["target"].squeeze()
                        loss = loss_fn(preds, target)
                        
                        epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                        num_samples_val += len(batch["image"])
                
                epoch_val_loss /= num_samples_val
                run.log({"epoch": epoch, f"val/loss_epoch_{seed}": epoch_val_loss})
                print(f"Epoch {epoch} val loss_{seed}: {epoch_val_loss:.4f}")
                
                # Scheduler step
                scheduler.step(epoch_val_loss)
                
                # Early stopping et sauvegarde du meilleur modèle
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    print(f"New best validation loss{seed}: {best_val_loss:.4f}")
                    run.log({f"best_val_loss{seed}": best_val_loss})
                    # torch.save(model.state_dict(), f"{run.name}_best_model_{seed}.pt")
                else:
                    patience_counter += 1
                    
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            if best_val_loss >6:
                break
            mean_best_val_loss += best_val_loss
            print(f"Training completed. Best validation loss_{seed}: {best_val_loss:.4f}")
            mean_best_val_loss /= len(test_seeds)
            run.log({"mean_best_val_loss": mean_best_val_loss})
            print(f"Mean best validation loss: {mean_best_val_loss:.4f}")


if __name__ == "__main__":
    # Pour lancer un sweep
    sweep_id = wandb.sweep(
        sweep_config_improved, 
        project="version_4__Challenge_Improved_CrossModal_AntiOverfit"
    )
    wandb.agent(sweep_id, function=train_improved_model, count=50)
    
    # Ou pour un run simple, décommentez:
    # train_improved_model()