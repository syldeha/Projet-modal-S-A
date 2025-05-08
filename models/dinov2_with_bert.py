import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

class DinoV2WithBert(nn.Module):
    def __init__(self, frozen=False , pretrained_model="distilbert-base-uncased", tokenizer_model_path="distilbert-base-uncased",vis_coef=1.,txt_coef=1.5):
        super().__init__()
        logger_std.info(f"Initialisation du model DinoV2WithBert")
        # Vision backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.vision_dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Text backbone (Transformer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self.text_dim = self.text_encoder.config.hidden_size
        if frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Fusion coefficients
        self.vis_coef = vis_coef
        self.txt_coef = txt_coef

        # Harmonisation des dims (projection si besoin)
        fusion_dim = max(self.vision_dim, self.text_dim)
        if self.vision_dim != fusion_dim:
            self.vis_proj = nn.Linear(self.vision_dim, fusion_dim)
        else:
            self.vis_proj = nn.Identity()
        if self.text_dim != fusion_dim:
            self.txt_proj = nn.Linear(self.text_dim, fusion_dim)
        else:
            self.txt_proj = nn.Identity()

        # Tête de régression finale (ou classification selon tes besoins)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W), x["title"]: list[str]

        # 1. Encode image
        v_feat = self.backbone(x["image"])
        v_feat = self.vis_proj(v_feat)    # [batch, fusion_dim]

        # 2. Encode text
        # Assurez-vous que le tokenizer est sur le même device que l'image
        device = v_feat.device
        encoded = self.tokenizer(x["prompt_resume"], padding=True, truncation=True, return_tensors='pt')
        # Déplacer les tenseurs vers le bon device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.text_encoder(**encoded)
        t_feat = out.last_hidden_state[:, 0]
        t_feat = self.txt_proj(t_feat)       # [batch, fusion_dim]

        # 3. Fusion (addition pondérée)
        z = self.vis_coef * v_feat + self.txt_coef * t_feat

        # 4. Régression
        out = self.regression_head(z)
        return out

    def load_checkpoint(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
        """
        Load a checkpoint for the entire model or just the backbone.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            load_full: Whether to load the full model or just the backbone
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Determine if checkpoint is a raw state_dict or a dict with 'model_state_dict'
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint
            
            # Try to load the full model state dict first
            if load_full:
                missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)
                print(f"Loaded full model from {checkpoint_path}")
            else:
                # If full load fails, try to load just the backbone
                print(f"Loading just the backbone from {checkpoint_path}")
                backbone_state_dict = {}
                for k, v in model_state_dict.items():
                    # Handle both 'backbone.X' and just 'X' formats
                    if k.startswith('backbone.'):
                        backbone_state_dict[k] = v
                    else:
                        # Try to load as backbone parameter
                        backbone_key = f"backbone.{k}"
                        backbone_state_dict[backbone_key] = v
                
                missing_keys, unexpected_keys = self.load_state_dict(backbone_state_dict, strict=False)
                print(f"Loaded backbone only from {checkpoint_path}")
            
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False
        
    def load_model(self, pretrained_model, frozen=True):
        pretrained_model_name = f"Syldehayem/{pretrained_model}"
        logger_std.info(f"Chargement du model {pretrained_model} sur le model DinoV2WithBert ")
        # Déterminer le device actuel du modèle
        device = next(self.parameters()).device
        # Charger le modèle BERT
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        # Déplacer le modèle sur le même device que le modèle principal
        self.text_encoder = self.text_encoder.to(device)
        self.text_dim = self.text_encoder.config.hidden_size
        if frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
