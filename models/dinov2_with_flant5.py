import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

class DinoV2WithFlant5(nn.Module):
    def __init__(self, frozen=False , pretrained_model="google/flan-t5-small", tokenizer_model_path="google/flan-t5-small",vis_coef=1.0,txt_coef=1.0, fusion_dropout=0.2):
        super().__init__()
        logger_std.info(f"Initialisation du model DinoV2WithFlant5")
        # Vision backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.vision_dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Text backbone (Transformer)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_path)
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model)
        self.text_dim = self.text_encoder.config.hidden_size
        if frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False 

        # Fusion coefficients variable
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
        #**Fusion complète par concaténation, puis projection**
        self.fusion_proj = nn.Linear(2*fusion_dim, fusion_dim)
        self.proj_2 = nn.Linear(2*fusion_dim, 256)

        # Normalisation
        self.norm = nn.BatchNorm1d(fusion_dim)
        self.last_layer = nn.Linear(256, 1)


        # Tête de régression finale (ou classification selon tes besoins)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W), x["prompt_resume"]: list[str]

        # 1. Encode image
        v_feat = self.backbone(x["image"])
        v_feat = self.vis_proj(v_feat)    # [batch, fusion_dim]

        # 2. Encode text
        # Assurez-vous que le tokenizer est sur le même device que l'image
        device = v_feat.device
        encoded = self.tokenizer(x["prompt_resume"], padding=True, truncation=True, return_tensors='pt')
        #Utilisation de tous les tokens de sorties et non uniquement le token [CLS]
        # Déplacer les tenseurs vers le bon device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.text_encoder(**encoded)
        # 1. Multiply embeddings by the attention mask (to "zero out" padded tokens)
        input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        # 2. Sum along the token dimension (dim=1)
        sum_embeddings = torch.sum(out.last_hidden_state * input_mask_expanded, 1)
        # 3. Count number of valid tokens (sum of attention mask along dim=1)
        sum_mask = input_mask_expanded.sum(1)
        # 4. Divide summed embeddings by the (nonzero) sum of mask: this gives the **mean** (average)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9)
        #Projection
        t_feat = self.txt_proj(t_feat)       # [batch, fusion_dim]

        # -FUSION par concat + projection
        z = torch.cat([self.vis_coef * v_feat, self.txt_coef * t_feat], dim=1)  # (Batch, 2*fusion_dim)
        # z=self.vis_coef * v_feat+self.txt_coef * t_feat
        z_1 = self.fusion_proj(z)   # (Batch, fusion_dim)
        # z_2 = self.proj_2(z)       # (Batch, 256)
        #normalisation
        # z_1 = self.norm(z_1)       # (Batch, fusion_dim)

        # 4. Régression
        z_1 = self.regression_head(z_1)    # (Batch, 256)
        #couche residuelle entre z_1 et z_2
        # out = z_1+z_2   # (Batch, 256)
        out = self.last_layer(z_1)        # (Batch, 1)
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
        logger_std.info(f"Chargement du model {pretrained_model} sur le model DinoV2WithFlant5 ")
        # Déterminer le device actuel du modèle
        device = next(self.parameters()).device
        # Charger le modèle T5Encoder au lieu de AutoModel
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name)
        # Déplacer le modèle sur le même device que le modèle principal
        self.text_encoder = self.text_encoder.to(device)
        self.text_dim = self.text_encoder.config.hidden_size
        if frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
