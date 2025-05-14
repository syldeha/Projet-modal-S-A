import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

os.environ['HYDRA_FULL_ERROR'] = '1'


# def remap_text_encoder_keys(model_state_dict):
#     """
#     Remap checkpoint keys for text_encoder when applying LoRA/PEFT:
#     - Ajoute .base_layer pour tous les Linear 'q_lin', 'k_lin', 'v_lin' (attention)
#     - Passe les chemins 'text_encoder.embeddings' → 'text_encoder.base_model.model.embeddings'
#     - Passe les chemins 'text_encoder.transformer' → 'text_encoder.base_model.model.transformer'
#     - NE TOUCHE PAS aux poids LoRA qui seront absents (normal), ils seront init par défaut
#     Compatible avec checkpoints vanilla ou déjà PEFT.

#     Args:
#         model_state_dict (dict): les poids issus d'un vanilla BERT ou d'un PEFT BERT

#     Returns:
#         dict: nouveau state_dict adapté pour un backbone LoRA/PEFT
#     """
#     # Liste des Linear qui sont PEFTés (à compléter si d'autres modules dans ton archi)
#     peftified_linears = ['q_lin', 'k_lin', 'v_lin']  # tu peux mettre 'out_lin' si c'est PEFT chez toi

#     new_state_dict = {}
#     for k, v in model_state_dict.items():
#         nk = k  # new key, potentiellement modifiée

#         # Mapping embeddings/transformer hiérarchie
#         if nk.startswith("text_encoder.embeddings"):
#             nk = nk.replace("text_encoder.embeddings", "text_encoder.base_model.model.embeddings", 1)
#         if nk.startswith("text_encoder.transformer"):
#             nk = nk.replace("text_encoder.transformer", "text_encoder.base_model.model.transformer", 1)

#         # Mapping des Linear vanilla -> base_layer pour PEFT
#         for lin in peftified_linears:
#             # Pour chaque couple attention.blabla.weight/bias à remplacer par .base_layer.weight/.base_layer.bias
#             if f".attention.{lin}.weight" in nk:
#                 nk = nk.replace(f".attention.{lin}.weight", f".attention.{lin}.base_layer.weight")
#             if f".attention.{lin}.bias" in nk:
#                 nk = nk.replace(f".attention.{lin}.bias", f".attention.{lin}.base_layer.bias")

#         # Ajoutez ici d'autres mappings spécialisés si d'autres modules Linear/PEFT
#         new_state_dict[nk] = v

#     print("[INFO] Remapping text_encoder keys for checkpoint/PEFT (embeddings, transformer, .base_layer… etc)")
#     return new_state_dict


class DinoV2WithBert(nn.Module):
    def __init__(self, frozen=False , pretrained_model="distilbert-base-uncased", tokenizer_model_path="distilbert-base-uncased",vis_coef=1.0,txt_coef=1.0, fusion_dropout=0.2):
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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
            self.text_encoder = AutoModel.from_pretrained(pretrained_model)
            self.text_dim = self.text_encoder.config.hidden_size
            if frozen:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        except Exception as e:
            logger_std.error(f"Error loading DistilBert model: {str(e)}")
            # Fallback to a simpler model if loading fails
            from transformers import DistilBertModel, DistilBertTokenizer
            logger_std.info("Attempting to load DistilBert model directly...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_model_path)
            self.text_encoder = DistilBertModel.from_pretrained(pretrained_model)
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
        #**Fusion complète par concaténation, puis projection**
        self.fusion_proj = nn.Linear(2*fusion_dim, fusion_dim)

        # Normalisation
        self.norm = nn.LayerNorm(fusion_dim)


        # Tête de régression finale (ou classification selon tes besoins)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.GELU(),
            nn.Linear(256, 1),
            nn.ReLU(),
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
        # Déplacer les tenseurs vers le bon device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.text_encoder(**encoded)
        # t_feat = out.last_hidden_state[:, 0]
        # 1. Multiply embeddings by the attention mask (to "zero out" padded tokens)
        input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        # 2. Sum along the token dimension (dim=1)
        sum_embeddings = torch.sum(out.last_hidden_state * input_mask_expanded, 1)
        # 3. Count number of valid tokens (sum of attention mask along dim=1)
        sum_mask = input_mask_expanded.sum(1)
        # 4. Divide summed embeddings by the (nonzero) sum of mask: this gives the **mean** (average)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9)
        t_feat = self.txt_proj(t_feat)       # [batch, fusion_dim]

        # -FUSION par concat + projection
        z = torch.cat([self.vis_coef * v_feat, self.txt_coef * t_feat], dim=1)  # (Batch, 2*fusion_dim)
        z = self.fusion_proj(z)   # (Batch, fusion_dim)
        #normalisation
        z = self.norm(z)

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

#MISE EN PLACE DE LORA

import torch
import torch.nn as nn
import timm
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import logging
import os

logger_std = logging.getLogger(__name__)

class DinoV2WithBertLoRA(nn.Module):
    def __init__(
        self, 
        frozen=False,
        pretrained_model="distilbert-base-uncased", 
        tokenizer_model_path="distilbert-base-uncased", 
        vis_coef=1.0, 
        txt_coef=1.0, 
        fusion_dropout=0.2,
        lora_vis=True,         # LoRA sur vision backbone
        lora_txt=True,         # LoRA sur texte backbone
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05
    ):
        super().__init__()
        logger_std.info(f"Initialisation du model DinoV2WithBertLoRA")

        # ---- Vision backbone ----
        self.backbone = timm.create_model('vit_base_patch14_reg4_dinov2', pretrained=True)
        self.backbone.head = nn.Identity()
        self.vision_dim = self.backbone.norm.normalized_shape[0]
        
        if lora_vis:
            logger_std.info("Applying LoRA to vision backbone...")
            vis_lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=["qkv", "proj"]  # timm naming!
            )
            self.backbone = get_peft_model(self.backbone, vis_lora_config)
            self.backbone.print_trainable_parameters()
        elif frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---- Text backbone ----
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self.text_dim = self.text_encoder.config.hidden_size
        
        if lora_txt:
            logger_std.info("Applying LoRA to text backbone...")
            # For BERT-like: target_modules are ['query', 'value']
            txt_lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=["q_lin", "v_lin"]
            )
            self.text_encoder = get_peft_model(self.text_encoder, txt_lora_config)
            self.text_encoder.print_trainable_parameters()
        elif frozen:
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
        # Fusion complète par concaténation, puis projection
        self.fusion_proj = nn.Linear(2*fusion_dim, fusion_dim)

        # Normalisation
        self.norm = nn.LayerNorm(fusion_dim)

        # Tête de régression finale (ou classification selon tes besoins)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W)
        # x["prompt_resume"]: list[str] size = batch_size

        # 1. Encode image
        v_feat = self.backbone(x["image"])
        v_feat = self.vis_proj(v_feat)

        # 2. Encode text
        device = v_feat.device
        encoded = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.text_encoder(**encoded)
        input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9)
        t_feat = self.txt_proj(t_feat)

        # Fusion par concaténation + projection
        z = torch.cat([self.vis_coef * v_feat, self.txt_coef * t_feat], dim=1)
        z = self.fusion_proj(z)
        z = self.norm(z)

        # Head
        out = self.regression_head(z)
        return out

    def load_model(self, pretrained_model, frozen=True):
        pretrained_model_name = f"{pretrained_model}"
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

    def load_checkpoint(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint

            if load_full:
                # --- AJOUT : remapping si besoin
                # model_state_dict = remap_text_encoder_keys(model_state_dict)
                # # ---
                missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)
                print(f"Loaded full model from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False