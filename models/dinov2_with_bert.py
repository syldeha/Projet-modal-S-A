import os
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

os.environ['HYDRA_FULL_ERROR'] = '1'
import torch
import torch.nn as nn
import timm
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import logging

class SimpleQFormerBlock(nn.Module):
    """One block of the Q-Former, with cross-attention + feedforward."""
    def __init__(self, query_dim, img_feature_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim, kdim=img_feature_dim, vdim=img_feature_dim, num_heads=num_heads, batch_first=True)
        hidden_dim = 2*query_dim
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim)
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
    
    def forward(self, queries, img_features):
        # queries: [batch, N_queries, query_dim]
        # img_features: [batch, N_patches, img_feature_dim]
        attn_output, _ = self.cross_attn(queries, img_features, img_features)
        x = self.norm1(queries + attn_output)
        y = self.ffn(x)
        y = self.norm2(x + y)
        return y

class SimpleQFormer(nn.Module):
    def __init__(self, query_dim, img_feature_dim, num_heads, num_layers, num_queries):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, query_dim))
        self.blocks = nn.ModuleList([
            SimpleQFormerBlock(query_dim, img_feature_dim, num_heads)
            for _ in range(num_layers)
        ])
    def forward(self, img_features):
        """
        img_features : [batch, n_patches+1, img_feature_dim]   # output from ViT encoder (ex: DINOv2)
        return : [batch, num_queries, query_dim]
        """
        #recuperation des features de l'image
        batch_size = img_features.shape[0]
        queries = self.query_tokens.expand(batch_size, -1, -1).contiguous()
        x = queries
        for block in self.blocks:
            x = block(x, img_features)
        return x # [batch, num_queries, query_dim]


class DinoV2WithBertLora_Backbone(nn.Module):
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
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
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
        fusion_dim = max(self.text_dim , self.vision_dim)
        self.fusion_dim = fusion_dim
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

    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W)
        # x["prompt_resume"]: list[str] size = batch_size

        # 1. Encode image
        v_feat = self.backbone.forward_features(x["image"]) # [batch, n_patches+1, self.vision_dim]
        #moyennage des features de l'image
        v_feat_pooled = v_feat.mean(dim=1) #[batch, self.vision_dim]

        # 2. Encode text
        device = v_feat_pooled.device
        encoded = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.text_encoder(**encoded)
        input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        t_feat = self.txt_proj(t_feat) # [batch, self.fusion_dim]

        # Fusion par concaténation + projection
        z = torch.cat([self.vis_coef * v_feat_pooled, self.txt_coef * t_feat], dim=1) # [batch, 2*self.fusion_dim]
        z = self.fusion_proj(z) # [batch, self.fusion_dim]
        z = self.norm(z)
        return z , v_feat , t_feat

    def load_checkpoint_only_lora_backbone(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
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


class DinoV2WithBertLoRA(nn.Module):
    def __init__(
        self, 
        frozen=False,
        pretrained_model="distilbert-base-uncased", 
        tokenizer_model_path="distilbert-base-uncased", 
        vis_coef=1.0, 
        txt_coef=1.0, 
        q_former=False,
        pretrained_model_checkpoint=None,
        frozen_backbone=True,
    ):
        super().__init__()
        logger_std.info(f"Initialisation du model DinoV2WithBertLoRA")


        # ---- Vision backbone ----
        self.backbone = DinoV2WithBertLora_Backbone(
            frozen=frozen,
            pretrained_model=pretrained_model,
            tokenizer_model_path=tokenizer_model_path,
            vis_coef=vis_coef,
            txt_coef=txt_coef,
        )
        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fusion_dim = self.backbone.fusion_dim
        self.vision_dim = self.backbone.vision_dim
        self.q_former = q_former
        if self.q_former:
            logger_std.info("Applying Q-Former...")
            self.qformer = SimpleQFormer(
                query_dim=self.fusion_dim,
                img_feature_dim=self.vision_dim,
                num_heads=16,
                num_layers=5,
                num_queries=16
            )

        # Tête de régression finale (ou classification selon tes besoins)

        # self.regression_head = nn.Sequential(
        #     nn.Linear(self.fusion_dim , 512),  # 
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 1),
        #     nn.ReLU(),
        # )
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        if pretrained_model_checkpoint:
            logger_std.info(f"Chargement du model {pretrained_model_checkpoint} sur le model DinoV2WithBert ")
            self.load_model_backbone(pretrained_model_checkpoint)


    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W)
        # x["prompt_resume"]: list[str] size = batch_size

        # 1. Encode image
        z , v_feat , t_feat = self.backbone(x) # [batch, self.fusion_dim], [batch, n_patches+1, self.vision_dim], [batch, self.fusion_dim]
        if self.q_former:
            #utilisation des features de l'image pour le Q-Former
            v_feat = self.qformer(v_feat) # [batch, n_queries, self.fusion_dim]
            #moyennage des features du Q-Former
            z = v_feat.mean(dim=1) # [batch, self.fusion_dim]
            z = self.regression_head(z) # [batch, 1]
            return z
        else :
            out = self.regression_head(z) # [batch, 1]
            return out

    def load_model_backbone(self, pretrained_model_checkpoint):
        self.backbone.load_checkpoint_only_lora_backbone(checkpoint_path=pretrained_model_checkpoint)

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




# DISTILLBERT SANS  LORA POUR RESNET 50
class Resnet50WithBertLora_Backbone(nn.Module):
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
        lora_dropout=0.05,
    ):
        super().__init__()
        logger_std.info(f"Initialisation du model DinoV2WithBertLoRA")


        # ---- Vision backbone ----
        self.backbone = timm.create_model('resnet50', pretrained=True)
        # on ne garde que les features de l'image
        self.vision_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        

        # on freeze les paramètres de la backbone
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
        fusion_dim = min(self.text_dim , self.vision_dim)
        self.fusion_dim = fusion_dim
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

    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W)
        # x["prompt_resume"]: list[str] size = batch_size

        # 1. Encode image
        v_feat = self.backbone(x["image"]) # [batch, self.vision_dim]
        #moyennage des features de l'image
        v_feat_pooled = self.vis_proj(v_feat)


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
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        t_feat = self.txt_proj(t_feat) # [batch, self.fusion_dim]

        # Fusion par concaténation + projection
        z = torch.cat([self.vis_coef * v_feat, self.txt_coef * t_feat], dim=1) # [batch, 2*self.fusion_dim]
        z = self.fusion_proj(z) # [batch, self.fusion_dim]
        z = self.norm(z)
        return z , v_feat , t_feat

    def load_checkpoint_only_lora_backbone(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
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


class Resnet50WithBertLoRA(nn.Module):
    def __init__(
        self, 
        frozen=False,
        pretrained_model="distilbert-base-uncased", 
        tokenizer_model_path="distilbert-base-uncased", 
        vis_coef=1.0, 
        txt_coef=1.0, 
        q_former=False,
        pretrained_model_checkpoint=None,
        frozen_backbone=True,
    ):
        super().__init__()
        logger_std.info(f"Initialisation du model DinoV2WithBertLoRA")


        # ---- Vision backbone ----
        self.backbone = Resnet50WithBertLora_Backbone(
            frozen=frozen,
            pretrained_model=pretrained_model,
            tokenizer_model_path=tokenizer_model_path,
            vis_coef=vis_coef,
            txt_coef=txt_coef,
        )
        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fusion_dim = self.backbone.fusion_dim
        self.vision_dim = self.backbone.vision_dim
        self.q_former = q_former
        if self.q_former:
            logger_std.info("Applying Q-Former...")
            self.qformer = SimpleQFormer(
                query_dim=self.fusion_dim,
                img_feature_dim=self.vision_dim,
                num_heads=16,
                num_layers=5,
                num_queries=16
            )

        # Tête de régression finale (ou classification selon tes besoins)
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        if pretrained_model_checkpoint:
            logger_std.info(f"Chargement du model {pretrained_model_checkpoint} sur le model Resnet50WithBert ")
            self.load_model_backbone(pretrained_model_checkpoint)


    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W)
        # x["prompt_resume"]: list[str] size = batch_size

        # 1. Encode image
        z , v_feat , t_feat = self.backbone(x) # [batch, self.fusion_dim], [batch, self.vision_dim], [batch, self.fusion_dim]
        if self.q_former:
            #utilisation des features de l'image pour le Q-Former
            v_feat = v_feat.unsqueeze(1) # [batch, 1, self.vision_dim]
            v_feat = self.qformer(v_feat) # [batch, self.fusion_dim]
            #moyennage des features du Q-Former
            z = v_feat.mean(dim=1) # [batch, self.fusion_dim]
            z = self.regression_head(z) # [batch, 1]
            return z
        else :
            out = self.regression_head(z) # [batch, 1]
            return out

    def load_model_backbone(self, pretrained_model_checkpoint):
        self.backbone.load_checkpoint_only_lora_backbone(checkpoint_path=pretrained_model_checkpoint)

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