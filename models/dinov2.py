import torch
import torch.nn as nn
import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

os.environ['HYDRA_FULL_ERROR'] = '1'

class SimpleQFormerBlock(nn.Module):
    """One block of the Q-Former, with cross-attention + feedforward."""
    def __init__(self, query_dim, img_feature_dim, num_heads, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim, kdim=img_feature_dim, vdim=img_feature_dim, num_heads=num_heads, batch_first=True)
        hidden_dim = 4*query_dim
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
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
            SimpleQFormerBlock(query_dim, img_feature_dim, num_heads, dropout=0.3)
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


class DinoV2Finetune(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()        
        self.dim = self.backbone.norm.normalized_shape[0]
        self.query_dim = 128
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.qformer = SimpleQFormer(
            query_dim=self.query_dim,
            img_feature_dim=self.dim,
            num_heads=8,
            num_layers=2,
            num_queries=8
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.query_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image

        feats = self.backbone.forward_features(imgs)  # [batch, n_tokens, dim]
        cls_token = feats["x_norm_clstoken"].unsqueeze(1) #(batch, 1, dim)
        patch_tokens = feats["x_norm_patchtokens"] #(batch, n_patches, dim)
        feats = torch.cat( [cls_token, patch_tokens],dim=1)  # shape: [batch, 1+N_patches, dim]
        #utilisation des features de l'image pour le Q-Former
        out = self.qformer(feats)  # [batch, n_queries, query_dim]

        #moyennage des features du Q-Former
        pooled = out.mean(dim=1)  # ou out[:,0,:] pour le premier token

        #regression
        pred = self.regression_head(pooled)  # [batch, 1]
        return pred
    def load_checkpoint(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint

            if load_full:
                missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)
                print(f"Loaded full model from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False