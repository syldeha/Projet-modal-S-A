import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer

class DinoV2WithEmbed(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        # Vision backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.vision_dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Text backbone (Transformer)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.text_dim = self.text_encoder.config.hidden_size


        # Fusion
        self.vis_coef = 0.75
        self.txt_coef = 0.25

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

        # Tête finale (identique à la tienne)
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
    # def mean_pooling(self, model_output, attention_mask):
    #     token_embeddings = model_output[0]
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W), x["title"]: list[str]

        # 1. Encode image
        v_feat = self.backbone(x["image"])
        v_feat = self.vis_proj(v_feat)    # [batch, fusion_dim]

        # 2. Encode text
        encoded = self.tokenizer(x["description"], padding=True, truncation=True, return_tensors='pt').to(v_feat.device)
        with torch.no_grad():
            out = self.text_encoder(**encoded)
        # t_feat = self.mean_pooling(out, encoded["attention_mask"])
        t_feat = self.txt_proj(out)       # [batch, fusion_dim]

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