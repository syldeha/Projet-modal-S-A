import torch
import torch.nn as nn
import os


class DinoV2Finetune_advanced(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone.norm.normalized_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )
        
    def load_checkpoint(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
        """
        Load a checkpoint for the entire model or just the backbone.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
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
                print(f"Could not load full model, trying to load just the backbone")
                # If full load fails, try to load just the backbone
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
    def forward(self, x):
        x = self.backbone(x["image"])
        x = self.regression_head(x)
        return x
