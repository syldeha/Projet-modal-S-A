import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import BlipForQuestionAnswering

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

class VisualModel(nn.Module):
    def __init__(self, frozen=False , pretrained_model="Salesforce/blip-vqa-base", tokenizer_model_path=None):
        super().__init__()
        logger_std.info(f"Initialisation du model VisualModel")
        # Vision backbone
        self.backbone = BlipForQuestionAnswering.from_pretrained(pretrained_model, from_tf=True)
        self.processor = AutoProcessor.from_pretrained(pretrained_model)
        hidden_size = self.backbone.language_projection.out_features  # 
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward(self, x):
        # x["image"]: image tensor (B,C,H,W), x["title"]: list[str]

        # 1. Encode image and text
        inputs = self.processor(images=x["image"], text=x["prompt_resume"], return_tensors="pt")
        outputs = self.backbone(
            pixel_values=inputs["pixel_values"], 
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )
        # Embedding CLS du texte (fusion image+texte)
        pooled_cls = outputs.language_output[:, 0]  # (batch, hidden)
        out = self.regression_head(pooled_cls)
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
            
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False
        

