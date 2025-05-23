
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import logging
import wandb
os.environ['HYDRA_FULL_ERROR'] = '1'
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

os.environ['HYDRA_FULL_ERROR'] = '1'


class DinoV2Finetune(nn.Module):
    def __init__(self, frozen=False, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()        
        self.vision_dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        logger_std.info("Applying LoRA to vision backbone...")
        vis_lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=["qkv", "proj"]  
            )
        self.backbone = get_peft_model(self.backbone, vis_lora_config)

        #text backbone
        logger_std.info("Loading text backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder_1 = AutoModel.from_pretrained("bert-base-uncased")
        self.text_encoder_2 = AutoModel.from_pretrained("bert-base-uncased")
        self.text_dim = self.text_encoder_1.config.hidden_size
        #apply lora to text backbone
        logger_std.info("Applying LoRA to text backbone...")

        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=["query", "key", "value"]
            )
        self.text_encoder_1 = get_peft_model(self.text_encoder_1, txt_lora_config)
        self.text_encoder_2 = get_peft_model(self.text_encoder_2, txt_lora_config)
        self.text_encoder_1.print_trainable_parameters()
        self.text_encoder_2.print_trainable_parameters()
        self.coef_1=0.5
        self.coef_2=0.5
        

        #fusion des dimensions
        fusion_dim=min(self.vision_dim, self.text_dim)
        self.fusion_dim=fusion_dim
        if self.fusion_dim!=self.vision_dim:
            self.vision_projection=nn.Linear(self.vision_dim, fusion_dim)
            self.vision_projection.requires_grad=True
            self.text_projection_1=nn.Identity()
            self.text_projection_2=nn.Identity()
        else:
            self.vision_projection=nn.Identity()
            self.text_projection_1=nn.Linear(self.text_dim, fusion_dim)
            self.text_projection_2=nn.Linear(self.text_dim, fusion_dim)
            self.text_projection_1.requires_grad=True
            self.text_projection_2.requires_grad=True
        self.final_projection=nn.Linear(2*fusion_dim, fusion_dim)
        self.final_projection.requires_grad=True

        #regresssion head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(256,1),
            # nn.Dropout(0.5),
            nn.Softplus()

        )
        for m in self.regression_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image
        feats = self.backbone.forward_features(imgs)  # [batch, n_tokens, dim]
        cls_token = feats["x_norm_clstoken"].unsqueeze(1) #(batch, 1, dim)
        patch_tokens = feats["x_norm_patchtokens"] #(batch, n_patches, dim)
        feats = torch.cat( [cls_token, patch_tokens],dim=1)  # shape: [batch, 1+N_patches, dim]
        device=feats.device
        out_image=feats.mean(dim=1)
        out_image = self.vision_projection(out_image)

        #recuperation des données textuelles
        encoded_1 = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        
        encoded_2 = self.tokenizer(
            x["title"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        out_text_1 = self.text_encoder_1(**encoded_1)
        out_text_2 = self.text_encoder_2(**encoded_2)
        out_text_1 = out_text_1.last_hidden_state[:,0,:]
        out_text_2 = out_text_2.last_hidden_state[:,0,:]
        out_text_1 = self.text_projection_1(out_text_1) #(batch, fusion_dim)
        out_text_2 = self.text_projection_2(out_text_2) #(batch, fusion_dim)
        out_text = self.coef_1*out_text_1 + self.coef_2*out_text_2

        #fusion des features
        out = self.final_projection(torch.cat([out_image, out_text], dim=1))
        #regression
        pred = self.regression_head(out)  # [batch, 1]
        print(pred)
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