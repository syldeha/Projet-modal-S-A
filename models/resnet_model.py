import torch
import torch.nn as nn
import torch
import torch.nn as nn
import os
import timm
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import logging
import wandb
os.environ['HYDRA_FULL_ERROR'] = '1'


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

class SimpleQFormerBlock(nn.Module):
    """One block of the Q-Former, with cross-attention + feedforward."""
    def __init__(self, query_dim, img_feature_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim, kdim=img_feature_dim, vdim=img_feature_dim, num_heads=num_heads, batch_first=True)
        hidden_dim = query_dim
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


#INSERTION DE Q-FORMER


class DinoV2WithQFormer(nn.Module):
    def __init__(self, frozen=False, lora_r=16, lora_alpha=32, lora_dropout=0.05, 
    coef_1=0.8, n_layers=1, hidden_dim=512, 
    num_heads=8, num_layers_q_former=3, num_queries=10,
     q_former_coef=0.4):
        super().__init__()
        # ---- Vision backbone ----
        # self.backbone = timm.create_model('resnet50', pretrained=False)
        # # on ne garde que les features de l'image
        # self.vision_dim = self.backbone.fc.in_features
        # self.backbone.fc = nn.Identity()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        self.vision_dim = self.dim
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
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder_1 = AutoModel.from_pretrained("Syldehayem/train_distilbert-base-uncased_100")
        self.text_encoder_2 = AutoModel.from_pretrained("Syldehayem/train_distilbert-base-uncased_100")
        self.text_dim = self.text_encoder_1.config.hidden_size
        #apply lora to text backbone
        logger_std.info("Applying LoRA to text backbone...")

        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=["q_lin", "k_lin", "v_lin"]
            )
        self.text_encoder_1 = get_peft_model(self.text_encoder_1, txt_lora_config)
        self.text_encoder_2 = get_peft_model(self.text_encoder_2, txt_lora_config)
        self.text_encoder_1.print_trainable_parameters()
        self.text_encoder_2.print_trainable_parameters()
        self.coef_1=coef_1
        self.coef_2=1-coef_1
    

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
        layers = []
        for i in range(n_layers):
            if i==0:
                layers.extend([
                    nn.Linear(fusion_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                    nn.GELU(),
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(0.2),
                    nn.GELU(),
                ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.regression_head = nn.Sequential(*layers)
        # Q-Former
        # logger_std.info("Applying Q-Former...")
        # self.q_former = SimpleQFormer(
        #     query_dim=fusion_dim, 
        #     img_feature_dim=self.vision_dim, 
        #     num_heads=num_heads, 
        #     num_layers=num_layers_q_former, 
        #     num_queries=num_queries)
        # self.q_former_coef=q_former_coef
        # ACTIVATE PARAMETERS OF RESNET50
        # for param in self.backbone.parameters():
        #     param.requires_grad = True
        # ACTIVATE PARAMETERS OF text encoder
        # for param in self.text_encoder_1.parameters():
        #     param.requires_grad = True
        # for param in self.text_encoder_2.parameters():
        #     param.requires_grad = True
        # ACTIVATE PARAMETERS OF regression head
        for m in self.regression_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image
        feats = self.backbone(imgs)  # [batch, vision_dim]
        device = feats.device
        # out_image_backbone = feats #(batch, vision_dim)
        # out_image_q_former=feats.unsqueeze(1) #(batch, 1, vision_dim)
        # out_image_q_former=self.q_former(out_image_q_former) #(batch, n_queries, fusion_dim)
        # out_image_q_former=out_image_q_former.mean(dim=1) #(batch, fusion_dim)
        feats = self.backbone.forward_features(imgs)
        cls_token = feats["x_norm_clstoken"].unsqueeze(1) #(batch, 1, dim)
        patch_tokens = feats["x_norm_patchtokens"] #(batch, n_patches, dim)
        feats = torch.cat( [cls_token, patch_tokens],dim=1)  # shape: [batch, 1+N_patches, dim]
        device=feats.device
        out_image=torch.mean(feats, dim=1)
        out_image_backbone = self.vision_projection(out_image) #(batch, fusion_dim)
        # out_image=self.q_former_coef*out_image_q_former+(1-self.q_former_coef)*out_image_backbone
        out_image=out_image_backbone

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
        # out_text = out_text_2

        #fusion des features
        out = self.final_projection(torch.cat([out_image, out_text], dim=1))
        #regression
        pred = self.regression_head(out)  # [batch, 1]
        # print(pred)
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