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




class DinoV2Finetune(nn.Module):# (two CNN models of vision and two text models with lora without Q-former)
    def __init__(self, 
     frozen=False,
     lora_r_text=16, lora_alpha_text=32, lora_dropout_text=0.05, 
     coef_1=0.5, coef_2=0.5, hidden_dim_1=512, hidden_dim_2=256,vision_coef=0.5,use_concatenation=True,vision_classification_model=False,
     dropout_p=0.2, init_method="kaiming", model_name="efficientnet_b5" , batch_or_layer="batch" ,coef_image_1=0.8, 
     classification_model_name="efficientnet_b1" ):
        super().__init__()

        # ---- Vision backbone 1----
        self.coef_image_1=coef_image_1
        self.backbone = timm.create_model(model_name, pretrained=True)
        # on ne garde que les features de l'image
        if hasattr(self.backbone, 'fc'):
            self.vision_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.vision_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head"):
            self.vision_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  
        self.vision_coef=vision_coef
        self.use_concatenation=use_concatenation
        for param in self.backbone.parameters():
            param.requires_grad = True

        # ---- Vision backbone 2----
        #Vision backbone 2 with the a pretrained classification model
        #TROUVER COMMENT SAUVEGARDER LE MODEL SUR HUGGINGFACE
        self.vision_classification_model=classification_model_name
        if self.vision_classification_model : 
            logger_std.info("Loading classification model...")
            self.backbone_2 = timm.create_model(self.vision_classification_model, pretrained=True)
                    # on ne garde que les features de l'image
            if hasattr(self.backbone_2, 'fc'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.fc.in_features
                self.backbone_2.fc = nn.Identity()
            elif hasattr(self.backbone_2, 'classifier'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.classifier.in_features
                self.backbone_2.classifier = nn.Identity()
            elif hasattr(self.backbone_2, "head"):
                self.backbone_2_hidden_dim_2 = self.backbone_2.head.in_features
                self.backbone_2.head = nn.Identity()  
            self.vision_coef=vision_coef
            self.use_concatenation=use_concatenation
            for param in self.backbone_2.parameters():
                param.requires_grad = True
        
        #text backbone
        logger_std.info("Loading text backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder_1 = AutoModel.from_pretrained("Syldehayem/train_distilbert-base-uncased_100")
        self.text_encoder_2 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder_1.config.hidden_size
        #apply lora to text backbone
        logger_std.info("Applying LoRA to text backbone...")

        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
                r=lora_r_text, lora_alpha=lora_alpha_text, lora_dropout=lora_dropout_text, bias="none",
                target_modules=["q_lin","v_lin"]
            )
        self.text_encoder_1 = get_peft_model(self.text_encoder_1, txt_lora_config)
        self.text_encoder_2 = get_peft_model(self.text_encoder_2, txt_lora_config)
        self.text_encoder_1.print_trainable_parameters()
        self.text_encoder_2.print_trainable_parameters()
        for name , param in self.text_encoder_1.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        for name , param in self.text_encoder_2.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        self.coef_1=coef_1
        self.coef_2=coef_2
    

        #fusion des dimensions
        if not self.vision_classification_model : 
            fusion_dim=max(self.vision_dim, self.text_dim)
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
        else : 
            fusion_dim=max(self.vision_dim, self.text_dim)
            self.fusion_dim=fusion_dim
            if self.fusion_dim!=self.vision_dim:
                self.vision_projection=nn.Linear(self.vision_dim, fusion_dim)
                self.vision_projection.requires_grad=True
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Identity()
                self.text_projection_2=nn.Identity()
            else:
                self.vision_projection=nn.Identity()
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_2=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_1.requires_grad=True
                self.text_projection_2.requires_grad=True
                self.vision_projection=nn.Identity()
            self.final_projection=nn.Linear(2*fusion_dim, fusion_dim)
            self.final_projection.requires_grad=True

        #regresssion head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_2, 1),
        )
        # ACTIVATE PARAMETERS OF regression head
        if init_method=="xavier":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
        elif init_method=="kaiming":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image
        feats = self.backbone(imgs)  # [batch, vision_dim]
        device = feats.device
        out_image_1 = feats #[batch, vision_dim]
        out_image_1 = self.vision_projection(out_image_1)
        if self.vision_classification_model : 
            out_image_2 = self.backbone_2(imgs) #[batch, hidden_dim_2]
            out_image_2 = self.backbone_2_vision_projection(out_image_2) #[batch, fusion_dim]
            out_image = self.coef_image_1*out_image_1 + (1-self.coef_image_1)*out_image_2
        else : 
            out_image = out_image_1 #(batch, fusion_dim)
        # out_image = out_image_1 #(batch, fusion_dim)
        #recuperation des données textuelles
        encoded_1 = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        
        encoded_2 = self.tokenizer(
            x["title"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        out_text_1 = self.text_encoder_1(**encoded_1)
        out_text_2 = self.text_encoder_2(**encoded_2)
        input_mask_expanded = encoded_1["attention_mask"].unsqueeze(-1).expand(out_text_1.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_1.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_1 = t_feat
        input_mask_expanded = encoded_2["attention_mask"].unsqueeze(-1).expand(out_text_2.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_2.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_2 = t_feat
        out_text_1 = self.text_projection_1(out_text_1) #(batch, fusion_dim)
        out_text_2 = self.text_projection_2(out_text_2) #(batch, fusion_dim)
        out_text = self.coef_1*out_text_1 + self.coef_2*out_text_2
        # out_text = out_text_2

        #fusion des features
        if self.use_concatenation:
            out = self.final_projection(torch.cat([out_image, out_text], dim=1))
        else:
            out = self.vision_coef*out_image+(1-self.vision_coef)*out_text
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


#INSERTION DE Q-FORMER (two CNN models of vision and two text models with lora and Q-former)
class DinoV2WithQFormer(nn.Module):
    def __init__(self, 
     frozen=False,
     lora_r_text=16, lora_alpha_text=32, lora_dropout_text=0.05, 
     coef_1=0.5, coef_2=0.5, hidden_dim_1=512, hidden_dim_2=256,vision_coef=0.5,use_concatenation=True,vision_classification_model=True,
     dropout_p=0.2, init_method="kaiming", model_name="efficientnet_b5" , batch_or_layer="batch" ,coef_image_1=0.8, 
     classification_model_name="efficientnet_b1",
     q_former_insertion=True,
     num_heads_q_former=8,
     num_layers_q_former=3,
     num_queries_q_former=10,
     q_former_coef=0.5
     ):
        super().__init__()

        # ---- Vision backbone 1----
        self.coef_image_1=coef_image_1
        self.backbone = timm.create_model(model_name, pretrained=True)
        # on ne garde que les features de l'image
        if hasattr(self.backbone, 'fc'):
            self.vision_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.vision_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head"):
            self.vision_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  
        self.vision_coef=vision_coef
        self.use_concatenation=use_concatenation
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ---- Vision backbone 2----
        #Vision backbone 2 with the a pretrained classification model
        #TROUVER COMMENT SAUVEGARDER LE MODEL SUR HUGGINGFACE
        self.vision_classification_model=classification_model_name
        if self.vision_classification_model : 
            logger_std.info("Loading classification model...")
            self.backbone_2 = timm.create_model(self.vision_classification_model, pretrained=True)
                    # on ne garde que les features de l'image
            if hasattr(self.backbone_2, 'fc'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.fc.in_features
                self.backbone_2.fc = nn.Identity()
            elif hasattr(self.backbone_2, 'classifier'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.classifier.in_features
                self.backbone_2.classifier = nn.Identity()
            elif hasattr(self.backbone_2, "head"):
                self.backbone_2_hidden_dim_2 = self.backbone_2.head.in_features
                self.backbone_2.head = nn.Identity()  
            self.vision_coef=vision_coef
            self.use_concatenation=use_concatenation
            for param in self.backbone_2.parameters():
                param.requires_grad = False
        
        #text backbone
        logger_std.info("Loading text backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder_1 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_encoder_2 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder_1.config.hidden_size
        #apply lora to text backbone
        logger_std.info("Applying LoRA to text backbone...")

        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
                r=lora_r_text, lora_alpha=lora_alpha_text, lora_dropout=lora_dropout_text, bias="none",
                target_modules=["q_lin","v_lin"]
            )
        self.text_encoder_1 = get_peft_model(self.text_encoder_1, txt_lora_config)
        self.text_encoder_2 = get_peft_model(self.text_encoder_2, txt_lora_config)
        self.text_encoder_1.print_trainable_parameters()
        self.text_encoder_2.print_trainable_parameters()
        for name , param in self.text_encoder_1.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        for name , param in self.text_encoder_2.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        self.coef_1=coef_1
        self.coef_2=coef_2
    

        #fusion des dimensions
        if not self.vision_classification_model : 
            fusion_dim=max(self.vision_dim, self.text_dim)
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
        else : 
            fusion_dim=max(self.vision_dim, self.text_dim)
            self.fusion_dim=fusion_dim
            if self.fusion_dim!=self.vision_dim:
                self.vision_projection=nn.Linear(self.vision_dim, fusion_dim)
                self.vision_projection.requires_grad=True
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Identity()
                self.text_projection_2=nn.Identity()
            else:
                self.vision_projection=nn.Identity()
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_2=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_1.requires_grad=True
                self.text_projection_2.requires_grad=True
                self.vision_projection=nn.Identity()
            self.final_projection=nn.Linear(2*fusion_dim, fusion_dim)
            self.final_projection.requires_grad=True

        self.q_former_insertion=q_former_insertion
        self.num_heads_q_former=num_heads_q_former
        self.num_layers_q_former=num_layers_q_former
        self.num_queries_q_former=num_queries_q_former
        self.q_former_coef=q_former_coef
        #Q-Former insertion
        if self.q_former_insertion:
            logger_std.info("Applying Q-Former...") #input(batch , 1, vision_dim)
            self.q_former = SimpleQFormer(
                query_dim=fusion_dim,  #taille de sorties 
                img_feature_dim=self.vision_dim, 
                num_heads=self.num_heads_q_former, 
                num_layers=self.num_layers_q_former, 
                num_queries=self.num_queries_q_former)
            #nombre de paramètres du q-former
            logger_std.info(f"Nombre de paramètres du q-former: {sum(p.numel() for p in self.q_former.parameters())}")


        #regresssion head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_2, 1),
        )
        # ACTIVATE PARAMETERS OF regression head
        if init_method=="xavier":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
        elif init_method=="kaiming":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image
        feats = self.backbone(imgs)  # [batch, vision_dim]
        device = feats.device
        out_image_1 = feats #[batch, vision_dim]
        if self.q_former_insertion:
            out_image_1 = out_image_1.unsqueeze(1) #(batch, 1, vision_dim)
            out_image_1 = self.q_former(out_image_1) #(batch, n_queries, fusion_dim)
            out_image_1 = out_image_1.mean(dim=1) #(batch, fusion_dim)
        else : 
            out_image_1 = self.vision_projection(out_image_1) #(batch, fusion_dim)
        if self.vision_classification_model : 
            out_image_2 = self.backbone_2(imgs) #[batch, hidden_dim_2]
            out_image_2 = self.backbone_2_vision_projection(out_image_2) #[batch, fusion_dim]
            out_image = self.coef_image_1*out_image_1 + (1-self.coef_image_1)*out_image_2
        else : 
            out_image = out_image_1 #(batch, fusion_dim)
        # out_image = out_image_1 #(batch, fusion_dim)
        #recuperation des données textuelles
        encoded_1 = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        
        encoded_2 = self.tokenizer(
            x["title"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        out_text_1 = self.text_encoder_1(**encoded_1)
        out_text_2 = self.text_encoder_2(**encoded_2)
        input_mask_expanded = encoded_1["attention_mask"].unsqueeze(-1).expand(out_text_1.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_1.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_1 = t_feat
        input_mask_expanded = encoded_2["attention_mask"].unsqueeze(-1).expand(out_text_2.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_2.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_2 = t_feat
        out_text_1 = self.text_projection_1(out_text_1) #(batch, fusion_dim)
        out_text_2 = self.text_projection_2(out_text_2) #(batch, fusion_dim)
        out_text = self.coef_1*out_text_1 + self.coef_2*out_text_2
        # out_text = out_text_2

        #fusion des features
        if self.use_concatenation:
            out = self.final_projection(torch.cat([out_image, out_text], dim=1))
        else:
            out = self.vision_coef*out_image+(1-self.vision_coef)*out_text
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

#INSERTION INPUT SINUSOIDAL
class DinoV2WithQFormerAndSinusoidal(nn.Module):#(two CNN models of vision and two text models with lora and Q-former and sinusoidal input)
    def __init__(self, 
     frozen=False,
     lora_r_text=16, lora_alpha_text=32, lora_dropout_text=0.05, 
     coef_1=0.5, coef_2=0.5, hidden_dim_1=512, hidden_dim_2=256,vision_coef=0.5,use_concatenation=True,vision_classification_model=True,
     dropout_p=0.2, init_method="kaiming", model_name="efficientnet_b5" , batch_or_layer="batch" ,coef_image_1=0.8, 
     classification_model_name="efficientnet_b1",
     q_former_insertion=True,
     num_heads_q_former=8,
     num_layers_q_former=3,
     num_queries_q_former=10,
     q_former_coef=0.5
     ):
        super().__init__()

        # ---- Vision backbone 1----
        self.coef_image_1=coef_image_1
        self.backbone = timm.create_model(model_name, pretrained=True)
        # on ne garde que les features de l'image
        if hasattr(self.backbone, 'fc'):
            self.vision_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.vision_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head"):
            self.vision_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  
        # self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        # self.backbone.head = nn.Identity()
        # self.vision_dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.vision_coef=vision_coef
        self.use_concatenation=use_concatenation
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ---- Vision backbone 2----
        #Vision backbone 2 with the a pretrained classification model
        #TROUVER COMMENT SAUVEGARDER LE MODEL SUR HUGGINGFACE
        self.vision_classification_model=classification_model_name
        if self.vision_classification_model : 
            logger_std.info("Loading classification model...")
            self.backbone_2 = timm.create_model(self.vision_classification_model, pretrained=True)
                    # on ne garde que les features de l'image
            if hasattr(self.backbone_2, 'fc'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.fc.in_features
                self.backbone_2.fc = nn.Identity()
            elif hasattr(self.backbone_2, 'classifier'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.classifier.in_features
                self.backbone_2.classifier = nn.Identity()
            elif hasattr(self.backbone_2, "head"):
                self.backbone_2_hidden_dim_2 = self.backbone_2.head.in_features
                self.backbone_2.head = nn.Identity()  
            self.vision_coef=vision_coef
            self.use_concatenation=use_concatenation
            for param in self.backbone_2.parameters():
                param.requires_grad = False
        
        #text backbone
        logger_std.info("Loading text backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder_1 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_encoder_2 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder_1.config.hidden_size
        #apply lora to text backbone
        logger_std.info("Applying LoRA to text backbone...")

        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
                r=lora_r_text, lora_alpha=lora_alpha_text, lora_dropout=lora_dropout_text, bias="none",
                target_modules=["q_lin","v_lin"]
            )
        self.text_encoder_1 = get_peft_model(self.text_encoder_1, txt_lora_config)
        self.text_encoder_2 = get_peft_model(self.text_encoder_2, txt_lora_config)
        self.text_encoder_1.print_trainable_parameters()
        self.text_encoder_2.print_trainable_parameters()
        for name , param in self.text_encoder_1.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        for name , param in self.text_encoder_2.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        self.coef_1=coef_1
        self.coef_2=coef_2
    


        #fusion des dimensions
        if not self.vision_classification_model : 
            fusion_dim=max(self.vision_dim, self.text_dim)
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
        else : 
            fusion_dim=max(self.vision_dim, self.text_dim)
            self.fusion_dim=fusion_dim
            if self.fusion_dim!=self.vision_dim:
                self.vision_projection=nn.Linear(self.vision_dim, fusion_dim)
                self.vision_projection.requires_grad=True
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Identity()
                self.text_projection_2=nn.Identity()
            else:
                self.vision_projection=nn.Identity()
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_2=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_1.requires_grad=True
                self.text_projection_2.requires_grad=True
                self.vision_projection=nn.Identity()
            self.final_projection=nn.Linear(3*fusion_dim, fusion_dim)
            self.final_projection.requires_grad=True

        self.q_former_insertion=q_former_insertion
        self.num_heads_q_former=num_heads_q_former
        self.num_layers_q_former=num_layers_q_former
        self.num_queries_q_former=num_queries_q_former
        self.q_former_coef=q_former_coef
        #Q-Former insertion
        if self.q_former_insertion:
            logger_std.info("Applying Q-Former...") #input(batch , 1, vision_dim)
            self.q_former = SimpleQFormer(
                query_dim=fusion_dim,  #taille de sorties 
                img_feature_dim=self.vision_dim, 
                num_heads=self.num_heads_q_former, 
                num_layers=self.num_layers_q_former, 
                num_queries=self.num_queries_q_former)
            #nombre de paramètres du q-former
            logger_std.info(f"Nombre de paramètres du q-former: {sum(p.numel() for p in self.q_former.parameters())}")

        # DAte encoding 
        date_embedding_dim=6
        self.date_projection= nn.Linear(date_embedding_dim, fusion_dim)
        self.date_projection.requires_grad=True

        #regresssion head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_2, 1),
            nn.ReLU()
        )
        # ACTIVATE PARAMETERS OF regression head
        if init_method=="xavier":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
        elif init_method=="kaiming":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image
        feats = self.backbone(imgs)  # [batch, vision_dim]
        device = feats.device
        out_image_1 = feats #[batch, vision_dim]
        #recuperation des features de l'image
        # feats = self.backbone.forward_features(imgs)  # [batch, vision_dim]
        # cls_token = feats["x_norm_clstoken"].unsqueeze(1) # (batch,1,D)
        # # patch_tokens = feats["x_norm_patchtokens"]        # (batch,N,D)
        # # out_image_1 = torch.cat([cls_token, patch_tokens], dim=1) # (batch,1+N,D)
        # out_image_1 = cls_token.squeeze(1) #(batch,D=self.vision_dim)
        device = out_image_1.device
        if self.q_former_insertion:
            out_image_1 = out_image_1.unsqueeze(1) #(batch, 1, vision_dim)
            out_image_1 = self.q_former(out_image_1) #(batch, n_queries, fusion_dim)
            out_image_1 = out_image_1.mean(dim=1) #(batch, fusion_dim)
        else : 
            out_image_1 = self.vision_projection(out_image_1) #(batch, fusion_dim)
        if self.vision_classification_model : 
            out_image_2 = self.backbone_2(imgs) #[batch, hidden_dim_2]
            out_image_2 = self.backbone_2_vision_projection(out_image_2) #[batch, fusion_dim]
            out_image = self.coef_image_1*out_image_1 + (1-self.coef_image_1)*out_image_2
        else : 
            out_image = out_image_1 #(batch, fusion_dim)
        # out_image = out_image_1 #(batch, fusion_dim)
        #recuperation des données textuelles
        encoded_1 = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        
        encoded_2 = self.tokenizer(
            x["title"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        out_text_1 = self.text_encoder_1(**encoded_1)
        out_text_2 = self.text_encoder_2(**encoded_2)
        input_mask_expanded = encoded_1["attention_mask"].unsqueeze(-1).expand(out_text_1.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_1.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_1 = t_feat
        input_mask_expanded = encoded_2["attention_mask"].unsqueeze(-1).expand(out_text_2.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_2.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_2 = t_feat
        out_text_1 = self.text_projection_1(out_text_1) #(batch, fusion_dim)
        out_text_2 = self.text_projection_2(out_text_2) #(batch, fusion_dim)
        out_text = self.coef_1*out_text_1 + self.coef_2*out_text_2
        # out_text = out_text_2

        #out_date
        tabular_feats = torch.stack([
            x["day_sin"].to(device), x["day_cos"].to(device), x["month_sin"].to(device), x["month_cos"].to(device), x["hour_sin"].to(device), x["hour_cos"].to(device)
        ], dim=1)
        tabular_feats = tabular_feats.float()
        out_date = self.date_projection(tabular_feats) #(batch, fusion_dim)

        #fusion des features
        if self.use_concatenation:
            out = self.final_projection(torch.cat([out_image, out_text, out_date], dim=1))
        else:
            out = self.vision_coef*out_image+(1-self.vision_coef)/2*out_text+(1-self.vision_coef)/2*out_date
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






#INSERTION INPUT SINUSOIDAL
class RealDinoV2WithQFormerAndSinusoidal(nn.Module): #(DinoV2 and CNN models for vision and two text models with lora and Q-former and sinusoidal input)
    def __init__(self, 
     frozen=False,
     lora_r_text=16, lora_alpha_text=32, lora_dropout_text=0.05, 
     coef_1=0.5, coef_2=0.5, hidden_dim_1=512, hidden_dim_2=256,vision_coef=0.5,use_concatenation=True,vision_classification_model=True,
     dropout_p=0.2, init_method="kaiming", model_name="efficientnet_b5" , batch_or_layer="batch" ,coef_image_1=0.8, 
     classification_model_name="efficientnet_b1",
     q_former_insertion=True,
     num_heads_q_former=8,
     num_layers_q_former=3,
     num_queries_q_former=10,
     q_former_coef=0.5,
     lora_r_vision=16, lora_alpha_vision=32, lora_dropout_vision=0.05, 
     ):
        super().__init__()

        # ---- Vision backbone 1----
        self.coef_image_1=coef_image_1

        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        self.backbone.head = nn.Identity()
        self.vision_dim = self.backbone.norm.normalized_shape[0]
        self.vision_coef=vision_coef
        self.use_concatenation=use_concatenation
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger_std.info("Applying LoRA to vision backbone...")
        vis_lora_config = LoraConfig(
        r=lora_r_vision, lora_alpha=lora_alpha_vision, lora_dropout=lora_dropout_vision, bias="none",
        target_modules=["qkv", "proj"]
        )
        self.backbone = get_peft_model(self.backbone, vis_lora_config)
        self.backbone.print_trainable_parameters()
        for name , param in self.backbone.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        # ---- Vision backbone 2----
        #Vision backbone 2 with the a pretrained classification model
        #TROUVER COMMENT SAUVEGARDER LE MODEL SUR HUGGINGFACE
        self.vision_classification_model=classification_model_name
        if self.vision_classification_model : 
            logger_std.info("Loading classification model...")
            self.backbone_2 = timm.create_model(self.vision_classification_model, pretrained=True)
                    # on ne garde que les features de l'image
            if hasattr(self.backbone_2, 'fc'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.fc.in_features
                self.backbone_2.fc = nn.Identity()
            elif hasattr(self.backbone_2, 'classifier'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.classifier.in_features
                self.backbone_2.classifier = nn.Identity()
            elif hasattr(self.backbone_2, "head"):
                self.backbone_2_hidden_dim_2 = self.backbone_2.head.in_features
                self.backbone_2.head = nn.Identity()  
            self.vision_coef=vision_coef
            self.use_concatenation=use_concatenation
            for param in self.backbone_2.parameters():
                param.requires_grad = False
        
        #text backbone
        logger_std.info("Loading text backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder_1 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_encoder_2 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder_1.config.hidden_size
        #apply lora to text backbone
        logger_std.info("Applying LoRA to text backbone...")

        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
                r=lora_r_text, lora_alpha=lora_alpha_text, lora_dropout=lora_dropout_text, bias="none",
                target_modules=["q_lin","v_lin"]
            )
        self.text_encoder_1 = get_peft_model(self.text_encoder_1, txt_lora_config)
        self.text_encoder_2 = get_peft_model(self.text_encoder_2, txt_lora_config)
        self.text_encoder_1.print_trainable_parameters()
        self.text_encoder_2.print_trainable_parameters()
        for name , param in self.text_encoder_1.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        for name , param in self.text_encoder_2.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        self.coef_1=coef_1
        self.coef_2=coef_2
    


        #fusion des dimensions
        if not self.vision_classification_model : 
            fusion_dim=max(self.vision_dim, self.text_dim)
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
        else : 
            fusion_dim=max(self.vision_dim, self.text_dim)
            self.fusion_dim=fusion_dim
            if self.fusion_dim!=self.vision_dim:
                self.vision_projection=nn.Linear(self.vision_dim, fusion_dim)
                self.vision_projection.requires_grad=True
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Identity()
                self.text_projection_2=nn.Identity()
            else:
                self.vision_projection=nn.Identity()
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_2=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_1.requires_grad=True
                self.text_projection_2.requires_grad=True
                self.vision_projection=nn.Identity()
            self.final_projection=nn.Linear(3*fusion_dim, fusion_dim)
            self.final_projection.requires_grad=True

        self.q_former_insertion=q_former_insertion
        self.num_heads_q_former=num_heads_q_former
        self.num_layers_q_former=num_layers_q_former
        self.num_queries_q_former=num_queries_q_former
        self.q_former_coef=q_former_coef
        #Q-Former insertion
        if self.q_former_insertion:
            logger_std.info("Applying Q-Former...") #input(batch , 1, vision_dim)
            self.q_former = SimpleQFormer(
                query_dim=fusion_dim,  #taille de sorties 
                img_feature_dim=self.vision_dim, 
                num_heads=self.num_heads_q_former, 
                num_layers=self.num_layers_q_former, 
                num_queries=self.num_queries_q_former)


        #regresssion head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_2, 1),
        )
        # ACTIVATE PARAMETERS OF regression head
        if init_method=="xavier":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
        elif init_method=="kaiming":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image
        # feats = self.backbone(imgs)  # [batch, vision_dim]
        # device = feats.device
        # out_image_1 = feats #[batch, vision_dim]
        #recuperation des features de l'image
        feats = self.backbone.forward_features(imgs)  # [batch, vision_dim]
        cls_token = feats["x_norm_clstoken"].unsqueeze(1) # (batch,1,D)
        # patch_tokens = feats["x_norm_patchtokens"]        # (batch,N,D)
        # out_image_1 = torch.cat([cls_token, patch_tokens], dim=1) # (batch,1+N,D)
        out_image_1 = cls_token.squeeze(1) #(batch,D=self.vision_dim)
        device = out_image_1.device
        if self.q_former_insertion:
            out_image_1 = out_image_1.unsqueeze(1) #(batch, 1, vision_dim)
            out_image_1 = self.q_former(out_image_1) #(batch, n_queries, fusion_dim)
            out_image_1 = out_image_1.mean(dim=1) #(batch, fusion_dim)
        else : 
            out_image_1 = self.vision_projection(out_image_1) #(batch, fusion_dim)
        if self.vision_classification_model : 
            out_image_2 = self.backbone_2(imgs) #[batch, hidden_dim_2]
            out_image_2 = self.backbone_2_vision_projection(out_image_2) #[batch, fusion_dim]
            out_image = self.coef_image_1*out_image_1 + (1-self.coef_image_1)*out_image_2
        else : 
            out_image = out_image_1 #(batch, fusion_dim)
        # out_image = out_image_1 #(batch, fusion_dim)
        #recuperation des données textuelles
        encoded_1 = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        
        encoded_2 = self.tokenizer(
            x["title"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        out_text_1 = self.text_encoder_1(**encoded_1)
        out_text_2 = self.text_encoder_2(**encoded_2)
        input_mask_expanded = encoded_1["attention_mask"].unsqueeze(-1).expand(out_text_1.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_1.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_1 = t_feat
        input_mask_expanded = encoded_2["attention_mask"].unsqueeze(-1).expand(out_text_2.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_2.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_2 = t_feat
        out_text_1 = self.text_projection_1(out_text_1) #(batch, fusion_dim)
        out_text_2 = self.text_projection_2(out_text_2) #(batch, fusion_dim)
        out_text = self.coef_1*out_text_1 + self.coef_2*out_text_2
        # out_text = out_text_2

        #out_date
        tabular_feats = torch.stack([
            x["day_sin"].to(device), x["day_cos"].to(device), x["month_sin"].to(device), x["month_cos"].to(device), x["hour_sin"].to(device), x["hour_cos"].to(device)
        ], dim=1)
        tabular_feats = tabular_feats.float()
        out_date = self.date_projection(tabular_feats) #(batch, fusion_dim)

        #fusion des features
        if self.use_concatenation:
            out = self.final_projection(torch.cat([out_image, out_text, out_date], dim=1))
        else:
            out = self.vision_coef*out_image+(1-self.vision_coef)/2*out_text+(1-self.vision_coef)/2*out_date
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


#REPRODUCTION MODEL DU LEADERBOARD
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import timm
import os
import logging

logger_std = logging.getLogger(__name__)
#
class MultiModalModel(nn.Module):
    def __init__(self, 
                 lora_r_text=16, lora_alpha_text=32, lora_dropout_text=0.05,
                 lora_r_vision=16, lora_alpha_vision=32, lora_dropout_vision=0.05,
                 hidden_dim_1=512, hidden_dim_2=256,
                 dropout_p=0.2, init_method="kaiming",
                 fusion_dim=768,
                 num_channels=47,
                 year_binary_dim=4,
                 use_vision_classification_model=True,
                 classification_model_name="efficientnet_b1",
                 dinov_vision_coef=0.5,
                 use_Q_former=False,
                 num_heads_q_former=8,
                 num_layers_q_former=3,
                 num_queries_q_former=10,
                 q_former_coef=0.5,
                 ):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.num_channels = num_channels
        self.year_binary_dim = year_binary_dim
        
        # ---- Vision backbone (Miniature) ----
        logger_std.info("Loading vision backbone (DINOv2)...")
        self.vision_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.vision_backbone.head = nn.Identity()
        self.vision_dim = self.vision_backbone.norm.normalized_shape[0]
        self.dinov_vision_coef=dinov_vision_coef
        
        # Freeze vision backbone parameters
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # Apply LoRA to vision backbone
        logger_std.info("Applying LoRA to vision backbone...")
        vis_lora_config = LoraConfig(
            r=lora_r_vision, 
            lora_alpha=lora_alpha_vision, 
            lora_dropout=lora_dropout_vision, 
            bias="none",
            target_modules=["qkv", "proj"]
        
        )
        self.vision_backbone = get_peft_model(self.vision_backbone, vis_lora_config)

        # Only LoRA parameters are trainable
        for name, param in self.vision_backbone.named_parameters():
            if "lora" not in name:
                param.requires_grad = False


         #second image classification
        self.use_classification=use_vision_classification_model
        self.classification_model_name=classification_model_name
        # if self.use_classification : 
        #     logger_std.info("Loading classification model...")
        #     self.backbone_2 = timm.create_model(self.classification_model_name, pretrained=True)
        #             # on ne garde que les features de l'image
        #     if hasattr(self.backbone_2, 'fc'):
        #         self.backbone_2_hidden_dim_2 = self.backbone_2.fc.in_features
        #         self.backbone_2.fc = nn.Identity()
        #     elif hasattr(self.backbone_2, 'classifier'):
        #         self.backbone_2_hidden_dim_2 = self.backbone_2.classifier.in_features
        #         self.backbone_2.classifier = nn.Identity()
        #     elif hasattr(self.backbone_2, "head"):
        #         self.backbone_2_hidden_dim_2 = self.backbone_2.head.in_features
        #         self.backbone_2.head = nn.Identity()  
        #     for param in self.backbone_2.parameters():
        #         param.requires_grad = False
        #     self.classification_projection=nn.Linear(self.backbone_2_hidden_dim_2, self.fusion_dim)
        #use dinov for classification
        if self.use_classification:
            self.backbone_2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
            self.backbone_2.head = nn.Identity()
            self.backbone_2_hidden_dim_2 = self.backbone_2.norm.normalized_shape[0]
            self.classification_projection=nn.Linear(self.backbone_2_hidden_dim_2, self.fusion_dim)
            for param in self.backbone_2.parameters():
                param.requires_grad = False
            #apply lora to classification backbone
            self.backbone_2 = get_peft_model(self.backbone_2, vis_lora_config)
            for name, param in self.backbone_2.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False

        #Q-former
        self.use_Q_former=use_Q_former
        self.num_heads_q_former=num_heads_q_former
        self.num_layers_q_former=num_layers_q_former
        self.num_queries_q_former=num_queries_q_former
        self.q_former_coef=q_former_coef
        if self.use_Q_former:
            logger_std.info("Loading Q-former...")
            self.q_former = SimpleQFormer(
                query_dim=self.fusion_dim,
                img_feature_dim=self.vision_dim,
                num_heads=self.num_heads_q_former,
                num_layers=self.num_layers_q_former,
                num_queries=self.num_queries_q_former
            )
            self.q_former_coef=q_former_coef
            #nombre de paramètres du q-former
            logger_std.info(f"Nombre de paramètres du q-former: {sum(p.numel() for p in self.q_former.parameters())}")

        # ---- Channel embedding ----
        # fixe number of channels to 43
        
        # ---- Year binary encoding ----
        # Year is encoded as binary string with fixed length
            
        # ---- Text encoders (Titre) ----
        logger_std.info("Loading text backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Apply LoRA to text encoder
        logger_std.info("Applying LoRA to text backbone...")
        txt_lora_config = LoraConfig(
            r=lora_r_text, 
            lora_alpha=lora_alpha_text, 
            lora_dropout=lora_dropout_text, 
            bias="none",
            target_modules=["q_lin", "v_lin"]
        )
        self.text_encoder = get_peft_model(self.text_encoder, txt_lora_config)
        
        # Only LoRA parameters are trainable
        for name, param in self.text_encoder.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        #text encoding (prompt resume)
        self.prompt_resume_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.prompt_resume_dim = self.prompt_resume_encoder.config.hidden_size
        self.prompt_resume_projection = nn.Sequential(
            nn.Linear(self.prompt_resume_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        self.prompt_resume_encoder = get_peft_model(self.prompt_resume_encoder, txt_lora_config)
        for name, param in self.prompt_resume_encoder.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        # ---- Date encoding ----
        self.date_embedding_dim = 6  # sin/cos for day, month, hour
        
        # ---- Projection layers (MLP + Layer Norm) ----
        # Vision projection
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # Channel embedding projection (already at fusion_dim, just add LayerNorm)
        self.channel_projection = nn.Sequential(
            nn.Linear(self.num_channels, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # Year binary projection
        self.year_projection = nn.Sequential(
            nn.Linear(self.year_binary_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # Date projection
        self.date_projection = nn.Sequential(
            nn.Linear(self.date_embedding_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # ---- Fusion layer ----
        # 5 modalities: vision, channel, year, text, date
        self.fusion_layer = nn.Sequential(
            nn.Linear(5 * self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim)
        )
        
        # ---- Regression head (Layer Norm + MLP) ----
        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, hidden_dim_1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.GELU(), 
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim_2, 1),
            nn.ReLU()
        )
        
        # Initialize weights
        self._initialize_weights(init_method)
        
    def _initialize_weights(self, init_method):
        """Initialize trainable parameters"""
        # Initialize weights
        for module in [self.vision_projection, self.channel_projection,
                      self.year_projection, self.text_projection, 
                      self.date_projection, self.fusion_layer, self.regression_head]:
            for m in module:
                if isinstance(m, nn.Linear):
                    if init_method == "xavier":
                        nn.init.xavier_uniform_(m.weight)
                    elif init_method == "kaiming":
                        nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                        
    def forward(self, x):
        device = next(self.parameters()).device
        
        # ---- Vision features (Miniature) ----
        imgs = x["image"].to(device)  # [batch, 3, H, W]
        vision_feats = self.vision_backbone.forward_features(imgs)
        #cas cls token
        vision_cls_token = vision_feats["x_norm_clstoken"].squeeze(1)  # [batch, vision_dim]
        vision_features = self.vision_projection(vision_cls_token)  # [batch, fusion_dim]
        #cas all tokens
        # cls_token = vision_feats["x_norm_clstoken"].unsqueeze(1) # (B,1,D)
        # patch_tokens = vision_feats["x_norm_patchtokens"]        # (B,N,D)
        # out_image_1 = torch.cat([cls_token, patch_tokens], dim=1) # (B,1+N,D)
        # vision_features = self.vision_projection(out_image_1.mean(dim=1))  # [batch, fusion_dim]
        #Q-former
        if self.use_Q_former:
            vision_cls_token_former=vision_cls_token.unsqueeze(1) # [batch, 1, fusion_dim]
            vision_cls_token_former = self.q_former(vision_cls_token_former) # [batch, num_queries_q_former, fusion_dim]
            vision_cls_token_former = vision_cls_token_former.mean(dim=1) # [batch, fusion_dim]
            vision_features = (1-self.q_former_coef) * vision_features + self.q_former_coef * vision_cls_token_former #fusion des features (batch, fusion_dim)

        #image classification
        if self.use_classification:
            vision_cls_token_2 = self.backbone_2.forward_features(imgs)
            vision_cls_token_2 = vision_cls_token_2["x_norm_clstoken"].squeeze(1) # [batch, vision_dim]
            vision_cls_token_2 = self.classification_projection(vision_cls_token_2)
            vision_features = self.dinov_vision_coef * vision_features + (1 - self.dinov_vision_coef) * vision_cls_token_2 #fusion des features (batch, fusion_dim)

        # ---- Channel features ----
        channel_id = x["channel_onehot"].to(device)
        channel_features = self.channel_projection(channel_id)  # [batch, fusion_dim]
        
        # ---- Year binary features ----
        year_bin = x["year_new"].to(device)
        # Ensure year_bin is 2D [batch, year_binary_dim]
        if year_bin.dim() > 2:
            year_bin = year_bin.view(year_bin.size(0), -1)
        year_bin = year_bin.float()
        year_features = self.year_projection(year_bin)  # [batch, fusion_dim]
        
        # ---- Text features (Titre) ----
        # Assuming we use title as the main text input
        encoded_text = self.tokenizer(
            x["title"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)

        # Get text output from encoder
        text_output = self.text_encoder(**encoded_text)

        # Mean pooling over sequence length
        input_mask_expanded = encoded_text["attention_mask"].unsqueeze(-1).expand(
            text_output.last_hidden_state.size()
        ).float()
        sum_embeddings = torch.sum(text_output.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        text_pooled = sum_embeddings / sum_mask.clamp(min=1e-9)  # [batch, text_dim]
        # text_features = self.text_projection(text_pooled)  # [batch, fusion_dim]

        # ---- Text features (Prompt resume) ----
        prompt_resume_encoded_text = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        #token de classification
        prompt_resume_text_output = self.prompt_resume_encoder(**prompt_resume_encoded_text)
        prompt_input_mask_expanded = prompt_resume_encoded_text["attention_mask"].unsqueeze(-1).expand(
            prompt_resume_text_output.last_hidden_state.size()
        ).float()
        prompt_sum_embeddings = torch.sum(prompt_resume_text_output.last_hidden_state * prompt_input_mask_expanded, 1)
        prompt_sum_mask = prompt_input_mask_expanded.sum(1)
        prompt_text_pooled = prompt_sum_embeddings / prompt_sum_mask.clamp(min=1e-9)  # [batch, text_dim]

        #somme prompt resume et titre
        text_features = 0.5 * text_pooled + 0.5 * prompt_text_pooled
        text_features = self.text_projection(text_features)  # [batch, fusion_dim]

        # ---- Date features ----
        tabular_feats = torch.stack([
            x["day_sin"], 
            x["day_cos"], 
            x["month_sin"], 
            x["month_cos"], 
            x["hour_sin"], 
            x["hour_cos"]
        ], dim=1).float()  # [batch, 6]
        
        date_features = self.date_projection(tabular_feats.to(device))  # [batch, fusion_dim]
        
        # ---- Fusion ----
        fused_features = torch.cat([
            vision_features, 
            channel_features,
            year_features,
            text_features, 
            date_features
        ], dim=1)  # [batch, 5 * fusion_dim]
        
        fused_output = self.fusion_layer(fused_features)  # [batch, fusion_dim]
        
        # ---- Prediction ----
        prediction = self.regression_head(fused_output)  # [batch, 1]
        
        return prediction
    
    def load_checkpoint(self, checkpoint_path, device=None, load_full=True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint

            if load_full:
                missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)
                print(f"Loaded full model from {checkpoint_path}")
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False



#INSERTION DE Q-FORMER
class RealDinoV2WithQFormer(nn.Module): #utilise 224 comme size d'image
    def __init__(self, 
     frozen=False,
     lora_r_vision=16, lora_alpha_vision=32, lora_dropout_vision=0.05, 
     lora_r_text=16, lora_alpha_text=32, lora_dropout_text=0.05, 
     coef_1=0.5, coef_2=0.5, hidden_dim_1=512, hidden_dim_2=256,vision_coef=0.5,use_concatenation=True,vision_classification_model=True,
     dropout_p=0.2, init_method="kaiming" , batch_or_layer="batch" ,coef_image_1=0.8, 
     classification_model_name="efficientnet_b1",
     q_former_insertion=True,
     num_heads_q_former=8,
     num_layers_q_former=3,
     num_queries_q_former=10,
     q_former_coef=0.5
     ):
        super().__init__()
        self.coef_image_1=coef_image_1
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        self.vision_dim = self.dim
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        logger_std.info("Applying LoRA to vision backbone...")
        vis_lora_config = LoraConfig(
        r=lora_r_vision, lora_alpha=lora_alpha_vision, lora_dropout=lora_dropout_vision, bias="none",
        target_modules=["qkv", "proj"]
        )
        self.backbone = get_peft_model(self.backbone, vis_lora_config)
        self.backbone.print_trainable_parameters()
        for name , param in self.backbone.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        # ---- Vision backbone 2----
        #Vision backbone 2 with the a pretrained classification model
        #TROUVER COMMENT SAUVEGARDER LE MODEL SUR HUGGINGFACE
        self.vision_classification_model=classification_model_name
        if self.vision_classification_model : 
            logger_std.info("Loading classification model...")
            self.backbone_2 = timm.create_model(self.vision_classification_model, pretrained=True)
                    # on ne garde que les features de l'image
            if hasattr(self.backbone_2, 'fc'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.fc.in_features
                self.backbone_2.fc = nn.Identity()
            elif hasattr(self.backbone_2, 'classifier'):
                self.backbone_2_hidden_dim_2 = self.backbone_2.classifier.in_features
                self.backbone_2.classifier = nn.Identity()
            elif hasattr(self.backbone_2, "head"):
                self.backbone_2_hidden_dim_2 = self.backbone_2.head.in_features
                self.backbone_2.head = nn.Identity()  
            self.vision_coef=vision_coef
            self.use_concatenation=use_concatenation
            for param in self.backbone_2.parameters():
                param.requires_grad = False
        
        #text backbone
        logger_std.info("Loading text backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder_1 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_encoder_2 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder_1.config.hidden_size
        #apply lora to text backbone
        logger_std.info("Applying LoRA to text backbone...")

        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
                r=lora_r_text, lora_alpha=lora_alpha_text, lora_dropout=lora_dropout_text, bias="none",
                target_modules=["q_lin", "k_lin", "v_lin"]
            )
        self.text_encoder_1 = get_peft_model(self.text_encoder_1, txt_lora_config)
        self.text_encoder_2 = get_peft_model(self.text_encoder_2, txt_lora_config)
        self.text_encoder_1.print_trainable_parameters()
        self.text_encoder_2.print_trainable_parameters()
        for name , param in self.text_encoder_1.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        for name , param in self.text_encoder_2.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        self.coef_1=coef_1
        self.coef_2=coef_2
    

        #fusion des dimensions
        if not self.vision_classification_model : 
            fusion_dim=max(self.vision_dim, self.text_dim)
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
        else : 
            fusion_dim=max(self.vision_dim, self.text_dim)
            self.fusion_dim=fusion_dim
            if self.fusion_dim!=self.vision_dim:
                self.vision_projection=nn.Linear(self.vision_dim, fusion_dim)
                self.vision_projection.requires_grad=True
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Identity()
                self.text_projection_2=nn.Identity()
            else:
                self.vision_projection=nn.Identity()
                self.backbone_2_vision_projection=nn.Linear(self.backbone_2_hidden_dim_2, fusion_dim)
                self.backbone_2_vision_projection.requires_grad=True
                self.text_projection_1=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_2=nn.Linear(self.text_dim, fusion_dim)
                self.text_projection_1.requires_grad=True
                self.text_projection_2.requires_grad=True
            self.final_projection=nn.Linear(2*fusion_dim, fusion_dim)
            self.final_projection.requires_grad=True

        self.q_former_insertion=q_former_insertion
        self.num_heads_q_former=num_heads_q_former
        self.num_layers_q_former=num_layers_q_former
        self.num_queries_q_former=num_queries_q_former
        self.q_former_coef=q_former_coef
        #Q-Former insertion
        if self.q_former_insertion:
            logger_std.info("Applying Q-Former...") #input(batch , 1, vision_dim)
            self.q_former = SimpleQFormer(
                query_dim=fusion_dim,  #taille de sorties 
                img_feature_dim=self.vision_dim, 
                num_heads=self.num_heads_q_former, 
                num_layers=self.num_layers_q_former, 
                num_queries=self.num_queries_q_former)


        #regresssion head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim_2, 1),
        )
        # ACTIVATE PARAMETERS OF regression head
        if init_method=="xavier":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
        elif init_method=="kaiming":
            for m in self.regression_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    try:
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    except:
                        pass
    def forward(self, x):
        imgs = x["image"]  # [batch, 3, H, W]

        #recuperation des features de l'image
        feats = self.backbone.forward_features(imgs)  # [batch, vision_dim]
        cls_token = feats["x_norm_clstoken"].unsqueeze(1) # (B,1,D)
        patch_tokens = feats["x_norm_patchtokens"]        # (B,N,D)
        out_image_1 = torch.cat([cls_token, patch_tokens], dim=1) # (B,1+N,D)
        device = out_image_1.device
        # out_image_1 = feats #[batch, vision_dim]
        if self.q_former_insertion:
            # out_image_1 = out_image_1.unsqueeze(1) #(batch, 1, vision_dim)
            out_image_1 = self.q_former(out_image_1) #(batch, n_queries, fusion_dim)
            out_image_1 = out_image_1.mean(dim=1) #(batch, fusion_dim)
        else : 
            out_image_1 = out_image_1.mean(dim=1)
            out_image_1 = self.vision_projection(out_image_1) #(batch, fusion_dim)
        if self.vision_classification_model : 
            out_image_2 = self.backbone_2(imgs) #[batch, hidden_dim_2]
            out_image_2 = self.backbone_2_vision_projection(out_image_2) #[batch, fusion_dim]
            out_image = self.coef_image_1*out_image_1 + (1-self.coef_image_1)*out_image_2
        else : 
            out_image = out_image_1 #(batch, fusion_dim)
        # out_image = out_image_1 #(batch, fusion_dim)
        #recuperation des données textuelles
        encoded_1 = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        
        encoded_2 = self.tokenizer(
            x["title"], padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        out_text_1 = self.text_encoder_1(**encoded_1)
        out_text_2 = self.text_encoder_2(**encoded_2)
        input_mask_expanded = encoded_1["attention_mask"].unsqueeze(-1).expand(out_text_1.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_1.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_1 = t_feat
        input_mask_expanded = encoded_2["attention_mask"].unsqueeze(-1).expand(out_text_2.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out_text_2.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        out_text_2 = t_feat
        out_text_1 = self.text_projection_1(out_text_1) #(batch, fusion_dim)
        out_text_2 = self.text_projection_2(out_text_2) #(batch, fusion_dim)
        out_text = self.coef_1*out_text_1 + self.coef_2*out_text_2
        # out_text = out_text_2

        #fusion des features
        if self.use_concatenation:
            out = self.final_projection(torch.cat([out_image, out_text], dim=1))
        else:
            out = self.vision_coef*out_image+(1-self.vision_coef)*out_text
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

