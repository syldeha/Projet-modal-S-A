import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import wandb
import hydra
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import math
class CrossModalAttention(nn.Module):
    """Cross-attention entre vision et texte avec regularisation"""
    def __init__(self, vision_dim, text_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projections pour vision et texte
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention layers
        self.vision_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_vision_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.ln_vision = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ffn_vision = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, vision_features, text_features):
        # Projection vers l'espace commun
        v_proj = self.vision_proj(vision_features)  # [B, 1, hidden_dim]
        t_proj = self.text_proj(text_features)      # [B, seq_len, hidden_dim]
        
        # Cross-attention: vision attend au texte
        v_attended, _ = self.vision_to_text_attn(v_proj, t_proj, t_proj)
        v_attended = self.ln_vision(v_proj + v_attended)
        v_out = v_attended + self.ffn_vision(v_attended)
        
        # Cross-attention: texte attend à la vision
        t_attended, _ = self.text_to_vision_attn(t_proj, v_proj, v_proj)
        t_attended = self.ln_text(t_proj + t_attended)
        t_out = t_attended + self.ffn_text(t_attended)
        
        return v_out.squeeze(1), t_out.mean(dim=1)  # [B, hidden_dim], [B, hidden_dim]

class AdaptiveModalityWeighting(nn.Module):
    """Pondération adaptative des modalités basée sur l'incertitude"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.vision_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.text_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vision_feat, text_feat):
        # Calcul des incertitudes (plus l'incertitude est haute, moins le poids)
        vision_uncertainty = self.vision_uncertainty(vision_feat)
        text_uncertainty = self.text_uncertainty(text_feat)
        
        # Poids inversement proportionnels à l'incertitude
        vision_weight = 1.0 / (1.0 + vision_uncertainty)
        text_weight = 1.0 / (1.0 + text_uncertainty)
        
        # Normalisation
        total_weight = vision_weight + text_weight
        vision_weight = vision_weight / total_weight
        text_weight = text_weight / total_weight
        
        return vision_weight, text_weight

class ImprovedMultiModalModel(nn.Module):
    def __init__(self, 
                 vision_model="efficientnet_b3",
                 hidden_dim=512,
                 num_heads=8,
                 dropout=0.3,
                 lora_r=8,
                 lora_alpha=16):
        super().__init__()
        
        # Vision backbone (plus léger)
        self.vision_backbone = timm.create_model(vision_model, pretrained=True)
        if hasattr(self.vision_backbone, 'fc'):
            self.vision_dim = self.vision_backbone.fc.in_features
            self.vision_backbone.fc = nn.Identity()
        elif hasattr(self.vision_backbone, 'classifier'):
            self.vision_dim = self.vision_backbone.classifier.in_features
            self.vision_backbone.classifier = nn.Identity()
        elif hasattr(self.vision_backbone, "head"):
            self.vision_dim = self.vision_backbone.head.in_features
            self.vision_backbone.head = nn.Identity()
            
        # Gel des paramètres vision
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # Text backbone avec LoRA plus léger
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Configuration LoRA plus conservative
        lora_config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=dropout,
            bias="none",
            target_modules=["q_lin", "v_lin"]
        )
        self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        
        # Gel des paramètres non-LoRA
        for name, param in self.text_encoder.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
                
        # Cross-modal attention
        self.cross_modal_attn = CrossModalAttention(
            self.vision_dim, self.text_dim, hidden_dim, num_heads, dropout
        )
        
        # Pondération adaptative
        self.adaptive_weighting = AdaptiveModalityWeighting(hidden_dim)
        
        # Prédiction finale avec activation adaptée pour les vues
        self.predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Assure des sorties strictement positives
        )
        
        # Initialisation des poids
        self._init_weights()
        
    def _init_weights(self):
        """Initialisation conservative des poids"""
        for module in [self.cross_modal_attn, self.adaptive_weighting, self.predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
    
    def forward(self, batch):
        """Forward compatible avec votre format de batch"""
        # Extraction des données du batch
        images = batch["image"]
        
        # Pour le texte, adapter selon votre format exact
        # Supposons que vous avez des clés 'input_ids' et 'attention_mask'
        text_inputs = self.tokenizer(
                    batch["prompt_resume"], 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512
                )
        input_ids = text_inputs["input_ids"].to(images.device)
        attention_mask = text_inputs["attention_mask"].to(images.device)

        # Extraction des features vision
        with torch.no_grad():
            vision_features = self.vision_backbone(images)
        vision_features = vision_features.unsqueeze(1)
        
        # Extraction des features texte
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state
        
        # Cross-modal attention
        vision_attended, text_attended = self.cross_modal_attn(
            vision_features, text_features
        )
        
        # Pondération adaptative
        vision_weight, text_weight = self.adaptive_weighting(
            vision_attended, text_attended
        )
        
        # Fusion pondérée
        fused_features = (vision_weight * vision_attended + 
                         text_weight * text_attended)
        
        # Prédiction
        output = self.predictor(fused_features)
        
        return output.squeeze(-1)
