import torch
from torch import nn
from transformers import BlipModel, BlipProcessor
from peft import get_peft_model, LoraConfig, TaskType
import os


class BlipFinetune(nn.Module):
    def __init__(self, frozen=False, use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        # Chargement du modÃ¨le BLIP de Salesforce
        self.blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        

        
        # Dimensions des embeddings
        self.embed_dim = self.blip_model.config.projection_dim
        
        # Geler les paramÃ¨tres si nÃ©cessaire
        if frozen:
            for param in self.blip_model.parameters():
                param.requires_grad = False
        
        # Tete de regression
        self.fusion_dim = self.embed_dim * 2
        # self.regression_head = nn.Sequential(
        #     nn.Linear(self.fusion_dim, 1024),  # *2 car on concatÃ¨ne image et texte
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 512),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256), 
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 1),
        #     nn.ReLU(),
        # )
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Softplus()
            )
        if use_lora:
            self.config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["query","value"],
        )

            self.blip_model = get_peft_model(self.blip_model, self.config)

    def forward(self, x):
        # Traitement de l'image et du texte avec BLIP
        inputs = self.processor(
            images=x["image"],
            text=x["prompt_resume"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(x["image"].device) for k, v in inputs.items()}
        
        # Passage dans le modÃ¨le BLIP
        outputs = self.blip_model(**inputs)
        
        # Extraire les embeddings d'image et de texte sÃ©parÃ©ment
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        # print("image_embeds.shape",image_embeds.shape)
        # print("text_embeds.shape",text_embeds.shape)
        # print("self.image_dim",self.image_dim)
        # print("self.text_dim",self.text_dim)
        # print("self.embed_dim",self.blip_model.config.projection_dim)
        
        # ConcatÃ©ner les embeddings
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # Passage dans la tÃªte de rÃ©gression
        output = self.regression_head(combined_embeds)
        return output

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
            
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False
        