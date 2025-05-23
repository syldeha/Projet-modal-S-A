import torch
import torch.nn as nn
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
def print_parameters_per_layer(model):
    """
    Affiche pour chaque couche du modèle :
    - son nom
    - sa classe (type)
    - la taille totale de ses paramètres
    - la taille des params entraînables
    """
    print(f"{'Layer':50} {'Type':25} {'Total Params':>14} {'Trainable Params':>18}")
    print('='*111)
    total = 0
    total_trainable = 0
    for name, module in model.named_modules():
        # On veut la somme des params du module (attention: pas du modèle global !)
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        trainable_count = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if param_count > 0:
            print(f"{name:50} {type(module).__name__:25} {param_count:14,} {trainable_count:18,}")
        total += param_count
        total_trainable += trainable_count
    print('='*111)
    print(f"{'Total':50} {'':25} {total:14,} {total_trainable:18,}")
    print()


class DynamicCNN_vision_backbone(nn.Module):
    def __init__(
        self,
        input_channels=3,
        n_layers=5,
        initial_num_filters=32,
        filter_increase=2,
        kernel_sizes=3,
        strides=1,
        paddings=1,
        use_batchnorm=True,
        use_dropout=True,
        dropout_p=0.25,
        use_maxpool=True,
        pool_every=2,  # maxpool every 'pool_every' conv layers
        regression_hidden_vision=256,
        regression_hidden2_vision=64,
        input_image_size=224,
    ):
        super().__init__()

        # Helper pour forcer int/tuple partout
        def process_param_list(param, n):
            if isinstance(param, (int, float)):
                return [int(param)] * n
            param = list(param)
            if len(param) < n:
                param += [param[-1]] * (n - len(param))
            return [tuple(p) if isinstance(p, (list, tuple)) else int(p) for p in param]

        self.n_layers = n_layers
        kernel_sizes = process_param_list(kernel_sizes, n_layers)
        strides = process_param_list(strides, n_layers)
        paddings = process_param_list(paddings, n_layers)

        layers = []
        in_c = input_channels
        out_c = initial_num_filters
        current_size = input_image_size

        for i in range(n_layers):
            ksize = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            # Forcer int ou tuple(2,) pour chaque param :
            if isinstance(ksize, list): ksize = tuple(ksize)
            if isinstance(stride, list): stride = tuple(stride)
            if isinstance(padding, list): padding = tuple(padding)

            layers.append(nn.Conv2d(
                in_c, out_c,
                kernel_size=ksize, stride=stride, padding=padding
            ))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout2d(dropout_p))
            # Optionally add pooling
            if use_maxpool and ((i + 1) % pool_every == 0 or i == n_layers - 1):
                layers.append(nn.MaxPool2d(2))
                current_size = current_size // 2
            in_c = out_c
            out_c = out_c * filter_increase
        # layers.append(nn.AdaptiveAvgPool2d(1))  # (C, 1, 1)
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        final_feature_dim = in_c * (current_size ** 2)
        # - DÉTERMINATION DYNAMIQUE de la sortie CNN à partir d'un dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_image_size, input_image_size)
            dummy_feat = self.cnn(dummy)
            flatten_dim = dummy_feat.shape[1]
            self.out_features = flatten_dim
            print(f"Flatten dimension: {flatten_dim}")
        layers.append(nn.Linear(flatten_dim, regression_hidden_vision))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm1d(regression_hidden_vision))
        layers.append(nn.Dropout(dropout_p))


        self.out_features = regression_hidden_vision
        self.cnn_head = nn.Sequential(*layers)
        self.regressor = nn.Sequential(
            nn.Linear(regression_hidden_vision, regression_hidden2_vision),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(regression_hidden2_vision, 1)
        )

        # Xavier init for all linear
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['image']
        # feats = self.cnn(x)  # x: (B, 3, H, W)
        feats = self.cnn_head(x)
        output = self.regressor(feats)
        return output.squeeze(1)

    # Load checkpoint
    def load_checkpoint(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
        import os
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint

            if load_full:
                self.load_state_dict(model_state_dict, strict=False)
                print(f"Loaded full model from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False

class CNN_with_bert(nn.Module):
    def __init__(self , 
        input_channels=3,
        n_layers=5,
        initial_num_filters=32,
        filter_increase=2,
        kernel_sizes=3,
        strides=1,
        paddings=1,
        use_batchnorm=True,
        use_dropout=True,
        dropout_p=0.25,
        use_maxpool=True,
        pool_every=2,  # maxpool every 'pool_every' conv layers
        regression_hidden_vision=128,
        regression_hidden2_vision=128,
        regression_hidden_head=128,
        regression_hidden2_head=128,
        input_image_size=224,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_concatenation=True,
        vision_coef=0.5,
        text="title", #use "title" or "prompt_resume"
        tokenizer_model_path="bert-base-uncased",
        pretrained_model="bert-base-uncased",
        load_backbone_checkpoint=None,
        token_classification=False,
        frozen_backbone=True,

    ):
        super().__init__()

        self.vision_backbone = DynamicCNN_vision_backbone(
            input_channels=input_channels,
            n_layers=n_layers,
            initial_num_filters=initial_num_filters,
            filter_increase=filter_increase,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_batchnorm=use_batchnorm,
            use_dropout=use_dropout,
            dropout_p=dropout_p,
            use_maxpool=use_maxpool,
            pool_every=pool_every,
            regression_hidden_vision=regression_hidden_vision,
            regression_hidden2_vision=regression_hidden2_vision,
            input_image_size=input_image_size,
        )
        if load_backbone_checkpoint:
            logger_std.info(f"Loading backbone checkpoint from {load_backbone_checkpoint}")
            self.vision_backbone.load_checkpoint(load_backbone_checkpoint)
        #recuperation de la dimension de la sortie de la partie CNN du model
        self.vision_dim=self.vision_backbone.out_features
        #recuperation de la partie CNN du model
        self.vision_backbone=self.vision_backbone.cnn_head
        #freeze the backbone if frozen_backbone is True
        if frozen_backbone and load_backbone_checkpoint:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.vision_backbone.parameters():
                param.requires_grad = True

        #SET the text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self.text_dim = self.text_encoder.config.hidden_size
        self.text_input=text
        self.token_classification=token_classification
        self.use_concatenation=use_concatenation
        self.vision_coef=vision_coef
        #SET the lora
        if use_lora:
            logger_std.info("Applying LoRA to text backbone...")
            # For BERT-like: target_modules are ['query', 'value']
            txt_lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=["query", "key", "value"]
            )
            self.text_encoder = get_peft_model(self.text_encoder, txt_lora_config)
            self.text_encoder.print_trainable_parameters()
            # Freeze ALL parameters in base backbone BERT
            for param in self.text_encoder.base_model.parameters():
                param.requires_grad = False

        # fusion dimension
        fusion_dim = self.text_dim
        self.fusion_dim = fusion_dim
        self.vis_proj = nn.Linear(self.vision_dim, fusion_dim)
        self.txt_proj = nn.Identity()
        # Fusion complète par concaténation, puis projection
        self.fusion_proj = nn.Linear(2*fusion_dim, fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, regression_hidden_head),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(regression_hidden_head),
            nn.Dropout(dropout_p),

            nn.Linear(regression_hidden_head, regression_hidden2_head),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(regression_hidden2_head, 1)
        )
        # Xavier init for all linear
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        #vision backbone
        feats = self.vision_backbone(x["image"])
        v_feat = self.vis_proj(feats)
        # print(v_feat.shape)
        device = v_feat.device

        #text encoder
        encoded = self.tokenizer(
            x[f"{self.text_input}"], padding=True, truncation=True, return_tensors='pt'
        )
        if self.token_classification : 
            encoded = {k: v.to(device) for k, v in encoded.items()}
            encoded = self.text_encoder(**encoded)
            t_feat = self.txt_proj(encoded.last_hidden_state[:, 0, :])
        else:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = self.text_encoder(**encoded)
            input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(out.last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
            t_feat = self.txt_proj(t_feat) # [batch, self.fusion_dim]

        # Harmonisation of vfeat and tfeat
        if self.use_concatenation:
            feats = torch.cat([v_feat, t_feat], dim=1) #(B,2*fusion_dim)
            feats = self.fusion_proj(feats) #(B,fusion_dim)
        else: #do a pondered sum
            feats = v_feat * self.vision_coef + t_feat * (1 - self.vision_coef) #(B,fusion_dim)
        
        output = self.regressor(feats)
        return output.squeeze(1)

    # Load checkpoint
    def load_checkpoint(self, checkpoint_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), load_full=True):
        import os
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint

            if load_full:
                self.load_state_dict(model_state_dict, strict=False)
                print(f"Loaded full model from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint file {checkpoint_path} not found.")
            return False

if __name__ == "__main__":
    model = CNN_with_bert()
    print_parameters_per_layer(model)