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





class DynamicCNNRegressor(nn.Module):
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
        regression_hidden=256,
        regression_hidden2=64,
        input_image_size=224,
        pretrained_model="bert-base-uncased", 
        tokenizer_model_path="bert-base-uncased", 
        fusion_dropout=0.2,
        lora_vis=True,         # LoRA sur vision backbone
        lora_txt=True,         # LoRA sur texte backbone
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
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
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        final_feature_dim = in_c * (current_size ** 2)
        # - DÉTERMINATION DYNAMIQUE de la sortie CNN à partir d'un dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_image_size, input_image_size)
            dummy_feat = self.cnn(dummy)
            vision_dim = dummy_feat.shape[1]
        self.vision_dim = vision_dim
        # ---- Text backbone ----
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self.text_dim = self.text_encoder.config.hidden_size
        
        logger_std.info("Applying LoRA to text backbone...")
        # For BERT-like: target_modules are ['query', 'value']
        txt_lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
            target_modules=["query", "key", "value"]
        )
        self.text_encoder = get_peft_model(self.text_encoder, txt_lora_config)
        self.text_encoder.print_trainable_parameters()

        # Harmonisation des dims (projection si besoin)
        fusion_dim = min(self.text_dim , self.vision_dim)
        self.fusion_dim = fusion_dim
        if self.vision_dim != fusion_dim:
            self.vis_proj = nn.Linear(self.vision_dim, fusion_dim)
        else:
            self.vis_proj = nn.Identity()
        if self.text_dim != fusion_dim:
            self.txt_proj = nn.Linear(self.text_dim, fusion_dim)
        else:
            self.txt_proj = nn.Identity()
        # Fusion complète par concaténation, puis projection
        self.fusion_proj = nn.Linear(2*fusion_dim, fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, regression_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(regression_hidden),
            nn.Dropout(dropout_p),

            nn.Linear(regression_hidden, regression_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(regression_hidden2, 1)
        )
        # Xavier init for all linear
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = self.cnn(x['image'])  #(B,vision_dim)
        v_feat = self.vis_proj(feats) #(B,fusion_dim)

        #text encoder
        device = v_feat.device
        encoded = self.tokenizer(
            x["prompt_resume"], padding=True, truncation=True, return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.text_encoder(**encoded)
        input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(out.last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        t_feat = sum_embeddings / sum_mask.clamp(min=1e-9) # [batch, self.text_dim]
        t_feat = self.txt_proj(t_feat) # [batch, self.fusion_dim]

        # Harmonisation des dims (projection si besoin)
        feats = torch.cat([v_feat, t_feat], dim=1) #(B,2*fusion_dim)
        feats = self.fusion_proj(feats) #(B,fusion_dim)
        output = self.regressor(feats)
        return output.squeeze(1)

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


# ----- Exemple d'utilisation -----
if __name__ == "__main__":
    # Exemples :
    # 1. Kernel/sizes/strides en INT
    print("test")
    model = DynamicCNNRegressor(
        input_channels=3,
        n_layers=4,
        initial_num_filters=16,
        filter_increase=2,
        kernel_sizes=3,
        strides=1,
        paddings=1,
        use_batchnorm=True,
        use_dropout=True,
        pool_every=2,
        dropout_p=0.3,
        pretrained_model="bert-base-uncased", 
        tokenizer_model_path="bert-base-uncased", 
        fusion_dropout=0.2,
        lora_vis=True,         # LoRA sur vision backbone
        lora_txt=True,         # LoRA sur texte backbone
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    x = torch.randn(8, 3, 224, 224)
    out = model(x)
    print(out.shape)  # (8,)

    # 2. Kernel/sizes/strides en liste hétérogène
    model2 = DynamicCNNRegressor(
        input_channels=3,
        n_layers=4,
        initial_num_filters=16,
        filter_increase=2,
        kernel_sizes=[3, 3, 5, 3],
        strides=[1, 2, 1, 1],
        paddings=[1, 1, 2, 1],
        use_batchnorm=True,
        use_dropout=True,
        pool_every=2,
        dropout_p=0.3
    )
    out2 = model2(x)
    print(out2.shape)  # (8,)