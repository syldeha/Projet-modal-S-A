import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from models.model_shap import ModelShap
from data.dataset import Dataset
import torchvision.transforms as transforms
import random

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Charger le modèle
model = ModelShap(
    num_channels=100,
    year_bin_dim=4,
    tabular_dim=6,
    emb_dim=16,
    img_dim=50,
    text_dim=384,
    hidden_dim=256
)
model.load_state_dict(torch.load("checkpoints/model_shap_2025-05-28_12-04-17_val2021_1_best.pt", map_location=device))
model.eval()
model.to(device)

# Wrapper simplifié pour le modèle
class SimpleModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, batch):
        out = self.model(batch)
        return out[0] if isinstance(out, tuple) else out

model_wrapper = SimpleModelWrapper(model)

# Transformations et dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = Dataset(
    "dataset",
    split="train_val",
    transforms=transform,
    metadata=["title", "description"],
    val_years=[2022, 2023],
    train_or_val_or_test="train"
)

def simple_collate(batch_list):
    out = {}
    out["image"] = torch.stack([x["image"] for x in batch_list])
    out["channel_id"] = torch.tensor([x["channel_id"] for x in batch_list])
    
    year_list = []
    for x in batch_list:
        year_val = x["year_new"]
        if isinstance(year_val, torch.Tensor):
            if year_val.numel() == 4:
                year_list.append(year_val.tolist())
            else:
                year_list.append([float(year_val.item())] * 4)
        else:
            year_list.append([float(year_val)] * 4)
    out["year_new"] = torch.tensor(year_list, dtype=torch.float32)
    
    out["day_sin"] = torch.tensor([x["day_sin"] for x in batch_list])
    out["day_cos"] = torch.tensor([x["day_cos"] for x in batch_list])
    out["month_sin"] = torch.tensor([x["month_sin"] for x in batch_list])
    out["month_cos"] = torch.tensor([x["month_cos"] for x in batch_list])
    out["hour_sin"] = torch.tensor([x["hour_sin"] for x in batch_list])
    out["hour_cos"] = torch.tensor([x["hour_cos"] for x in batch_list])
    out["title"] = [x["title"] for x in batch_list]
    out["description"] = [x["description"] for x in batch_list]
    return out

# Échantillonnage
n_samples = 5
samples = [dataset[i] for i in random.sample(range(len(dataset)), n_samples)]
background_batch = simple_collate([samples[0]])
explain_batch = simple_collate([samples[1]])

# Déplacer vers le device
for k, v in background_batch.items():
    if isinstance(v, torch.Tensor):
        background_batch[k] = v.to(device)
for k, v in explain_batch.items():
    if isinstance(v, torch.Tensor):
        explain_batch[k] = v.to(device)

# Wrapper SHAP pour les variables temporelles
class TabularOnlyWrapper:
    def __init__(self, model, device, template_batch):
        self.model = model
        self.device = device
        self.template = template_batch
        self.features = ["day_sin", "day_cos", "month_sin", "month_cos", "hour_sin", "hour_cos"]
        
    def extract_features(self, batch):
        day_sin = batch["day_sin"].cpu().numpy().reshape(-1, 1)
        day_cos = batch["day_cos"].cpu().numpy().reshape(-1, 1)
        month_sin = batch["month_sin"].cpu().numpy().reshape(-1, 1)
        month_cos = batch["month_cos"].cpu().numpy().reshape(-1, 1)
        hour_sin = batch["hour_sin"].cpu().numpy().reshape(-1, 1)
        hour_cos = batch["hour_cos"].cpu().numpy().reshape(-1, 1)
        return np.concatenate([day_sin, day_cos, month_sin, month_cos, hour_sin, hour_cos], axis=1)
        
    def __call__(self, X):
        batch_size = X.shape[0]
        batch = {}
        
        for k, v in self.template.items():
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 1:
                    batch[k] = v[:1].repeat(batch_size)
                elif len(v.shape) == 2:
                    batch[k] = v[:1].repeat(batch_size, 1)
                else:
                    batch[k] = v[:1].repeat(batch_size, *([1] * (len(v.shape) - 1)))
            elif isinstance(v, list):
                batch[k] = [v[0]] * batch_size
            else:
                batch[k] = [v] * batch_size
        
        batch["day_sin"] = torch.tensor(X[:, 0], dtype=torch.float32).to(self.device)
        batch["day_cos"] = torch.tensor(X[:, 1], dtype=torch.float32).to(self.device)
        batch["month_sin"] = torch.tensor(X[:, 2], dtype=torch.float32).to(self.device)
        batch["month_cos"] = torch.tensor(X[:, 3], dtype=torch.float32).to(self.device)
        batch["hour_sin"] = torch.tensor(X[:, 4], dtype=torch.float32).to(self.device)
        batch["hour_cos"] = torch.tensor(X[:, 5], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            return self.model(batch).cpu().numpy().flatten()

# Wrapper SHAP pour le texte
class TitleDescWrapper:
    def __init__(self, model, device, template_batch):
        self.model = model
        self.device = device
        self.template = template_batch
        self.features = ["title", "description"]

    def extract_features(self, batch):
        return np.array([[t, d] for t, d in zip(batch["title"], batch["description"])])

    def __call__(self, X):
        batch_size = X.shape[0]
        batch = {}
        for k, v in self.template.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[:1].repeat(batch_size, *([1] * (len(v.shape) - 1)))
            elif isinstance(v, list):
                batch[k] = [v[0]] * batch_size
            else:
                batch[k] = [v] * batch_size
        batch["title"] = list(X[:, 0])
        batch["description"] = list(X[:, 1])
        with torch.no_grad():
            return self.model(batch).cpu().numpy().flatten()

# Analyse SHAP pour les variables temporelles
print("Analyse SHAP des variables temporelles...")
tabular_wrapper = TabularOnlyWrapper(model_wrapper, device, background_batch)
X_background_tab = tabular_wrapper.extract_features(background_batch)
X_explain_tab = tabular_wrapper.extract_features(explain_batch)

explainer_tab = shap.KernelExplainer(tabular_wrapper, X_background_tab)
shap_values_tab = explainer_tab(X_explain_tab)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_tab.values, X_explain_tab, 
                 feature_names=tabular_wrapper.features, 
                 show=False)
plt.title("Influence des Variables Temporelles (SHAP)")
plt.tight_layout()
plt.savefig("shap_summary_temporal.png", dpi=150, bbox_inches='tight')
plt.show()

# Analyse SHAP pour le texte
print("Analyse SHAP du texte...")
text_wrapper = TitleDescWrapper(model_wrapper, device, background_batch)
X_background_text = text_wrapper.extract_features(background_batch)
X_explain_text = text_wrapper.extract_features(explain_batch)

explainer_text = shap.KernelExplainer(text_wrapper, X_background_text)
shap_values_text = explainer_text(X_explain_text)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_text.values, X_explain_text, 
                 feature_names=text_wrapper.features, 
                 show=False)
plt.title("Influence du Texte (SHAP)")
plt.tight_layout()
plt.savefig("shap_summary_text.png", dpi=150, bbox_inches='tight')
plt.show()




