import logging
import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_std = logging.getLogger(__name__)

# ----- CSV Processing -----
def process_youtube_csv(csv_path):
    df = pd.read_csv(csv_path)
    view_thresholds = [0, 1000, 10000, 100000, 1000000, float('inf')]
    labels = ["Hidden Gems", "Rising Stars", "Solid Performers", "Viral Hits", "Mega Blockbusters"]
    def assign_view_class(views):
        for i in range(len(view_thresholds) - 1):
            if view_thresholds[i] <= views < view_thresholds[i+1]:
                return labels[i]
        return labels[-1]
    df['view_classes'] = df['views'].apply(assign_view_class)
    if 'channel' in df.columns:
        unique_channels = df['channel'].unique()
        channel_mapping = {channel: f"channel{i+1}" for i, channel in enumerate(unique_channels)}
        df['channel_real_name'] = df['channel'].map(channel_mapping)
    return df


# ----- Dataset -----
class EmbedderTrainingDataset(Dataset):
    def __init__(self, tokenizer, csv_path, max_length=32):
        self.dataframe = process_youtube_csv(csv_path)
        self.encodings = tokenizer(
            list(self.dataframe["title"]), 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        unique_classes = sorted(self.dataframe["view_classes"].unique())
        self.class2idx = {c: i for i, c in enumerate(unique_classes)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}
        self.labels = torch.tensor([self.class2idx[c] for c in self.dataframe["view_classes"]], dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["title"] = self.dataframe["title"].iloc[idx]
        item["label_str"] = self.dataframe["view_classes"].iloc[idx]
        return item

def train_bert_tiny(train_set, val_set, tokenizer_name, model_name, device, epochs=5):
    """
    Entrainement du model de LLM bert tiny
    """
    criterion = nn.CrossEntropyLoss()
    model = MyLLM(tokenizer_name, model_name, num_classes=len(train_set.class2idx), device=device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    logger_std.info(f"Entrainement du model {model_name} sur {epochs} epochs")
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for batch in pbar:
            # batch['title'] = liste de strings
            input_titles = batch['prompt_resume']
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            output = model(input_titles)  # le model doit tokeniser en interne
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        logger_std.info(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_titles = batch['prompt_resume']
            labels = batch['labels'].to(device)
            output = model(input_titles)
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_accuracy = correct / total if total > 0 else 0
    logger_std.info(f"Validation accuracy: {val_accuracy:.4f}")

    # Sauvegarde du modèle au format pytorch classique
    save_name = f"train_{model_name}_{epochs}"
    model.text_encoder.push_to_hub(
    f"{save_name}_for_prompt_resume", 
    exist_ok=True, 
    )
    logger_std.info(f"Model {model_name} saved on huggingface")

    return total_loss / len(train_loader), save_name


#--------MODEL DE BERT TINY--------
class MyLLM(nn.Module):
    def __init__(self, tokenizer_name, base_model_path, num_classes, device):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        hidden_size = self.text_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.device = device
    def forward(self, titles):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # Tokenize in forward!
        encoded = self.tokenizer(
            titles, 
            padding=True, truncation=True, 
            return_tensors="pt",
            max_length=64
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return self.classifier(pooled)
# ----- Model -----
class MyCustomLLM(nn.Module):
    def __init__(self, base_model_path, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_path)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return self.classifier(pooled)

# ----- Training with tqdm -----
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(loader)

# ----- Prediction for N data points -----
def predict_on_dataset_samples(model, dataset, n=10):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            batch = dataset[i]
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            true_idx = batch['labels'].item()
            true_str = batch['label_str']
            title = batch['title']
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_str = dataset.idx2class[pred_idx]
            results.append({
                'title': title,
                'true_str': true_str,
                'pred_str': pred_str,
                'true_idx': true_idx,
                'pred_idx': pred_idx,
                'probs': probs[0].cpu().tolist()
            })
    return results

def print_sample_predictions(results, title="Sample predictions"):
    print(f"\n--- {title} ---")
    for r in results:
        print(
            f"\"{r['title']}\"\n  Vrai label: {r['true_str']}\n  Prédit: {r['pred_str']} ({', '.join([f'{p:.2f}' for p in r['probs']])})\n"
        )

def compute_accuracy_on_results(results):
    correct = sum(1 for r in results if r['true_idx'] == r['pred_idx'])
    return correct / len(results) if len(results) else 0.0

if __name__ == "__main__":
    pretrained_model = "distilbert-base-uncased"
    # pretrained_model = "prajjwal1/bert-tiny"
    tokenizer_model_path = "distilbert-base-uncased"
    num_classes = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = "/users/eleves-b/2023/sylvain.dehayem-kenfouo/projet_final_modal/dataset/train_val.csv"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    dataset = ToyDataset(tokenizer, csv_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MyCustomLLM(pretrained_model, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ----- Print 10 predictions BEFORE training -----
    results_before = predict_on_dataset_samples(model, dataset, n=10)
    print_sample_predictions(results_before, "Prédictions avant entraînement")
    acc_before = compute_accuracy_on_results(results_before)
    print(f"Accuracy sur les 10 premiers exemples avant entraînement: {acc_before*100:.2f}%\n")

    # ----- Training -----
    for epoch in range(50):
        loss = train_one_epoch(model, loader, optimizer, criterion)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    # ----- Print 10 predictions AFTER training -----
    results_after = predict_on_dataset_samples(model, dataset, n=10)
    print_sample_predictions(results_after, "Prédictions après entraînement")
    acc_after = compute_accuracy_on_results(results_after)
    print(f"Accuracy sur les 10 premiers exemples après entraînement: {acc_after*100:.2f}%\n")

    #save the model on my account 
    model.backbone.push_to_hub(
    "distilbert_embedder_train_best", 
    # organization="embedding-data",
    # train_datasets=["embedding-data/QQP_triplets"],
    exist_ok=True, 
    )