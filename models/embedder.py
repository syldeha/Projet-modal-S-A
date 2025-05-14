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


def train_bert_tiny(train_set, val_set, tokenizer_name, model_name, device, epochs=5):
    """
    Entrainement du model de LLM bert tiny
    """
    import numpy as np
    # 1. Calcul automatique des poids de classes à partir du train set
    labels_np = train_set.labels.numpy() if hasattr(train_set, "labels") else None
    class_counts = np.bincount(labels_np)
    class_weights = 1. / (class_counts + 1e-6)  # Evite division par zéro
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # (optionnel) normalisation
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 2. Plug les poids dans la loss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    model = MyLLM(tokenizer_name, model_name, num_classes=len(train_set.class2idx), device=device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    logger_std.info(f"Entrainement du model {model_name} sur {epochs} epochs")
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    best_val_acc = 0
    best_model=None
    for epoch in range(epochs):
        if epoch%10 == 0:
            #save the best model 
            save_name = f"train_bert_distill_base_{epoch}"
            if best_model is not None:
                best_model.text_encoder.push_to_hub(
                f"{save_name}", 
                exist_ok=True, 
                )
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
        # model.eval()
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
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model = model
    # Sauvegarde du modèle au format pytorch classique
    save_name = f"train_{model_name}_{epochs}"
    best_model.text_encoder.push_to_hub(
    f"{save_name}", 
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
        # self.linear = nn.Linear(hidden_size, hidden_size)
        self.device = device
        for param in self.text_encoder.parameters():
            param.requires_grad = True
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
        # pooled = self.linear(pooled)
        return self.classifier(pooled)
