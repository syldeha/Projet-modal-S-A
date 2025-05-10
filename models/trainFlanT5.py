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
    labels = ["low", "medium", "high", "viral", "top"]
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



def train_flant5(train_set, val_set, tokenizer_name, model_name, device, epochs=5, learning_rate=1e-5):
    """
    Entrainement du model de LLM bert tiny
    """
    model = MyLLM(tokenizer_name, model_name, num_classes=len(train_set.class2idx), device=device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger_std.info(f"Entrainement du model {model_name} sur {epochs} epochs")
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            output = model(batch)  # return outputs.loss, outputs.logits
            loss = output.loss
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
                generated_ids = model(batch)
                preds=model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # preds: List of strings, ['low', 'high', ...]
                pred_indices = [class2idx[pred.lower().strip()] for pred in preds]
                ref_indices = [class2idx[label.lower().strip()] for label in batch['labels']]
                correct += sum([p==r for p, r in zip(pred_indices, ref_indices)])
                total += len(pred_indices)
        val_accuracy = correct / total if total > 0 else 0
        logger_std.info(f"Validation accuracy: {val_accuracy:.4f}")

    # Sauvegarde du modèle au format pytorch classique
    save_name = f"train_{model_name}_{epochs}"
    model.text_encoder.push_to_hub(
    f"{save_name}", 
    exist_ok=True, 
    )
    logger_std.info(f"Model {model_name} saved on huggingface")

    return total_loss / len(train_loader), save_name


from transformers import AutoModelForSeq2SeqLM , AutoTokenizer
#--------MODEL DE BERT TINY--------
class MyLLM(nn.Module):
    def __init__(self, tokenizer_name, base_model_path, num_classes, device):
        super().__init__()
        self.text_encoder = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.text_encoder=self.text_encoder.to(device)

    def forward(self, inputs):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # Tokenize in forward!
        encoded = self.tokenizer(
            inputs["prompt_resume"], 
            padding=True, truncation=True, 
            return_tensors="pt",
            max_length=256
        ).to(self.device)
        if "label_str" in inputs:
            #training  part 
            targets = inputs["label_str"]
            encoder_targets=self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt", max_length=10).to(self.device)
            outputs = self.text_encoder(
                **encoded,
                labels=encoder_targets["input_ids"]
            )
            return outputs   # outputs.loss, outputs.logits
        else:
            #inference part 
            generated_ids = self.text_encoder.generate(**encoded, max_length=10 , num_beams=5)
            return generated_ids






if __name__ == "__main__":
