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
        val_loss = 0
        n_val = 0
        all_batch= []
        with torch.no_grad():
            for batch in val_loader:
                output=model(batch)
                # preds: List of strings, ['low', 'high', ...]
                val_loss += output.loss.item()*len(batch['prompt_resume'])
                n_val += len(batch['prompt_resume'])
                # correct += sum([p==r for p, r in zip(pred_indices, ref_indices)])
                # total += len(pred_indices)
                all_batch.append(batch)
        val_loss = val_loss / n_val if n_val > 0 else 0
        logger_std.info(f"Validation loss: {val_loss:.4f}")

        #affichage des 3 premières predictions
        last_batch = all_batch[0]
        prompts=last_batch['prompt_resume'][:3]
        true_labels=last_batch['label_str'][:3]
        # Prédiction par génération
        model.eval()
        with torch.no_grad():
            gen_ids = model.text_encoder.generate(
                **model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device),
                max_length=64
            )
            preds = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for i, (prompt, true_label, pred) in enumerate(zip(prompts, true_labels, preds)):
            print(f"\n----- EXEMPLE {i+1} -----")
            print("PROMPT:")
            print(prompt)
            print(f"GOLD LABEL: {true_label}")
            print(f"PREDICTED: {pred}")

    # Sauvegarde du modèle au format pytorch classique
    save_name = f"train_flant5_{epochs}"
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



