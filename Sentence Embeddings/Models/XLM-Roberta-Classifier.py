#%pip install sentencepiece


import matplotlib.pyplot as plt                    
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class Config:
    def __init__(self, p):
        self.learning_rate = p['learning_rate']
        self.epoch = p['epoch']
        self.batch_size = p['batch_size']
        self.max_len = p['max_len']
        self.model_save_path = p['model_save_path']
        self.warmup_rate = p['warmup_rate']
        self.weight_decay = p['weight_decay']
        self.model_pretrain_dir = p['model_pretrain_dir']
        self.training_set_path = p['training_set_path']
        self.testing_set_path = p['testing_set_path']
        self.seed = p['seed']

class MMClassifier(nn.Module):
    def __init__(self, model_path):
        super(MMClassifier, self).__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_output))
        return logits

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_pretrain_dir)
        set_seed(config.seed)

    def dataset(self, path):
        input_ids, attention_masks, labels = [], [], []
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            enc = self.tokenizer(row['text1'], row['text2'], padding='max_length', truncation=True,
                                 max_length=self.config.max_len, return_tensors='pt')
            input_ids.append(enc['input_ids'].squeeze(0))
            attention_masks.append(enc['attention_mask'].squeeze(0))
            label = int(round(float(row['Overall']))) - 1  # 1–4 → 0–3
            labels.append(label)
        return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels)

    def data_loader(self, ids, masks, labels, shuffle=True):
        return DataLoader(TensorDataset(ids, masks, labels),
                          batch_size=self.config.batch_size, shuffle=shuffle)

    def evaluate(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, att, y in loader:
                ids, att, y = ids.to(self.device), att.to(self.device), y.to(self.device)
                logits = model(ids, att)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

    def train(self):
        # 1) DataLoader
        ids, masks, labels = self.dataset(self.config.training_set_path)
        train_loader      = self.data_loader(ids, masks, labels)

        dev_ids, dev_masks, dev_labels = self.dataset(self.config.testing_set_path)
        dev_loader     = self.data_loader(dev_ids, dev_masks, dev_labels, shuffle=False)

        # 2) Model, Optimizer, Scheduler, Loss
        model = MMClassifier(self.config.model_pretrain_dir).to(self.device)
        optimizer = AdamW(model.parameters(),
                          lr=self.config.learning_rate,
                          weight_decay=self.config.weight_decay)

        total_steps = len(train_loader) * self.config.epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.warmup_rate * total_steps),
            num_training_steps=total_steps
        )

        criterion = nn.CrossEntropyLoss()

        # 3) Listen für Kurven
        train_losses, train_accs, dev_accs = [], [], []
        best_acc = 0.0

        # 4) Training über Epochen
        for epoch in range(self.config.epoch):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for ids, att, y in train_loader:
                ids, att, y = ids.to(self.device), att.to(self.device), y.to(self.device)

                # Forward
                logits = model(ids, att)
                loss   = criterion(logits, y)

                # Backward + Step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Statistiken
                running_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct  += (preds == y).sum().item()
                total    += y.size(0)

            # Ende Epoche: Metriken berechnen
            train_loss = running_loss / len(train_loader)
            train_acc  = correct / total
            dev_acc    = self.evaluate(model, dev_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            dev_accs.append(dev_acc)

            print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | "
                  f"Dev Acc: {dev_acc:.4f} | Loss: {train_loss:.4f}")

            # Bestes Modell speichern
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(model.state_dict(), self.config.model_save_path)

        print(f"✅ Training finished. Best Dev Acc: {best_acc:.4f}")

        # 5) Nach Training: Kurven plotten
        epochs = list(range(1, self.config.epoch + 1))
        plt.figure(figsize=(10,4))

        # Loss-Kurve
        plt.subplot(1,2,1)
        plt.plot(epochs, train_losses, marker='o')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Accuracy-Kurve
        plt.subplot(1,2,2)
        plt.plot(epochs, train_accs, label="Train", marker='o')
        plt.plot(epochs, dev_accs,   label="Dev",   marker='x')
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()




params = {
  "learning_rate": 2e-5,
  "epoch": 5,
  "batch_size": 8,
  "max_len": 512,
  "model_save_path": "best_classifier.pth",
  "warmup_rate": 0.1,
  "weight_decay": 0.01,
  "model_pretrain_dir": "xlm-roberta-base",
  "training_set_path": "training_set.csv",
  "testing_set_path": "testing_set.csv",
  "seed": 42
}

config = Config(params)
trainer = Trainer(config)
trainer.train()