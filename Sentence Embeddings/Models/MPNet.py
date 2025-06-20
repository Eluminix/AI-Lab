import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Lade SentenceTransformer Modell
model_name = "paraphrase-multilingual-MPNet-base-v2"
encoder = SentenceTransformer(model_name)

# 2. Klassifikationskopf definieren
class Classifier(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=4):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# 3. Daten vorbereiten (CSV laden & embeddings erzeugen)
def load_data(path):
    df = pd.read_csv(path)
    sentences = [f"{a} [SEP] {b}" for a, b in zip(df['text1'], df['text2'])]
    embeddings = encoder.encode(sentences, convert_to_tensor=True)
    labels = torch.tensor([int(round(score)) - 1 for score in df['Overall']])
    return embeddings, labels

# 4. Trainings-Setup
train_embeddings, train_labels = load_data("training_set.csv")
test_embeddings, test_labels = load_data("testing_set.csv")

train_loader = DataLoader(TensorDataset(train_embeddings, train_labels), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(test_embeddings, test_labels), batch_size=16)

classifier = Classifier().to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
device = next(classifier.parameters()).device

# 5. Training
epochs = 5
for epoch in range(epochs):
    classifier.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = classifier(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Accuracy={acc:.4f}")

# 6. Evaluation
classifier.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = classifier(x)
        preds = torch.argmax(logits, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["1", "2", "3", "4"]))

# 7. Konfusionsmatrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()