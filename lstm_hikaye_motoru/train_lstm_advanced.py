# Bu betikte, duygu koşullu metin üretimi için BiLSTM tabanlı bir model eğitilmektedir.
# Eğitim sürecinde metinler tokenize edilmekte, eğitim/validasyon ayrımı yapılmakta
# ve modelin başarımı (loss, accuracy, F1-score) her epoch sonunda kaydedilmektedir.

import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Parametreler tanımlanmıştır. 
DATA_PATH = "./merged_lstm_dataset_conditional.csv"  # Girdi veri seti
TOKENIZER_PATH = "./tokenizer.pickle"                # Tokenizer dosyası
MODEL_PATH = "./best_model_checkpoint.pth"           # En iyi modelin kaydedileceği yol
LOG_PATH = "./loss_log_lstm.csv"                     # Eğitim kayıtlarının tutulacağı CSV
BATCH_SIZE = 128
EPOCHS = 10
MAX_SEQUENCE_LEN = 100
EMBED_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer yüklenmiştir.
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

# Veri hazırlanmakta ve diziler oluşturulmaktadır. 
def prepare_sequences(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN, padding='post')
    X = torch.tensor(padded[:, :-1], dtype=torch.long)  #Girdi dizisi(n-1)
    y = torch.tensor(padded[:, 1:], dtype=torch.long)   #Hedef dizisi(n-shift)
    return X, y

df = pd.read_csv(DATA_PATH).sample(n=10000, random_state=42)   # Rastgele örneklenmiş 10.000 satır
texts = df["text"].astype(str).tolist()
split_idx = int(0.9 * len(texts))                              # %90 eğitim, %10 doğrulama
train_texts, val_texts = texts[:split_idx], texts[split_idx:]
train_X, train_y = prepare_sequences(train_texts)
val_X, val_y = prepare_sequences(val_texts)


# PyTorch Dataset tanımı yapılmıştır.
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader'lar oluşturulmuştur.
train_loader = DataLoader(TextDataset(train_X, train_y),
                           batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=True)
val_loader = DataLoader(TextDataset(val_X, val_y),
                         batch_size=BATCH_SIZE,
                         num_workers=0, pin_memory=True)

# BiLSTM Model tanımı yapılmıştır.
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

#Model, loss fonksiyonu ve optimizer ayarlanmıştır.
model = BiLSTMModel(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)                 # Pad token göz ardı edilir.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Doğruluk ve F1-score hesaplama fonksiyonu
def calculate_metrics(preds, labels):
    preds = preds.argmax(dim=2).view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    mask = labels != 0
    preds = preds[mask]
    labels = labels[mask]
    acc = (preds == labels).mean() * 100
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

# Eğitim başlatılmaktadır.
best_val_loss = float("inf")
loss_log = []

print(f"🧠 Eğitim başlıyor (veri sınırı: {len(df)} satır, {EPOCHS} epoch)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_acc, total_f1 = 0, 0, 0

    #Eğitim döngüsü
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.view(-1, vocab_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, f1 = calculate_metrics(output, y_batch)
        total_acc += acc
        total_f1 += f1

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / len(train_loader)
    avg_train_f1 = total_f1 / len(train_loader)

    # Doğrulama döngüsü
    model.eval()
    val_loss, val_acc, val_f1 = 0, 0, 0
    with torch.no_grad():
        for val_Xb, val_yb in val_loader:
            val_Xb, val_yb = val_Xb.to(DEVICE), val_yb.to(DEVICE)
            val_out = model(val_Xb)
            vloss = criterion(val_out.view(-1, vocab_size), val_yb.view(-1))
            val_loss += vloss.item()
            acc, f1 = calculate_metrics(val_out, val_yb)
            val_acc += acc
            val_f1 += f1

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    avg_val_f1 = val_f1 / len(val_loader)
    
    
    # Epoch sonuçları
    print(f"\n📉 Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}% | "
          f"Train F1: {avg_train_f1:.4f} | Val F1: {avg_val_f1:.4f}")

    #Kayıt
    loss_log.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "train_acc": avg_train_acc,
        "val_acc": avg_val_acc,
        "train_f1": avg_train_f1,
        "val_f1": avg_val_f1
    })

    # En iyi model checkpoint olarak kaydedilir.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 En iyi model checkpoint olarak kaydedildi.")

# Eğitim kayıtları CSV dosyasına yazılmaktadır.
pd.DataFrame(loss_log).to_csv(LOG_PATH, index=False)
print("📊 Kayıtlar CSV'ye yazıldı:", LOG_PATH)

# Grafikler 
plt.figure()
plt.plot([l["epoch"] for l in loss_log], [l["train_loss"] for l in loss_log], label="Train Loss")
plt.plot([l["epoch"] for l in loss_log], [l["val_loss"] for l in loss_log], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Grafiği")
plt.savefig("loss_plot_lstm.png")
plt.close()

plt.figure()
plt.plot([l["epoch"] for l in loss_log], [l["train_acc"] for l in loss_log], label="Train Acc (%)")
plt.plot([l["epoch"] for l in loss_log], [l["val_acc"] for l in loss_log], label="Val Acc (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Grafiği")
plt.savefig("accuracy_plot_lstm.png")
plt.close()

plt.figure()
plt.plot([l["epoch"] for l in loss_log], [l["train_f1"] for l in loss_log], label="Train F1")
plt.plot([l["epoch"] for l in loss_log], [l["val_f1"] for l in loss_log], label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1-Score Grafiği")
plt.savefig("f1_plot_lstm.png")
plt.close()

print("✅ Eğitim tamamlandı ve grafikler oluşturuldu.")
