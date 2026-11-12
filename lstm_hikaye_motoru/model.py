# Bu dosyada, kelime seviyesinde metin üretimi yapabilen çift yönlü (bidirectional) bir LSTM mimarisi tanımlanmıştır.
# Model, gömülme (embedding) katmanı, çok katmanlı çift yönlü LSTM katmanı ve bir çıkış (linear) katmanından oluşmaktadır.

import torch
import torch.nn as nn

# LSTM tabanlı model sınıfı tanımlanmıştır.
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        # Girdi kelime ID'leri, sabit boyutlu vektörlere dönüştürülmektedir.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
         # Çift yönlü (bidirectional) çok katmanlı LSTM katmanı tanımlanmıştır.
        self.lstm = nn.LSTM(
            input_size=embed_dim,         # Gömülme katmanından gelen giriş boyutu
            hidden_size=hidden_dim,       # Gizli durum boyutu
            num_layers=num_layers,        # LSTM katman sayısı
            dropout=dropout,              # Katmanlar arası dropout
            batch_first=True,             # Girdiler [batch, seq, feature] biçimindedir
            bidirectional=True            # LSTM ileri ve geri yönde çalışır (context genişletilir)
        )
        
        # Çıkış katmanı: LSTM çıktısını kelime tahminine çevirir.
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # ✅ Bidirectional: hidden_dim*2

    def forward(self, x):
        x = self.embedding(x)      # Girdi dizisi gömülme vektörlerine dönüştürülür.
        out, _ = self.lstm(x)      # Gömülme vektörleri LSTM'e verilir.
        out = self.fc(out)         # LSTM çıktıları, tam bağlantılı katman ile kelime tahminine çevrilir.
        return out                 # [batch_size, seq_len, vocab_size] boyutunda çıktı üretilir.