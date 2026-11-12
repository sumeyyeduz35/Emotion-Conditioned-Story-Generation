"""
Bu betik, duygu etiketli metinlerin yer aldığı 'merged_lstm_dataset.csv' dosyası kullanılarak:
 Metinlerin temizlenmesini sağlar,
 TensorFlow Tokenizer nesnesinin eğitilmesini gerçekleştirir,
 Tokenizer nesnesini 'tokenizer.pickle' dosyasına kaydeder,
 Kullanılan parametreleri raporlar.
"""

import pandas as pd
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Parametreler 
DATA_PATH = "./merged_lstm_dataset.csv"        # Girdi veri dosyasının yolu
TOKENIZER_PATH = "./tokenizer.pickle"          # Eğitilmiş tokenizer'ın kaydedileceği yol
MAX_TEXTS = 100000                             # Maksimum kullanılacak metin sayısı
OOV_TOKEN = "<OOV>"                            # Kelime dağarcığında olmayan kelimeler için sembol
MAX_LEN = 100                                  # Model için kullanılacak maksimum metin uzunluğu


# === Temizlik fonksiyonu tanımlanmıştır. ===
def clean_text(text):
    text = str(text).lower()                                # Küçült
    text = re.sub(r"http\S+", "", text)                     # Linkleri kaldır
    text = re.sub(r"[^a-zA-Z0-9şçöüğğıİŞÖÇÜı ]", "", text)  # Özel karakterleri kaldır
    text = re.sub(r"\s+", " ", text).strip()                # Fazla boşlukları kaldır
    return text

print("🔠 Tokenizer hazırlanıyor...")

# Veri kümesi yüklenmekte ve temizlenmektedir.
df = pd.read_csv(DATA_PATH).sample(n=MAX_TEXTS, random_state=42)  # Belirli sayıda örnek rastgele seçilmiştir
texts = [clean_text(t) for t in df["text"].astype(str).tolist()]  # Tüm metinlere temizlik işlemi uygulanmıştır


# Tokenizer eğitimi gerçekleştirilmiştir.
tokenizer = Tokenizer(num_words=None, oov_token=OOV_TOKEN)   # OOV token tanımlanarak tokenizer örneği oluşturulur
tokenizer.fit_on_texts(texts)                                # Temizlenmiş metinlere göre tokenizer eğitilir
vocab_size = len(tokenizer.word_index) + 1                   # Kelime dağarcığı boyutu hesaplanır

# Eğitilmiş tokenizer nesnesi kaydedilmektedir.
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)


# Bilgilendirme çıktıları verilmektedir.
print("✅ Tokenizer kaydedildi:", TOKENIZER_PATH)
print("📊 PARAMETRELER:")
print(f"  📄 Veri dosyası: {DATA_PATH}")
print(f"  🔢 MAX_TEXTS: {MAX_TEXTS}")
print(f"  🪙 OOV_TOKEN: {OOV_TOKEN}")
print(f"  📏 MAX_LEN (model input): {MAX_LEN}")
print(f"  📚 Vocab size: {vocab_size}")
