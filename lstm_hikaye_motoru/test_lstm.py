# Bu betikte, eğitimli LSTM modelinden yararlanılarak terminal üzerinden etkileşimli şekilde metin üretimi sağlanmaktadır.
# Kullanıcıdan alınan başlangıç cümlesine göre modelin devam tahmini yapılmakta ve çıktı terminale yazdırılmaktadır.

import os
import torch
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import LSTMModel                                                     # Eğitilmiş model sınıfı içe aktarılmıştır.


# Parametreler
TOKENIZER_PATH = "./tokenizer.pickle"
MODEL_PATH = "./best_model_checkpoint.pth"
MAX_SEQUENCE_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

# Modeli hazırlar.
model = LSTMModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()                                                                      #Model değerlendirme moduna alınır.
print("✅ Model checkpoint yüklendi ve hazır!")

# Sampling tabanlı metin üretim fonksiyonu tanımlanmıştır.
def generate_text(prompt, temperature=0.8, max_len=50):
    
    """
    Verilen başlangıç metnine dayalı olarak LSTM modelinden devam tahmini alınmaktadır.
    Sıcaklık (temperature) değeri örnekleme çeşitliliğini belirler.
    """
    
    generated = tokenizer.texts_to_sequences([prompt.lower()])[0]                #Token ID'lerine dönüştürülür.
    generated = generated[-MAX_SEQUENCE_LEN:]                                    # Maksimum uzunlukla sınırlandırılır.

    for _ in range(max_len):
        # Giriş dizisi pad edilerek modele uygun hale getirilir.
        input_seq = pad_sequences([generated], maxlen=MAX_SEQUENCE_LEN, padding='pre')
        input_tensor = torch.tensor(input_seq, dtype=torch.long).to(DEVICE)

        # Tahmin işlemi gerçekleştirilir.
        with torch.no_grad():
            output = model(input_tensor)

        # Sıcaklık ayarlı sampling işlemi gerçekleştirilir.
        next_token_logits = output[0, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=0).cpu().numpy()
        predicted_id = np.random.choice(len(probs), p=probs)                    #Olasılığa göre yeni token seçilir.

        if predicted_id == 0:                                                   #Pad token üretildiyse üretim durdurulur.
            break

        # Yeni kelimeyi dizine ekle
        generated.append(predicted_id)                                          # Yeni token diziye eklenir

    # Üretilen ID'ler metne dönüştürülür.
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    generated_text = ' '.join([reverse_word_index.get(i, '') for i in generated if i != 0])
    return generated_text

# Etkileşimli terminal arayüzü başlatmıştır. 
print("\n👉 Başlangıç metni girin (çıkmak için 'exit'): ")
while True:
    user_input = input().strip()
    if user_input.lower() == 'exit':
        print("👋 Program sonlandırıldı.")
        break

    output_text = generate_text(user_input, temperature=0.8, max_len=50)
    print("\n🌟 Modelin devam önerisi:")
    print(output_text)
    print("\n👉 Yeni başlangıç metni girin (çıkmak için 'exit'): ")
