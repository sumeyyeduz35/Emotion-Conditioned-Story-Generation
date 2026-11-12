# Bu betikte, eğitilmiş bir LSTM modelinden yararlanılarak GUI üzerinden duyguya koşullu metin üretimi sağlanmaktadır.
# Kullanıcıdan başlangıç cümlesi ve duygu girdisi alınarak, örnekleme yoluyla devam eden bir metin oluşturulmaktadır.

import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import LSTMModel                                              # LSTM modeli kullanıcı tarafından tanımlanmış modülden içe aktarılır

# Model ve Tokenizer için parametreler tanımlanmıştır.
TOKENIZER_PATH = "./tokenizer.pickle"                                    # Eğitilmiş tokenizer'ın yolu
MODEL_PATH = "./best_model_checkpoint.pth"                               # En iyi model checkpoint dosyasının yolu
MAX_SEQUENCE_LEN = 100                                                   # Maksimum dizilim uzunluğu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Uygun cihaz belirlenir

# Tokenizer yüklenmiştir.
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1                               # Kelime sayısı hesaplanır

# Model hazırlanmıştır. 
model = LSTMModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))       # Eğitimli ağırlıklar yüklenir.
model.to(DEVICE)
model.eval()                                                             # Model değerlendirme moduna alınır.

# Örnekleme yöntemi ile metin üretimi fonksiyonu tanımlanmıştır.
def generate_text(prompt, temperature=0.8, max_len=50):
    """
    Kullanıcıdan alınan başlangıç ifadesine göre LSTM modeli ile örnekleme yapılarak metin üretimi sağlanır.
    :param prompt: Başlangıç metni
    :param temperature: Rastgelelik derecesi
    :param max_len: Maksimum üretilecek kelime sayısı
    :return: Üretilmiş metin (string)
    """
    generated = tokenizer.texts_to_sequences([prompt.lower()])[0]
    generated = generated[-MAX_SEQUENCE_LEN:]                            # Maksimum uzunlukla sınırlandırılır.

    for _ in range(max_len):
        input_seq = pad_sequences([generated], maxlen=MAX_SEQUENCE_LEN, padding='pre')  #Sıralar pad edilir.
        input_tensor = torch.tensor(input_seq, dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)                                 # Model tahmini alınır.

        next_token_logits = output[0, -1, :] / temperature               #Son adımın tahmini alınır ve sıcaklıkla ölçeklenir.
        probs = torch.softmax(next_token_logits, dim=0).cpu().numpy()    #Olasılık dağılımı elde edilir.
        predicted_id = np.random.choice(len(probs), p=probs)             # Bir sonraki kelime örneklenir.

        if predicted_id == 0:                                            # 0 ID boş/sonlandırıcı kabul edilmiştir.
            break
        generated.append(predicted_id)

    # ID'ler tekrar kelimelere çevrilir.
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    generated_text = ' '.join([reverse_word_index.get(i, '') for i in generated if i != 0])
    return generated_text

# Grafiksel arayüz oluşturmaktadır. (GUI)
def create_gui():
    """
    Kullancııdan başlangıç metni ve duygu girdisi alan, çıktıyı ekranda gösteren grafik arayüz oluşturulmuştur.
    """
    window = tk.Tk()
    window.title("LSTM Metin Oluşturucu")
    window.geometry("600x500")
    window.resizable(False, False)

    # Başlangıç metni girişi
    tk.Label(window, text="Başlangıç Metni:", font=("Arial", 12)).pack(pady=5)
    entry_prompt = tk.Entry(window, width=60, font=("Arial", 12))
    entry_prompt.pack(pady=5)

    # Duygu seçimi için açılır kutu
    tk.Label(window, text="Duygu Seçin:", font=("Arial", 12)).pack(pady=5)
    emotion_var = tk.StringVar()
    combo_emotion = ttk.Combobox(window, textvariable=emotion_var, values=["happy", "sad", "fear", "anger", "neutral"], font=("Arial", 12))
    combo_emotion.pack(pady=5)
    combo_emotion.current(0)                                              #Varsayılan olarak ilk seçenek(happy) seçilir.

    # Metin çıktısı için kaydırılabilir alan. 
    tk.Label(window, text="Üretilen Metin:", font=("Arial", 12)).pack(pady=5)
    output_text = scrolledtext.ScrolledText(window, width=70, height=10, font=("Arial", 12))
    output_text.pack(pady=5)

    # Buton fonksiyonu
    def generate_and_show():
        prompt = entry_prompt.get().strip()
        emotion = combo_emotion.get().strip()
        if not prompt:
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, "⚠️ Lütfen bir başlangıç metni girin.")
            return
        prompt_with_emotion = f"{emotion} {prompt}"                      # Prefix conditioning uygulanır.
        generated = generate_text(prompt_with_emotion)                   # Metin üretilir.
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, generated)

    # Buton
    tk.Button(window, text="Metin Oluştur", command=generate_and_show, font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)

    window.mainloop()                                                    #Arayüz başlatılır.

if __name__ == "__main__":
    create_gui()
