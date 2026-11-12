# Bu betik, GoEmotions ve DailyDialog veri setlerinden elde edilen metinlerin birleştirilerek
# duygu temelli bir biçimde yapılandırılmasını ve eğitim için uygun hale getirilmesini sağlar.
# Çıktı dosyası, LSTM tabanlı metin üretim modellerine giriş olarak kullanılabilecek şekilde hazırlanır.

import os
import pandas as pd
from datasets import load_dataset

# Çıktı CSV dosyasının ismi belirlenmiştir.
OUTPUT_CSV = "merged_lstm_dataset_conditional.csv"

# Veri setlerinin kaydedileceği klasör oluşturulmuştur (zaten mevcutsa hata vermez).
os.makedirs("datasets", exist_ok=True)

# GoEmotions veri seti yüklenmektedir.
print("📥 GoEmotions veri seti yükleniyor...")
goemotions = load_dataset("go_emotions", split="train")

# Etiketler ilk duygu olarak alınır; boş olanlara "neutral" atanır.
go_df = pd.DataFrame({
    "text": goemotions["text"],
    "label": [goemotions.features["labels"].feature.names[label[0]] if label else "neutral" for label in goemotions["labels"]]
})
print(f"✅ GoEmotions: {len(go_df)} satır")

# 📦 DailyDialog veri seti yüklenmektedir.
print("📥 DailyDialog veri seti yükleniyor...")
dailydialog = load_dataset("daily_dialog", trust_remote_code=True)
dd_texts, dd_labels = [], []

# Etiketleri duygulara çeviren bir eşleme tablosu tanımlanmıştır.
emotion_map = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "happy", 5: "sad", 6: "surprise"
}


# DailyDialog veri setinden metin ve etiketler toplanmaktadır.
for split in ["train", "validation", "test"]:
    for dialog, emotions in zip(dailydialog[split]["dialog"], dailydialog[split]["emotion"]):
        for text, label in zip(dialog, emotions):
            dd_texts.append(text)
            dd_labels.append(emotion_map.get(label, "neutral"))  # Tanımsız etiketler "neutral" olarak değerlendirilir.
            
# Toplanan veriler DataFrame formatına dönüştürülmektedir.           
dd_df = pd.DataFrame({
    "text": dd_texts,
    "label": dd_labels
})
print(f"✅ DailyDialog: {len(dd_df)} satır")

# GoEmotions ve DailyDialog veri setleri birleştirilmektedir.
print("📦 Veri setleri birleştiriliyor...")
merged_df = pd.concat([go_df, dd_df], ignore_index=True)

# Duygu etiketleri hedef duygu kümelerine (happy, sad, fear, anger, neutral, disgust) indirgenmektedir.
label_map = {
    "admiration": "happy", "amusement": "happy", "approval": "happy", "caring": "happy",
    "desire": "happy", "excitement": "happy", "gratitude": "happy", "joy": "happy",
    "love": "happy", "optimism": "happy", "pride": "happy", "relief": "happy",
    "surprise": "happy", "happiness": "happy",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "embarrassment": "sad", "fear": "fear", "grief": "sad", "nervousness": "fear",
    "remorse": "sad", "sadness": "sad", "disgust": "disgust",
    "neutral": "neutral"
}

# Etiketlerin normalize edilmesi sağlanmaktadır.
merged_df["label"] = merged_df["label"].str.lower().map(label_map).fillna("neutral")

# Her metin, ilgili duygu etiketiyle birleştirilerek "duygu: metin" formatına dönüştürülmüştür.
# Bu yapı, modelin duyguya şartlı metin üretimini kolaylaştırmak amacıyla tercih edilmiştir. (prefix conditioning)
merged_df["text"] = merged_df["label"] + ": " + merged_df["text"].astype(str).str.lower()

# Metin temizleme işlemi uygulanmaktadır.
# URL'ler ve özel karakterler kaldırılmakta, fazla boşluklar sadeleştirilmektedir.
def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)                         # URL'ler temizlenir.
    text = re.sub(r"[^a-zA-Z0-9şçöüğğıİŞÖÇÜı: ]", "", text)     # Özel karakterler kaldırılır.
    text = re.sub(r"\s+", " ", text).strip()                    # Fazla boşluklar sadeleştirilir.
    return text

# Sadece "text" sütunu içeren nihai veri seti oluşturulmaktadır.
merged_df["text"] = merged_df["text"].apply(clean_text)


# Oluşturulan veri kümesi belirtilen CSV dosyasına yazılmaktadır.
merged_df = merged_df[["text"]]
merged_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"🎉 Final veri seti oluşturuldu: {OUTPUT_CSV} ({len(merged_df)} satır)")
