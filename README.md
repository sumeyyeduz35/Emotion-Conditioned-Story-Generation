
# Duygu Temelli Hikâye Üretimi: BiLSTM ve Transformer
# Emotion-Conditioned Story Generation with BiLSTM and Transformer

PyTorch kullanılarak geliştirilen, duygu koşullu metin üretimi için Bidirectional LSTM ve Transformer mimarilerini inceleyen ve karşılaştıran derin öğrenme projesi.

A deep learning project exploring and comparing Bidirectional LSTM and Transformer architectures for emotion-conditioned story generation using PyTorch.

**Technologies:** Python | PyTorch | BiLSTM | Transformer | NLP | Deep Learning | TensorFlow/Keras | Tkinter

---

# Türkçe

## Proje Hakkında

Bu proje, doğal dil işleme (Natural Language Processing - NLP) ve derin öğrenme yöntemleri kullanılarak duygu temelli hikâye üretimi üzerine geliştirilmiştir.

Çalışmada iki farklı sinir ağı mimarisi kullanılmıştır:

- Bidirectional Long Short-Term Memory (BiLSTM)
- Transformer

Her iki mimari için veri hazırlama, tokenizasyon, model eğitimi, doğrulama ve metin üretme işlemleri geliştirilmiştir.

Proje kapsamında modellerin eğitim ve doğrulama sonuçları analiz edilmiş; ayrıca LSTM modeli için Tkinter tabanlı bir masaüstü kullanıcı arayüzü oluşturulmuştur.

## Projenin Amacı

- Girilen başlangıç metnine göre hikâye üretmek
- Duygu bilgisini metin üretim sürecine dahil etmek
- BiLSTM ve Transformer mimarilerini karşılaştırmak
- Farklı metin üretim ve örnekleme yöntemlerini incelemek
- Model performansını eğitim ve doğrulama metrikleriyle değerlendirmek
- Kullanıcıların metin üretimini grafiksel arayüz üzerinden deneyimlemesini sağlamak

## Duygu Kategorileri

LSTM kullanıcı arayüzünde beş farklı duygu seçeneği bulunmaktadır:

| Duygu | Açıklama |
|---|---|
| Happy | Mutluluk |
| Sad | Üzüntü |
| Fear | Korku |
| Anger | Öfke |
| Neutral | Nötr |

Duygu bilgisi, başlangıç metniyle birleştirilerek model girişine dahil edilmektedir.

Bu mekanizmanın uygulanmış olması, üretilen hikâyelerin seçilen duyguyu her zaman doğru yansıttığı anlamına gelmez.

## Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| Python | Uygulama geliştirme |
| PyTorch | Derin öğrenme modelleri |
| LSTM | Sıralı metin modelleme |
| Transformer | Attention tabanlı metin modelleme |
| TensorFlow/Keras | Tokenizasyon ve dizi hazırlama |
| Pandas | Veri işleme ve eğitim kayıtları |
| NumPy | Sayısal işlemler |
| Scikit-learn | Performans metrikleri |
| Matplotlib | Eğitim grafiklerinin oluşturulması |
| Tkinter | Masaüstü kullanıcı arayüzü |

## 1. BiLSTM Modeli

BiLSTM modeli, kelime seviyesinde sıralı metin üretimi amacıyla geliştirilmiştir.

### Model Mimarisi

Model aşağıdaki bileşenlerden oluşmaktadır:

- Embedding katmanı
- İki katmanlı Bidirectional LSTM
- Dropout
- Fully Connected çıkış katmanı

### Eğitim Parametreleri

| Parametre | Değer |
|---|---|
| Eğitim örneği | 10.000 |
| Epoch | 10 |
| Batch Size | 128 |
| Embedding Dimension | 128 |
| Hidden Dimension | 512 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Sequence Length | 100 |
| Eğitim / Doğrulama | %90 / %10 |

Modelin eğitiminde CrossEntropyLoss kullanılmış ve padding token'ları kayıp hesabı dışında bırakılmıştır.

En düşük doğrulama kaybına sahip model ağırlıkları checkpoint olarak kaydedilmiştir.

### LSTM Metin Üretimi

Metin üretim aşamasında:

1. Kullanıcının seçtiği duygu ve başlangıç metni alınır.
2. Duygu bilgisi metne önek olarak eklenir.
3. Metin token dizisine dönüştürülür.
4. Eğitimli BiLSTM modeliyle sonraki token tahminleri oluşturulur.
5. Temperature sampling kullanılarak yeni token'lar seçilir.
6. Üretilen token'lar metne dönüştürülür.

### Kullanıcı Arayüzü

`gui_lstm.py` dosyasında Tkinter kullanılarak masaüstü uygulaması geliştirilmiştir.

Arayüzde:

- Başlangıç metni girişi
- Duygu seçimi
- Metin oluşturma butonu
- Üretilen metin alanı

bulunmaktadır.

## 2. Transformer Modeli

Transformer modeli, attention mekanizması kullanılarak duygu temelli metin üretimini araştırmak amacıyla geliştirilmiştir.

### Model Mimarisi

Modelde:

- Token embedding
- Emotion embedding
- Öğrenilebilir positional encoding
- Transformer encoder-decoder yapısı
- Multi-head attention
- Fully Connected çıkış katmanı

bulunmaktadır.

Uygulamada bir encoder ve altı decoder katmanıyla model oluşturulmuştur.

### Eğitim Parametreleri

| Parametre | Değer |
|---|---|
| Eğitim örneği | 20.000 |
| Epoch | 10 |
| Batch Size | 32 |
| Embedding Dimension | 256 |
| Feedforward Dimension | 512 |
| Attention Heads | 8 |
| Encoder Layers | 1 |
| Decoder Layers | 6 |
| Dropout | 0.1 |
| Learning Rate | 0.00005 |
| Optimizer | AdamW |
| Sequence Length | 60 |
| Eğitim / Doğrulama | %90 / %10 |

### Transformer Metin Üretimi

Transformer üretim sürecinde:

1. Duygu bilgisi ve başlangıç metni birleştirilir.
2. Giriş metni token dizisine dönüştürülür.
3. Model, yeni token'lar için tahmin skorları üretir.
4. Temperature ve Top-p (Nucleus Sampling) yöntemleri kullanılır.
5. Üretilen token'lar birleştirilerek hikâye metni oluşturulur.

Üretim kodunda varsayılan Top-p değeri 0.9, Temperature değeri 1.0 olarak tanımlanmıştır.

Transformer modelinde ayrı emotion embedding yapısı bulunmasına rağmen eğitimde duygu etiketi sabit olarak kullanılmaktadır. Bu nedenle bağımsız duygu embedding'leri üzerinden etkili duygu koşullandırması yapıldığı doğrulanmış değildir.

## 3. Eğitim ve Performans Analizi

Her iki model için aşağıdaki metrikler kaydedilmiştir:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Training F1-score
- Validation F1-score

### Son Epoch Sonuçları

| Metrik | BiLSTM | Transformer |
|---|---:|---:|
| Epoch | 10 | 10 |
| Training Loss | 0.0478 | 1.2660 |
| Validation Loss | 1.0093 | 1.3300 |
| Training Accuracy | %99.99 | %85.26 |
| Validation Accuracy | Yaklaşık %92.31 | Yaklaşık %86.50 |

Bu sonuçlar eğitim sürecinde kaydedilen son epoch verileridir.

Accuracy değerleri, hedef token tahminleri üzerinden hesaplanmıştır. Hikâyelerin anlam bütünlüğünü, yaratıcılığını veya duygusal tutarlılığını doğrudan ölçmez.

Modeller farklı eğitim örneği sayıları ve yapılandırmalarla çalıştırıldığından, sonuçlar kontrollü bir mimari karşılaştırmasının kesin kanıtı olarak değerlendirilmemelidir.

LSTM modelinin yüksek eğitim doğruluğuna karşılık daha düşük doğrulama doğruluğu elde etmesi, genelleme performansı ve olası aşırı öğrenme açısından ayrıca incelenmelidir.

## 4. Görselleştirme

Her iki model için eğitim sürecini incelemeye yönelik grafikler oluşturulmuştur:

- Loss grafikleri
- Accuracy grafikleri
- F1-score grafikleri

Ek olarak proje raporunda kullanıcı arayüzüne ve üretilen metin örneklerine ilişkin görseller bulunmaktadır.

## 5. Proje Yapısı

```text
Emotion-Conditioned-Story-Generation/
├── lstm_hikaye_motoru/
│   ├── model.py
│   ├── train_lstm_advanced.py
│   ├── data_prepare_lstm_conditional.py
│   ├── tokenizer_prepare_lstm.py
│   ├── test_lstm.py
│   ├── gui_lstm.py
│   ├── kodlar/
│   ├── accuracy_plot_lstm.png
│   ├── f1_plot_lstm.png
│   ├── loss_plot_lstm.png
│   └── tokenizer.pickle
│
├── transformer__hikaye/
│   ├── model.py
│   ├── train_transformer.py
│   ├── data_prepare_transformer.py
│   ├── tokenizer.py
│   ├── generate.py
│   ├── gpt-2_generate.py
│   ├── gpt2_generate_translate.py
│   ├── gpt2_generate_translate_bidir.py
│   ├── gpu.py
│   ├── accuracy_plot_transformer.png
│   ├── f1_plot_transformer.png
│   ├── loss_plot_transformer.png
│   ├── tokenizer.pickle
│   └── vocab.json
│
├── RAPOR + VERİ/
│   ├── GÖRSELLER/
│   └── DUYGU TEMELLİ HİKAYE ÜRETİMİ.docx
│
├── .gitignore
└── README.md
```

Depoda LSTM kaynak dosyalarının `kodlar/` altında ayrıca bir kopyası bulunmaktadır.

## 6. Kurulum

### Gereksinimler

- Python
- PyTorch
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- tqdm
- Tkinter

### Repository'yi Klonlama

```bash
git clone https://github.com/sumeyyeduz35/Emotion-Conditioned-Story-Generation.git
cd Emotion-Conditioned-Story-Generation
```

### Python Bağımlılıkları

```bash
python -m pip install torch tensorflow pandas numpy scikit-learn matplotlib tqdm
```

### LSTM Arayüzünü Çalıştırma

Eğitilmiş model checkpoint dosyasının `lstm_hikaye_motoru` klasöründe bulunması gerekmektedir.

```bash
cd lstm_hikaye_motoru
python gui_lstm.py
```

### Transformer ile Metin Üretimi

Eğitilmiş Transformer checkpoint dosyasının `transformer__hikaye` klasöründe bulunması gerekmektedir.

```bash
cd transformer__hikaye
python generate.py
```

**Not:** Büyük model checkpoint dosyaları ve eğitim veri setleri GitHub deposuna eklenmemiştir. Bu nedenle yeni bir ortamda doğrudan metin üretimi yapmak için ilgili ağırlıkların ve veri dosyalarının ayrıca sağlanması veya modellerin yeniden eğitilmesi gerekir.

Veri hazırlama ve eğitim betikleri kaynak kodda bulunmaktadır; ancak eksik veri dosyaları nedeniyle yeniden eğitim süreci yalnızca depoyu klonlayarak başlatılamaz.

## 7. Teknik Sınırlamalar

- Model başarı metrikleri token tahminine dayanmaktadır.
- Üretilen hikâyelerin duygusal tutarlılığı için ayrıca değerlendirme gereklidir.
- BiLSTM ve Transformer farklı veri miktarlarıyla eğitilmiştir.
- Transformer modelinde duygu etiketleri ayrı embedding üzerinden etkin biçimde kullanılmamaktadır.
- Büyük checkpoint ve veri dosyaları depoda bulunmamaktadır.
- Mevcut kodların farklı Python ve kütüphane sürümleriyle uyumluluğu ayrıca doğrulanmalıdır.
- Otomatik hikâye üretiminin anlamsal tutarlılığı garanti edilmemektedir.

## Proje Durumu

**Araştırma ve model geliştirme çalışması**

BiLSTM ve Transformer mimarileri geliştirilmiş, eğitim deneyleri gerçekleştirilmiş ve performans kayıtları oluşturulmuştur.

LSTM için masaüstü metin üretim arayüzü hazırlanmıştır.

Projenin yeniden üretilebilirliği, model dağıtımı ve duygu koşullandırmasının değerlendirilmesi geliştirilebilecek alanlardır.

---

# English

## About the Project

This project investigates emotion-conditioned story generation using natural language processing and deep learning techniques.

Two neural network architectures are implemented:

- Bidirectional Long Short-Term Memory (BiLSTM)
- Transformer

The project includes data preparation, tokenization, model training, validation, and text generation workflows.

Training and validation metrics are recorded for both architectures. A Tkinter-based desktop interface is also implemented for the LSTM model.

## Project Objectives

- Generate stories from user-provided text prompts
- Incorporate emotion information into text generation
- Explore BiLSTM and Transformer architectures
- Investigate token sampling strategies
- Evaluate training and validation performance
- Provide a graphical interface for experimenting with generated text

## Emotion Categories

The LSTM interface supports five emotion options:

| Emotion | Category |
|---|---|
| Happy | Happiness |
| Sad | Sadness |
| Fear | Fear |
| Anger | Anger |
| Neutral | Neutral |

Emotion information is included in the input text through prefix conditioning.

The presence of this mechanism does not guarantee that generated stories consistently express the selected emotion.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| PyTorch | Deep learning models |
| BiLSTM | Sequential language modeling |
| Transformer | Attention-based language modeling |
| TensorFlow/Keras | Text preprocessing and padding |
| Pandas | Data processing and experiment logging |
| NumPy | Numerical computations |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Training visualizations |
| Tkinter | Desktop GUI |

## 1. BiLSTM Architecture

A Bidirectional LSTM model is implemented for word-level text modeling and generation.

The architecture includes:

- Embedding layer
- Two-layer Bidirectional LSTM
- Dropout
- Fully connected output layer

### Training Configuration

| Parameter | Value |
|---|---|
| Training Samples | 10,000 |
| Epochs | 10 |
| Batch Size | 128 |
| Embedding Dimension | 128 |
| Hidden Dimension | 512 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Sequence Length | 100 |
| Training / Validation | 90% / 10% |

Cross-entropy loss is used, excluding padding tokens from the loss computation.

The model checkpoint is saved when the validation loss improves.

### Text Generation

The LSTM generation workflow:

1. Receives the initial text and selected emotion.
2. Adds the emotion as a text prefix.
3. Converts the input into token IDs.
4. Performs next-token prediction using the trained model.
5. Samples tokens using temperature scaling.
6. Converts generated tokens back into text.

### Desktop Interface

The Tkinter interface provides:

- Prompt input
- Emotion selection
- Text generation button
- Generated text display

## 2. Transformer Architecture

The Transformer architecture is implemented to explore attention-based text generation.

The model includes:

- Token embeddings
- Emotion embeddings
- Learnable positional encodings
- Transformer encoder-decoder structure
- Multi-head attention
- Fully connected output layer

The training configuration uses one encoder layer and six decoder layers.

### Training Configuration

| Parameter | Value |
|---|---|
| Training Samples | 20,000 |
| Epochs | 10 |
| Batch Size | 32 |
| Embedding Dimension | 256 |
| Feedforward Dimension | 512 |
| Attention Heads | 8 |
| Encoder Layers | 1 |
| Decoder Layers | 6 |
| Dropout | 0.1 |
| Learning Rate | 0.00005 |
| Optimizer | AdamW |
| Sequence Length | 60 |
| Training / Validation | 90% / 10% |

### Text Generation

The Transformer generation process includes:

1. Combining the emotion and starting prompt.
2. Tokenizing the input text.
3. Computing next-token prediction scores.
4. Applying temperature scaling and Top-p nucleus sampling.
5. Combining generated tokens into a story.

The default generation parameters include Top-p = 0.9 and Temperature = 1.0.

Although an emotion embedding layer is implemented, training uses a constant emotion label. Therefore, effective emotion conditioning through distinct emotion embeddings has not been established.

## 3. Model Evaluation

Both architectures record:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Training F1-score
- Validation F1-score

### Final Epoch Results

| Metric | BiLSTM | Transformer |
|---|---:|---:|
| Epoch | 10 | 10 |
| Training Loss | 0.0478 | 1.2660 |
| Validation Loss | 1.0093 | 1.3300 |
| Training Accuracy | 99.99% | 85.26% |
| Validation Accuracy | Approximately 92.31% | Approximately 86.50% |

The values represent recorded final-epoch results.

Accuracy is calculated from token prediction performance rather than human evaluation of generated stories.

The two models were trained on different sample sizes and configurations, so the results do not establish a fully controlled architectural benchmark.

The gap between BiLSTM training and validation accuracy also warrants further investigation of generalization and possible overfitting.

## 4. Visualization

The repository includes:

- Training and validation loss plots
- Accuracy plots
- F1-score plots
- Screenshots of the LSTM interface
- Experiment-related figures and documentation

## 5. Project Structure

```text
Emotion-Conditioned-Story-Generation/
├── lstm_hikaye_motoru/
│   ├── model.py
│   ├── train_lstm_advanced.py
│   ├── data_prepare_lstm_conditional.py
│   ├── tokenizer_prepare_lstm.py
│   ├── test_lstm.py
│   ├── gui_lstm.py
│   ├── kodlar/
│   └── training plots and tokenizer files
│
├── transformer__hikaye/
│   ├── model.py
│   ├── train_transformer.py
│   ├── data_prepare_transformer.py
│   ├── tokenizer.py
│   ├── generate.py
│   ├── gpt-2_generate.py
│   ├── gpt2_generate_translate.py
│   ├── gpt2_generate_translate_bidir.py
│   ├── vocab.json
│   └── training plots and tokenizer files
│
├── RAPOR + VERİ/
├── .gitignore
└── README.md
```

## 6. Installation and Usage

### Requirements

- Python
- PyTorch
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- tqdm
- Tkinter

### Clone the Repository

```bash
git clone https://github.com/sumeyyeduz35/Emotion-Conditioned-Story-Generation.git
cd Emotion-Conditioned-Story-Generation
```

### Install Python Dependencies

```bash
python -m pip install torch tensorflow pandas numpy scikit-learn matplotlib tqdm
```

### Run the LSTM Interface

The trained LSTM checkpoint must be available in the `lstm_hikaye_motoru` directory.

```bash
cd lstm_hikaye_motoru
python gui_lstm.py
```

### Generate Text with Transformer

The trained Transformer checkpoint must be available in the `transformer__hikaye` directory.

```bash
cd transformer__hikaye
python generate.py
```

**Note:** Large model checkpoints and training datasets are excluded from the repository.

Running inference in a new environment requires the relevant checkpoints and tokenizer files. Retraining also requires the corresponding datasets.

## 7. Technical Limitations

- Accuracy metrics reflect token-level prediction performance.
- Emotional consistency and narrative coherence require additional evaluation.
- BiLSTM and Transformer were trained with different dataset sizes.
- Distinct emotion embeddings are not effectively trained in the current Transformer implementation.
- Large checkpoint and dataset files are not included.
- Compatibility across Python and library versions has not been independently verified.
- Generated text quality and coherence are not guaranteed.

## Project Status

**Research and model development project**

BiLSTM and Transformer architectures have been implemented, training experiments have been conducted, and performance metrics have been recorded.

A Tkinter-based LSTM text generation interface has also been developed.

Further improvements may focus on reproducibility, model deployment, and evaluation of emotion-conditioned generation.
