# 🇹🇷 Türkçe Metin Duygu Analizi Sistemi

Bu proje, **Yapay Zeka Staj Projesi** kapsamında geliştirilmiş uçtan uca bir
Doğal Dil İşleme (NLP) uygulamasıdır. Kullanıcı tarafından girilen Türkçe
metinlerin duygu durumunu **Pozitif**, **Negatif** veya **Nötr** olarak
sınıflandırır.

> **Durum:** Proje tamamlandı ✅ — veri toplama, ön işleme, model eğitimi/karşılaştırma
> ve Streamlit demo arayüzü uçtan uca test edilmiştir.

## ⚡ Hızlı Başlatma (En Kolay Yol)

Kurulum bir kere yapıldıktan sonra (aşağıdaki "Kurulum" bölümüne bakın),
demoyu her açmak istediğinizde tek yapmanız gereken proje klasöründeki
**`uygulamayi_calistir.bat`** dosyasına **çift tıklamak**. Bu dosya sanal
ortamı otomatik etkinleştirir ve Streamlit arayüzünü açar; tarayıcı
kendiliğinden `http://localhost:8501` adresini açacaktır. Kapatmak için
açılan siyah pencereyi kapatmanız yeterlidir.

## 📌 Proje Özeti

| | |
|---|---|
| **Görev** | 3 sınıflı metin duygu sınıflandırması |
| **Veri seti** | [TRSAv1](https://huggingface.co/datasets/maydogan/Turkish_SentimentAnalysis_TRSAv1) — 150.000 örnek, sınıf başına tam 50.000 (dengeli) |
| **Modeller** | Naive Bayes, Logistic Regression, Linear SVM (TF-IDF öznitelikleri ile) |
| **En iyi model** | Naive Bayes (~%77.5 accuracy, ~%77.4 macro-F1) |
| **Arayüz** | Streamlit demo uygulaması |

## 📂 Proje Yapısı
```text
turkish-sentiment-analysis/
├── app/
│   └── streamlit_app.py        # Gömülü preprocessing içeren demo web arayüzü
├── data/                        # Veri seti klasörü (ham ve işlenmiş veriler)
│   ├── README.md/
├── images/                      # EDA ve veri seti analiz grafikleri
├── models/                      # Eğitilmiş hazır .joblib model dosyaları
├── notebooks/
│   └── Turkce_Duygu_Analizi_Colab.ipynb  # Google Colab'da çalıştırılabilir sürüm
├── reports/
│   └── teknik_rapor.pdf       
├── venv/                        # Python Yerel Sanal Ortam klasörü
├── .gitignore                   # GitHub'a yüklenmeyecek dosyaların listesi
├── README.md                    # Proje tanıtım ve kılavuz belgesi
├── requirements.txt             # Gerekli bağımlılıklar ve kütüphaneler listesi
└── uygulamayi_calistir.bat      # Windows için tek tıkla uygulamayı başlatma aracı

## 🚀 Kurulum

```bash
# 1) Sanal ortam oluştur ve etkinleştir
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2) Bağımlılıkları kur
pip install -r requirements.txt

# 3) NLTK stopwords verisini indir
python -c "import nltk; nltk.download('stopwords')"
```

## ▶️ Projeyi Baştan Sona Çalıştırma

```bash
# 1) Veri setini indir (Hugging Face - internet gerektirir)
python src/download_dataset.py

# 2) Keşifsel veri analizi (grafikler reports/figures altına kaydedilir)
python src/eda.py

# 3) Veri temizleme ve train/test ayrımı
python src/prepare_data.py

# 4) Modelleri eğit ve karşılaştır
python src/train_models.py

# 5) Demo arayüzünü başlat
streamlit run app/streamlit_app.py

# (İsteğe bağlı) Word raporu ve sunum slaytlarını yeniden üret
python tools/generate_report_docx.py
python tools/generate_presentation.py
```

Uygulama açıldıktan sonra tarayıcıda `http://localhost:8501` adresinden
erişilebilir. Kurulum gerektirmeden çalıştırmak isterseniz
`notebooks/Turkce_Duygu_Analizi_Colab.ipynb` dosyasını Google Colab'a
yükleyebilirsiniz.

## 📊 Veri Seti Hakkında

**TRSAv1 (Turkish Sentiment Analysis v1)**, Türkçe e-ticaret sitelerindeki
kullanıcı yorumlarından oluşan, akademik olarak yayınlanmış bir referans
veri setidir (Aydoğan & Kocaman, 2022, *Journal of Information Science*).

- 150.000 toplam örnek
- **Pozitif: 50.000 / Negatif: 50.000 / Nötr: 50.000** → tamamen dengeli
- Bu dengeli yapı, modelin herhangi bir sınıfa karşı yanlı (biased)
  öğrenmesini engeller ve sınıflar arası performansın adil şekilde
  karşılaştırılmasını sağlar.

Detaylı analiz ve grafikler için `reports/figures/` klasörüne ve
`reports/teknik_rapor.docx` dosyasının 3. bölümüne bakabilirsiniz.

## 🧹 Ön İşleme (Preprocessing)

`src/preprocessing.py` içindeki `clean_text()` fonksiyonu şu adımları uygular:

1. Türkçe'ye özgü küçük harfe çevirme (İ→i, I→ı)
2. URL, e-posta, kullanıcı adı/hashtag temizleme
3. Sayı ve noktalama işaretlerinin kaldırılması
4. Türkçe stopword (etkisiz kelime) filtreleme (NLTK + özel liste)
5. Çok kısa kelimelerin (<2 karakter) filtrelenmesi

## 🤖 Modelleme

Metinler **TF-IDF** (unigram + bigram, max 30.000 öznitelik) ile
vektörleştirilip üç farklı klasik makine öğrenmesi modeli eğitilmiştir:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Naive Bayes** | **0.7750** | **0.7737** | **0.7748** |
| Logistic Regression | 0.7730 | 0.7713 | 0.7725 |
| SVM (Linear) | 0.7684 | 0.7663 | 0.7675 |



## 🖥️ Demo Arayüzü

Geliştirilen Streamlit web uygulaması, kullanıcı deneyimini artırmak ve projenin akademik/teknik tüm aşamalarını şeffaf bir şekilde sunabilmek adına 3 ana sekme (Tab) ve 1 interaktif yan menü (Sidebar) olarak yapılandırılmıştır:

### 📱 1. İnteraktif Yan Menü (Sidebar)
* **Model Seçimi:** Sistemde hazır bulunan üç farklı eğitilmiş algoritma (Naive Bayes, Logistic Regression, SVM) arasında anlık geçiş yapmayı sağlar.
* **Dinamik Başarı Rozeti:** Proje metriklerine göre en yüksek genel başarıyı ve kütüphane kararlılığını gösteren Naive Bayes modeli seçildiğinde otomatik olarak `⭐ En iyi performanslı model` rozeti belirir.

### 📊 2. Fonksiyonel Sekmeler
1. **Demo (Anlık Tahmin Ekranı):**
   * **Serbest Metin Girişi:** Kullanıcının kendi cümlelerini yazarak canlı duygu analizi yapabileceği alan.
   * **Hızlı Test Örnekleri:** Uzun cümle yazmak istemeyen kullanıcılar için tek tıkla metin alanına transfer olan Pozitif, Negatif ve Nötr hazır örnek butonları.
   * **Gömülü Ön İşleme (Preprocessing) Göstergesi:** Arka planda anlık olarak temizlenen (küçük harfe çevirme, URL/sayı/noktalama temizliği, stopword filtreleme) metni, şeffaflık ilkesi gereği `Modele giren temizlenmiş metin` panelinde kod formatında gösterir.
   * **Güven Skoru Grafiği:** Tahmin sonucuna göre dinamik olarak renk değiştiren animasyonlu sonuç kartının hemen altında, modelin tahmin olasılıklarını anlık bir çubuk grafik (`Bar Chart`) ile yüzdelik oranlar halinde sunar.



## 🛠️ Kullanılan Teknolojiler

- Python 3.12
- Pandas, NumPy
- Scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, SVM)
- NLTK (Türkçe stopwords)
- Matplotlib, Seaborn, WordCloud
- Streamlit
- Hugging Face `datasets`

## 📚 Kaynakça

Aydoğan M, Kocaman V. "TRSAv1: A new benchmark dataset for classifying user
reviews on Turkish e-commerce websites." *Journal of Information Science*,
February 2022. doi:10.1177/01655515221074328
