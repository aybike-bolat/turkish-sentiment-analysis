"""
Türkçe Metin Duygu Analizi - Streamlit Demo Uygulaması

Çalıştırmak için:
    streamlit run app/streamlit_app.py
"""
import sys

# Windows konsolu (özellikle Türkçe/cp1254 kod sayfası) bazı emoji/özel
# karakterleri yazdıramadığı için ara sıra "UnicodeEncodeError" ile
# çökebiliyor. stdout/stderr'i buradan UTF-8'e zorlayıp bu tür karakterleri
# sessizce değiştirerek (crash etmeden) uygulamanın kararlı çalışmasını
# sağlıyoruz.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import json
import traceback
from pathlib import Path
import re
import os

import joblib
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords

BASE_DIR = Path(os.path.abspath(__file__)).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "images" # Görsellerin ve tablonun olduğu ana klasör

# ---- YENİ EKLENEN İÇE GÖMÜLÜ METİN TEMİZLEME FONKSİYONU ----
# NLTK Türkçe etkisiz kelimeler listesini garantiye alıyoruz
try:
    stop_words = set(stopwords.words('turkish'))
except Exception:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('turkish'))

# Özel stopword eklemeleri
custom_stopwords = {"bir", "fakat", "ancak", "ve", "veya"} 
stop_words.update(custom_stopwords)

def clean_text(text):
    """
    Türkçe duygu analizine uygun gömülü metin temizleme fonksiyonu.
    """
    if not isinstance(text, str):
        return ""
    
    # Türkçe harf duyarlılığına uygun küçük harfe çevirme (İ->i, I->ı)
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    
    # URL, E-posta ve Sosyal Medya etiketlerini temizleme
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\S+|#\S+', '', text)
    
    # Sayıları ve noktalama işaretlerini kaldırma
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Stopwords (etkisiz kelimeler) ve 2 karakterden kısa kelimeleri filtreleme
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    return " ".join(cleaned_words)
# -----------------------------------------------------------

st.set_page_config(
    page_title="Turkce Duygu Analizi",
    page_icon="💬",
    layout="wide",
)

LABEL_TR = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
LABEL_EMOJI = {0: "😠", 1: "😐", 2: "😊"}
LABEL_COLOR = {0: "#e74c3c", 1: "#f1c40f", 2: "#2ecc71"}

MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "SVM": "svm.joblib",
}

EXAMPLES = [
    "Bu ürün tam bir hayal kırıklığı, param boşa gitti.",
    "Kargo süresi normal, ürün açıklamada belirtildiği gibi.",
    "Harika bir alışveriş deneyimiydi, herkese tavsiye ederim!",
]

DEFAULT_TEXT = "Kargo çok hızlı geldi, ürün de tam beklediğim gibiydi, teşekkürler!"


@st.cache_resource(show_spinner="Modeller yükleniyor...")
def load_artifacts():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    meta = {}
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return vectorizer, models, meta


def predict(text: str, vectorizer, model):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None, None, cleaned
    X = vectorizer.transform([cleaned])
    
    # Düz sınıf tahminini alıyoruz (0, 1 veya 2)
    pred = int(model.predict(X)[0])

    proba = None
    # Modelin olasılık fonksiyonu varsa ve kütüphane versiyonu uyumluysa çalıştırır
    if hasattr(model, "predict_proba"):
        try:
            proba_values = model.predict_proba(X)
            if hasattr(proba_values, "toarray"):
                proba_values = proba_values.toarray()
            proba = dict(zip(model.classes_, proba_values[0]))
        except Exception:
            proba = None
            
    return pred, proba, cleaned


def render_result_card(pred, proba):
    label = LABEL_TR[pred]
    emoji = LABEL_EMOJI[pred]
    color = LABEL_COLOR[pred]

    confidence_txt = ""
    if proba:
        confidence = proba.get(pred, None)
        if confidence is not None:
            confidence_txt = f"<div style='font-size:16px;color:#666;margin-top:6px;'>Güven: %{confidence * 100:.1f}</div>"

    st.markdown(
        f"""
        <div style='padding:28px;border-radius:14px;background-color:{color}1A;
                    border:2px solid {color};text-align:center;margin-bottom:10px;'>
            <div style='font-size:42px;'>{emoji}</div>
            <div style='font-size:30px;font-weight:700;color:{color};margin-top:4px;'>{label}</div>
            {confidence_txt}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if proba:
        proba_df = pd.DataFrame(
            {"Sınıf": [LABEL_TR[c] for c in proba.keys()], "Olasılık": list(proba.values())}
        ).sort_values("Olasılık", ascending=False)
        st.bar_chart(proba_df.set_index("Sınıf"), height=200)


def _apply_example(example_text: str):
    st.session_state.input_text = example_text


def render_demo_tab(vectorizer, models, meta):
    if "input_text" not in st.session_state:
        st.session_state.input_text = DEFAULT_TEXT

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("#### Metninizi yazın")
        st.text_area(
            "metin_girisi",
            key="input_text",
            height=140,
            label_visibility="collapsed",
            placeholder="Örn: Ürün çok kaliteliydi, hızlı kargo için teşekkürler!",
        )

        st.caption("Hızlı denemek için bir örneğe tıklayın:")
        ex_cols = st.columns(len(EXAMPLES))
        for i, (col, ex) in enumerate(zip(ex_cols, EXAMPLES)):
            col.button(
                ex[:28] + ("…" if len(ex) > 28 else ""),
                key=f"example_{i}",
                on_click=_apply_example,
                args=(ex,),
                use_container_width=True,
            )

        analyze_clicked = st.button("🔍 Duyguyu Analiz Et", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Sonuç")
        result_box = st.container()

        if analyze_clicked:
            text = st.session_state.input_text
            if not text or not text.strip():
                with result_box:
                    st.warning("Lütfen önce bir metin girin.")
            else:
                try:
                    model_name = st.session_state.get("selected_model", meta.get("best_model"))
                    model = models[model_name]
                    with st.spinner("Analiz ediliyor..."):
                        pred, proba, cleaned = predict(text, vectorizer, model)
                    with result_box:
                        if pred is None:
                            st.warning(
                                "Metin temizleme sonrası anlamlı içerik kalmadı "
                                "(çok kısa veya sadece etkisiz kelimelerden oluşuyor olabilir)."
                            )
                        else:
                            render_result_card(pred, proba)
                            with st.expander("Modele giren temizlenmiş metin"):
                                st.code(cleaned if cleaned else "(boş)")
                except Exception as exc:  # noqa: BLE001
                    with result_box:
                        st.error(
                            "Tahmin sırasında beklenmeyen bir hata oluştu. "
                            "Lütfen metni değiştirip tekrar deneyin."
                        )
                        with st.expander("Hata detayı (geliştirici için)"):
                            st.code("".join(traceback.format_exception(exc)))
        else:
            with result_box:
                st.info("Sonuç burada görünecek.")


def render_comparison_tab():
    st.subheader("Model Karşılaştırma Sonuçları")
    comparison_path = FIG_DIR / "model_comparison.csv"
    
    # Eğer csv dosyası reports altındaysa kontrolü garantiye alalım
    if not comparison_path.exists():
        comparison_path = REPORTS_DIR / "model_comparison.csv"

    try:
        df = pd.read_csv(comparison_path)
        numeric_cols = [c for c in df.columns if c != "model"]
        st.dataframe(
            df.style.format({c: "{:.4f}" for c in numeric_cols}),
            width="stretch",
            hide_index=True,
        )
    except Exception:
        pass

    fig_path = FIG_DIR / "05_model_comparison.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Model Performans Karşılaştırması")

    st.markdown("#### Confusion Matrix'ler")
    cols = st.columns(3)
    for col, name in zip(cols, ["naive_bayes", "logistic_regression", "svm"]):
        cm_path = FIG_DIR / f"cm_{name}.png"
        if cm_path.exists():
            with col:
                st.image(str(cm_path), caption=name.replace("_", " ").title())


def render_dataset_tab():
    st.subheader("Veri Seti: TRSAv1")
    st.markdown(
        """
        **TRSAv1 (Turkish Sentiment Analysis v1)** veri seti, Türkçe e-ticaret
        sitelerindeki kullanıcı yorumlarından oluşan, akademik olarak yayınlanmış
        bir referans veri setidir.

        - **Kaynak:** Aydoğan M, Kocaman V. *"TRSAv1: A new benchmark dataset for
          classifying user reviews on Turkish e-commerce websites."*
          Journal of Information Science, 2022.
        - **Toplam örnek:** 150.000
        - **Sınıf dağılımı:** Pozitif 50.000 / Negatif 50.000 / Nötr 50.000
          (**tam dengeli**)
        - **Hugging Face:** `maydogan/Turkish_SentimentAnalysis_TRSAv1`
        """
    )

    fig_names = [
        ("01_class_distribution.png", "Sınıf Dağılımı"),
        ("02_char_length_distribution.png", "Karakter Uzunluğu Dağılımı"),
        ("03_word_count_distribution.png", "Kelime Sayısı Dağılımı"),
        ("04_wordclouds.png", "Kelime Bulutları"),
    ]
    for filename, caption in fig_names:
        path = FIG_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption)


def render_sidebar(models, meta):
    with st.sidebar:
        st.markdown("### Türkçe Duygu Analizi")
        st.caption("Yapay Zeka Staj Projesi")
        st.divider()

        available_models = list(models.keys())
        default_idx = (
            available_models.index(meta.get("best_model"))
            if meta.get("best_model") in available_models
            else 0
        )
        st.selectbox(
            "Kullanılacak model",
            available_models,
            index=default_idx,
            key="selected_model",
        )
        if meta.get("best_model") == st.session_state.get("selected_model"):
            st.caption("⭐ En iyi performanslı model")

        st.divider()
        st.markdown(
            "**Sınıflar:** Pozitif / Negatif / Nötr\n\n"
            "**Veri seti:** TRSAv1 (150.000 örnek, dengeli)\n\n"
            "**Yöntem:** TF-IDF + klasik ML"
        )


def main():
    st.title("Türkçe Metin Duygu Analizi Sistemi")
    st.caption("NLP tabanlı duygu sınıflandırma — Pozitif / Negatif / Nötr")

    if not (MODELS_DIR / "tfidf_vectorizer.joblib").exists():
        st.error(
            "Model dosyaları bulunamadı. Lütfen 'models/' klasöründe 'tfidf_vectorizer.joblib' dosyasının olduğundan emin olun."
        )
        return

    try:
        vectorizer, models, meta = load_artifacts()
    except Exception as exc:  # noqa: BLE001
        st.error("Modeller yüklenirken bir hata oluştu. Model dosyalarının bozulmamış olduğundan emin olun.")
        with st.expander("Hata detayı"):
            st.code("".join(traceback.format_exception(exc)))
        return

    if not models:
        st.error("Hiçbir eğitilmiş model bulunamadı. Lütfen 'models/' klasöründeki .joblib dosyalarını kontrol edin.")
        return

    render_sidebar(models, meta)

    tab1, tab2, tab3 = st.tabs(["Demo", "Model Karşılaştırması", "Veri Seti"])
    tab_renderers = [
        (tab1, render_demo_tab, (vectorizer, models, meta)),
        (tab2, render_comparison_tab, ()),
        (tab3, render_dataset_tab, ()),
    ]
    for tab, renderer, args in tab_renderers:
        with tab:
            try:
                renderer(*args)
            except Exception as exc:  # noqa: BLE001
                st.error("Bu bölüm yüklenirken beklenmeyen bir hata oluştu.")
                with st.expander("Hata detayı (geliştirici için)"):
                    st.code("".join(traceback.format_exception(exc)))


if __name__ == "__main__":
    main()