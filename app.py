"""
app.py — Streamlit demosu.

Tek sayfa: text gir → 4 sınıflama tahmini + House BOT cevabı + analitik grafikler.

Çalıştırma:
    venv/bin/python -m streamlit run app.py
"""
import sys
from pathlib import Path

import joblib
import streamlit as st
from scipy.sparse import csr_matrix, hstack


ROOT = Path(__file__).parent
# HouseBot class'ı joblib unpickle için import'lanır
sys.path.insert(0, str(ROOT / "src"))
from house_bot import HouseBot, HouseReply  # noqa: E402


TARGETS = ["Intent", "diagnosis_stage", "Emotion", "Sarcasm"]
TARGET_LABELS = {
    "Intent": "Konuşma Niyeti (Intent)",
    "diagnosis_stage": "Tanı Aşaması (diagnosis_stage)",
    "Emotion": "Duygu (Emotion)",
    "Sarcasm": "Sarkazm (binary)",
}


@st.cache_resource
def load_models() -> dict:
    out = {}
    for t in TARGETS:
        out[t] = joblib.load(ROOT / "models" / f"{t}_model.joblib")
    return out


@st.cache_resource
def load_bot() -> HouseBot:
    return joblib.load(ROOT / "models" / "house_bot.joblib")


def featurize(text: str, bundle: dict):
    q = text.count("?")
    ex = text.count("!")
    Xc = bundle["char_vec"].transform([text])
    Xw = bundle["word_vec"].transform([text])
    Xe = csr_matrix([[q, ex]], dtype=float)
    return hstack([Xc, Xw, Xe]).tocsr()


def render_predictions(models: dict, text: str) -> None:
    st.subheader("Sınıflama Tahminleri")
    for target in TARGETS:
        bundle = models[target]
        X = featurize(text, bundle)
        probs = bundle["model"].predict_proba(X)[0]
        labels = bundle["label_encoder"].classes_
        ranked = sorted(zip(labels, probs), key=lambda x: -x[1])[:3]

        st.markdown(f"**{TARGET_LABELS[target]}**")
        for cls, p in ranked:
            st.progress(float(p), text=f"{cls}  —  {p * 100:.1f}%")
        st.write("")


def render_bot(bot: HouseBot, text: str) -> None:
    st.subheader("House BOT — En Yakın Replikler")
    replies = bot.reply(text, top_k=3)
    if not replies:
        st.info("Bir replik yazın.")
        return
    for i, r in enumerate(replies, 1):
        st.markdown(f"**{i}.** *\"{r.text}\"*")
        st.caption(f"Sezon {r.season} · Bölüm {r.episode} · benzerlik {r.similarity:.2f}")
        st.write("")


def render_about() -> None:
    with st.expander("Proje hakkında"):
        st.markdown("""
**BIM432 Doğal Dil İşleme Projesi**

- **Veri seti:** House MD Türkçe diyalog (7282 → 6954 satır, 7 aşamalı preprocess)
- **Hedef değişkenler:** Intent (7+diğer), diagnosis_stage (6+diğer), Emotion (5+diğer), Sarcasm (binary)
- **Model:** TF-IDF (char 2-5 + word 1-2) + ekstra feature (q_count, ex_count) → Logistic Regression
- **House BOT:** TF-IDF + cosine similarity üzerinden 2463 House repliği içinde retrieval
- **Bölüm bazlı split** (GroupShuffleSplit) — aynı bölüm hem train hem test'te değil

**Etik notları:** Bu sistem tıbbi tavsiye veremez, sadece kurgusal diyalog analizidir.
Veri kaynağı dizi senaryolarıdır, gerçek hasta verisi içermez.
""")


def main() -> None:
    st.set_page_config(
        page_title="House MD NLP",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("House MD — Türkçe NLP Analizi")
    st.caption("BIM432 Doğal Dil İşleme Projesi  ·  "
               "Ahmet Şafak YILDIRIM, Ogün ŞAHİN, Muhammet Berşan KURTCEPHE")
    st.divider()

    models = load_models()
    bot = load_bot()

    default = "Bu hasta lupus olabilir, MR çekelim ve antibiyotik başlatalım."
    text = st.text_area(
        "Bir replik yazın (Türkçe):",
        value=default,
        height=90,
    )
    analyze = st.button("Analiz Et", type="primary")

    if analyze and text.strip():
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            render_predictions(models, text)
        with col2:
            render_bot(bot, text)

    st.divider()
    render_about()


if __name__ == "__main__":
    main()
