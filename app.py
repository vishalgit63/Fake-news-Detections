from pathlib import Path
import json
import math
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import joblib
import pandas as pd
import streamlit as st

from utils import preprocess_text


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "fake_news_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CM_PATH = ARTIFACT_DIR / "confusion_matrix.png"
COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_article(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a complete article URL starting with http:// or https://.")

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"},
        timeout=15,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    content = soup.find("article") or soup.find("main") or soup.body
    text = content.get_text(" ", strip=True) if content else ""
    if len(text) < 30:
        raise ValueError("The page did not contain enough readable article text.")

    return title, text


def confidence_score(model, processed_text):
    """
    Returns a display score, not always a calibrated probability.
    LogisticRegression / NB: actual predict_proba value.
    Linear-margin models: sigmoid of decision score for UI only.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([processed_text])[0]
        pred = int(model.predict([processed_text])[0])
        return float(probs[pred]), "probability"

    if hasattr(model, "decision_function"):
        score = float(model.decision_function([processed_text])[0])
        p_fake = 1.0 / (1.0 + math.exp(-score))
        pred = int(model.predict([processed_text])[0])
        display = p_fake if pred == 1 else (1.0 - p_fake)
        return float(display), "margin-derived score"

    return None, None


bundle = load_bundle()
metrics = load_metrics()

st.title("📰 Fake News Detection")
st.caption(
    "TF-IDF + classical machine learning with stopword removal, "
    "stemming/lemmatization, hyperparameter tuning, and evaluation."
)

if bundle is None:
    st.error(
        "Model not found. Train the project first, for example:\n\n"
        "`python train.py --data data/news.csv --preprocess lemmatization --fake-label 1`"
    )
    st.stop()

model = bundle["model"]
preprocess_method = bundle["preprocess_method"]
best_model_name = bundle["best_model_name"]

with st.sidebar:
    st.header("Model Information")
    st.write(f"**Selected model:** {best_model_name}")
    st.write(f"**Preprocessing:** {preprocess_method}")
    st.write("**Features:** TF-IDF unigrams + bigrams")
    st.info(
        "This app predicts linguistic patterns in the supplied text. "
        "It does not independently verify claims against authoritative sources."
    )

tab1, tab2, tab3 = st.tabs(["🔎 Detect", "📊 Evaluation", "ℹ️ Method"])

with tab1:
    st.subheader("Check an article")
    source_url = st.text_input(
        "Article URL (optional)",
        placeholder="https://example.com/news/article",
    )
    title = st.text_input("Article title (optional)")
    article = st.text_area(
        "Paste the news article or claim (or provide a URL above)",
        height=280,
        placeholder="Paste enough text for the model to analyze...",
    )

    if st.button("Analyze News", type="primary", use_container_width=True):
        if source_url.strip():
            try:
                fetched_title, fetched_article = fetch_article(source_url.strip())
                title = title or fetched_title
                article = fetched_article
                st.info("Article text fetched from the provided URL.")
            except (requests.RequestException, ValueError) as error:
                st.error(f"Could not read that URL: {error}")
                st.stop()

        combined = f"{title} {article}".strip()
        if len(combined) < 30:
            st.warning("Please enter a longer headline/article for a meaningful prediction.")
        else:
            processed = preprocess_text(combined, preprocess_method)
            prediction = int(model.predict([processed])[0])
            label = bundle["labels"][prediction]

            score, score_type = confidence_score(model, processed)

            if prediction == 1:
                st.error("Prediction: FAKE / UNRELIABLE")
            else:
                st.success("Prediction: REAL / RELIABLE")

            if score is not None:
                st.metric(
                    "Model confidence",
                    f"{score * 100:.1f}%",
                    help=(
                        "For models with predict_proba this is a probability. "
                        "For margin-based models it is only a display score and is not calibrated."
                    ),
                )
                st.caption(f"Score type: {score_type}")

            with st.expander("View processed text"):
                st.write(processed if processed else "(No tokens remained after preprocessing.)")

            st.warning(
                "Use this prediction as a screening signal, not as definitive fact-checking. "
                "Important claims should be checked against reliable primary/independent sources."
            )

with tab2:
    st.subheader("Model evaluation")

    if metrics:
        best = metrics["results"][metrics["best_model"]]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{best['accuracy']:.3f}")
        c2.metric("Precision", f"{best['precision']:.3f}")
        c3.metric("Recall", f"{best['recall']:.3f}")
        c4.metric("F1-score", f"{best['f1']:.3f}")

        st.write(f"**Best model:** {metrics['best_model']}")
        st.write(f"**Dataset rows used:** {metrics['dataset_rows']:,}")
        st.write(f"**Test rows:** {metrics['test_rows']:,}")

    if COMPARISON_PATH.exists():
        df = pd.read_csv(COMPARISON_PATH)
        st.markdown("#### Model comparison")
        st.dataframe(
            df[["model", "accuracy", "precision", "recall", "f1", "cv_best_f1"]],
            use_container_width=True,
            hide_index=True,
        )

    if CM_PATH.exists():
        st.markdown("#### Confusion matrix")
        st.image(str(CM_PATH), width=520)

with tab3:
    st.subheader("How the system works")
    st.markdown(
        """
1. **Input:** title + article text.
2. **Cleaning:** lowercasing, URL/HTML/email removal, punctuation/digit removal.
3. **Stopword removal:** common English words are removed.
4. **Normalization:** stemming or lemmatization.
5. **Feature extraction:** TF-IDF using unigrams and bigrams.
6. **Models:** Logistic Regression, Linear SVM, Multinomial Naive Bayes, and Passive Aggressive.
7. **Hyperparameter tuning:** 5-fold `GridSearchCV` with F1-score.
8. **Evaluation:** accuracy, precision, recall, F1-score, classification report, and confusion matrix.
9. **Deployment:** the model with the best cross-validation F1-score is saved and loaded into this Streamlit app.
        """
    )
