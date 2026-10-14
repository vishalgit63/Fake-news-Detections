import re
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=1)
def ensure_wordnet():
    """Ensure NLTK WordNet resources are available."""
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def basic_clean(text: str) -> str:
    """Lowercase and remove URLs, HTML, emails, digits, and punctuation."""
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, method: str = "lemmatization") -> str:
    """
    Preprocess text using:
    1. cleaning
    2. stopword removal
    3. stemming OR lemmatization

    method: 'stemming', 'lemmatization', or 'none'
    """
    cleaned = basic_clean(text)
    tokens = [t for t in cleaned.split() if len(t) > 1 and t not in STOPWORDS]

    if method == "stemming":
        tokens = [STEMMER.stem(t) for t in tokens]
    elif method == "lemmatization":
        ensure_wordnet()
        # Verb then noun lemmatization gives a simple general-purpose normalization.
        tokens = [LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(t, pos="v"), pos="n") for t in tokens]
    elif method != "none":
        raise ValueError("method must be 'stemming', 'lemmatization', or 'none'")

    return " ".join(tokens)
