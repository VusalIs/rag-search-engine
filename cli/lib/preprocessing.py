import string
from nltk.stem import PorterStemmer
from .config import STOPWORDS_PATH


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def _load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = _load_stopwords()
stemmer = PorterStemmer()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = [token for token in text.split() if token]
    filtered = [word for word in tokens if word not in STOPWORDS]
    return [stemmer.stem(word) for word in filtered]


def tokenize_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]
