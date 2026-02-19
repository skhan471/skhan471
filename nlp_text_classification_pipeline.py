"""
End-to-end NLP text classification pipeline on the SMS Spam Collection dataset.

What this script does:
1) Downloads a public text classification dataset.
2) Cleans and preprocesses text (lowercase, punctuation removal, stopwords, tokenization, lemmatization).
3) Builds TF-IDF features and saves them to Excel, then reloads them.
4) Splits data into train/validation/test = 70/15/15.
5) Trains a Logistic Regression model and evaluates it.
6) Builds Word2Vec sentence embeddings, saves/reloads from Excel, trains another model, and evaluates it.
7) Prints a side-by-side comparison table for both approaches.
"""

from __future__ import annotations

import os
import re
import string
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------
# Configuration
# -------------------------------
SEED = 42
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = DATA_DIR / "smsspamcollection.zip"
RAW_PATH = DATA_DIR / "SMSSpamCollection"

TFIDF_EXCEL_PATH = ARTIFACTS_DIR / "sms_tfidf_features.xlsx"
W2V_EXCEL_PATH = ARTIFACTS_DIR / "sms_word2vec_features.xlsx"


# -------------------------------
# Step 1: Download and load data
# -------------------------------
def download_sms_spam_dataset() -> None:
    """Download and extract the SMS Spam Collection dataset from UCI."""
    if RAW_PATH.exists():
        print("Dataset already available locally.")
        return

    print("Downloading dataset...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    import zipfile

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    print(f"Dataset downloaded and extracted to: {DATA_DIR.resolve()}")


def load_dataset() -> pd.DataFrame:
    """Load dataset into a DataFrame with columns: label, text."""
    df = pd.read_csv(RAW_PATH, sep="\t", header=None, names=["label", "text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# -------------------------------
# Step 2: Text preprocessing
# -------------------------------
def setup_nltk_resources() -> None:
    """Download required NLTK resources once."""
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        nltk_download(res, quiet=True)


STOP_WORDS = None
LEMMATIZER = None


def initialize_text_tools() -> None:
    global STOP_WORDS, LEMMATIZER
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str, use_lemmatization: bool = True) -> str:
    """
    Apply:
      1) lowercasing
      2) punctuation removal
      3) tokenization
      4) stopword removal
      5) optional lemmatization
    Returns cleaned text string.
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    cleaned_tokens = []

    for tok in tokens:
        if not tok.isalpha():
            continue
        if tok in STOP_WORDS:
            continue
        if use_lemmatization:
            tok = LEMMATIZER.lemmatize(tok)
        cleaned_tokens.append(tok)

    return " ".join(cleaned_tokens)


# -------------------------------
# Step 3,4,5: TF-IDF + Save/Load Excel
# -------------------------------
def build_tfidf_features(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(max_features=2000)
    X_tfidf = vectorizer.fit_transform(df["clean_text"])  # sparse matrix

    feature_names = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)
    tfidf_df.insert(0, "label_num", df["label_num"].values)
    return tfidf_df


def save_and_reload_excel(features_df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    features_df.to_excel(file_path, index=False)
    reloaded_df = pd.read_excel(file_path)
    return reloaded_df


# -------------------------------
# Step 6: Train/Val/Test split
# -------------------------------
def split_train_val_test(features_df: pd.DataFrame):
    X = features_df.drop(columns=["label_num"])
    y = features_df["label_num"]

    # First split: train 70%, temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    # Second split: temp -> val 15%, test 15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# -------------------------------
# Step 7,8,9,10: Train + evaluate
# -------------------------------
def train_and_evaluate(
    model_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) -> dict:
    model = LogisticRegression(max_iter=3000, random_state=SEED)
    model.fit(X_train, y_train)

    # Validation predictions
    y_val_pred = model.predict(X_val)

    # Test predictions
    y_test_pred = model.predict(X_test)

    results = {
        "Model": model_name,
        "Validation Accuracy": accuracy_score(y_val, y_val_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Test Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Test Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "Test F1": f1_score(y_test, y_test_pred, zero_division=0),
        "Test Confusion Matrix": confusion_matrix(y_test, y_test_pred),
    }

    print(f"\n===== {model_name} =====")
    print(f"Validation Accuracy : {results['Validation Accuracy']:.4f}")
    print(f"Test Accuracy       : {results['Test Accuracy']:.4f}")
    print(f"Test Precision      : {results['Test Precision']:.4f}")
    print(f"Test Recall         : {results['Test Recall']:.4f}")
    print(f"Test F1-Score       : {results['Test F1']:.4f}")
    print("Confusion Matrix:")
    print(results["Test Confusion Matrix"])

    return results


# -------------------------------
# Word2Vec pipeline
# -------------------------------
def sentence_to_avg_embedding(tokens: list[str], model: Word2Vec, embedding_dim: int) -> np.ndarray:
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)


def build_word2vec_features(df: pd.DataFrame, embedding_dim: int = 100) -> pd.DataFrame:
    tokenized_sentences = [txt.split() for txt in df["clean_text"]]

    w2v_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=os.cpu_count() or 1,
        seed=SEED,
    )

    embeddings = np.vstack(
        [
            sentence_to_avg_embedding(tokens, w2v_model, embedding_dim)
            for tokens in tokenized_sentences
        ]
    )

    col_names = [f"w2v_{i}" for i in range(embedding_dim)]
    emb_df = pd.DataFrame(embeddings, columns=col_names)
    emb_df.insert(0, "label_num", df["label_num"].values)
    return emb_df


# -------------------------------
# Main execution
# -------------------------------
def main() -> None:
    print("Step 1: Downloading and loading dataset...")
    download_sms_spam_dataset()
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head(3), "\n")

    print("Step 2: Preprocessing text...")
    setup_nltk_resources()
    initialize_text_tools()
    df["clean_text"] = df["text"].apply(preprocess_text)
    print(df[["text", "clean_text"]].head(3), "\n")

    print("Step 3/4/5 (TF-IDF): Building features, saving to Excel, loading again...")
    tfidf_features = build_tfidf_features(df)
    tfidf_loaded = save_and_reload_excel(tfidf_features, TFIDF_EXCEL_PATH)
    print(f"TF-IDF feature dataset shape: {tfidf_loaded.shape}")
    print(f"Saved TF-IDF Excel: {TFIDF_EXCEL_PATH.resolve()}\n")

    print("Step 6: Splitting TF-IDF data into train/val/test = 70/15/15...")
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = split_train_val_test(tfidf_loaded)
    print(
        f"TF-IDF split sizes -> train: {len(X_train_t)}, val: {len(X_val_t)}, test: {len(X_test_t)}\n"
    )

    print("Step 7/8/9/10 (TF-IDF): Training and evaluation...")
    tfidf_results = train_and_evaluate(
        "Logistic Regression + TF-IDF",
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        X_test_t,
        y_test_t,
    )

    print("\nStep 3/4/5 (Word2Vec): Building embeddings, saving to Excel, loading again...")
    w2v_features = build_word2vec_features(df, embedding_dim=100)
    w2v_loaded = save_and_reload_excel(w2v_features, W2V_EXCEL_PATH)
    print(f"Word2Vec feature dataset shape: {w2v_loaded.shape}")
    print(f"Saved Word2Vec Excel: {W2V_EXCEL_PATH.resolve()}\n")

    print("Step 6: Splitting Word2Vec data into train/val/test = 70/15/15...")
    X_train_w, X_val_w, X_test_w, y_train_w, y_val_w, y_test_w = split_train_val_test(w2v_loaded)
    print(
        f"Word2Vec split sizes -> train: {len(X_train_w)}, val: {len(X_val_w)}, test: {len(X_test_w)}\n"
    )

    print("Step 7/8/9/10 (Word2Vec): Training and evaluation...")
    w2v_results = train_and_evaluate(
        "Logistic Regression + Word2Vec(avg)",
        X_train_w,
        y_train_w,
        X_val_w,
        y_val_w,
        X_test_w,
        y_test_w,
    )

    comparison_table = pd.DataFrame(
        [
            {
                "Model": tfidf_results["Model"],
                "Val Accuracy": round(tfidf_results["Validation Accuracy"], 4),
                "Test Accuracy": round(tfidf_results["Test Accuracy"], 4),
                "Precision": round(tfidf_results["Test Precision"], 4),
                "Recall": round(tfidf_results["Test Recall"], 4),
                "F1": round(tfidf_results["Test F1"], 4),
            },
            {
                "Model": w2v_results["Model"],
                "Val Accuracy": round(w2v_results["Validation Accuracy"], 4),
                "Test Accuracy": round(w2v_results["Test Accuracy"], 4),
                "Precision": round(w2v_results["Test Precision"], 4),
                "Recall": round(w2v_results["Test Recall"], 4),
                "F1": round(w2v_results["Test F1"], 4),
            },
        ]
    )

    print("\n===== Comparison Table (TF-IDF vs Word2Vec) =====")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
