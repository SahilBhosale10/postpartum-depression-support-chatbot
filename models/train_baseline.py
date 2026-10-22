"""
Baseline explainable distress-detection model.
Runs fully offline (no pretrained transformer weights needed) so it works
end-to-end in any environment, and is what the FastAPI app uses by default.

Pipeline: TF-IDF (1-2 grams) -> Logistic Regression (class_weight=balanced)
Explainability: SHAP LinearExplainer on the TF-IDF/LogReg pipeline, so every
prediction returns the top words that pushed it towards its predicted class.

For the transformer-based model (BioBERT/RoBERTa) required by the project
title, see notebooks/train_transformer_colab.ipynb -- run that on Google
Colab (needs internet + GPU to download pretrained weights) and drop the
resulting model into models/transformer/ ; app/model_service.py will
auto-detect and use it instead of this baseline.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Paths are resolved relative to this file's location, so this script works
# no matter where the project folder is placed on disk.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_pipeline.joblib")

LABEL_NAMES = {0: "Normal", 1: "Mild Distress", 2: "Emotionally Distressed"}

# Standard English stopwords, MINUS negation words that flip meaning
# (removing "not"/"can't" etc. would break distress detection).
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
KEEP_WORDS = {
    "not", "no", "never", "nor", "none", "cannot", "n't", "without",
    "alone", "nobody", "nothing", "myself",
}
SAFE_STOP_WORDS = sorted(ENGLISH_STOP_WORDS - KEEP_WORDS)

df = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        stop_words=SAFE_STOP_WORDS,
    )),
    ("clf", LogisticRegression(
        max_iter=1000, class_weight="balanced", C=2.0, random_state=42
    )),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(pipeline, MODEL_PATH)
print("\nSaved model to", MODEL_PATH)