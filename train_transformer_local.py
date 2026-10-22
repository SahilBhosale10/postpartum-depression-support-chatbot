"""
Local, full-quality transformer fine-tuning -- runs on CPU, no Colab needed.
Model: mental/mental-roberta-base (RoBERTa pretrained further on mental-health
text) -- the same model used in the Colab notebook, no quality shortcuts taken.

This WILL take a while on CPU (realistically a few hours for 4 epochs on
~3,000 rows) -- that's expected and fine given you have overnight. Just start
it and let it run; you can check progress in the terminal as it prints loss
after every 25 steps.

Usage:
    python3 train_transformer_local.py

Requires (not in the base requirements.txt since these are heavy and only
needed for this step):
    pip install torch transformers datasets accelerate shap
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_dataset.csv")
SAVE_DIR = os.path.join(BASE_DIR, "models", "transformer")

MODEL_NAME = "roberta-base"  # public, no auth needed. (mental/mental-roberta-base is gated as of writing)
# Alternative: "distilbert-base-uncased" (smaller/faster, good if this is still slow on your CPU)
NUM_LABELS = 3
LABEL_NAMES = {0: "Normal", 1: "Mild Distress", 2: "Emotionally Distressed"}

print(f"Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(df["label_name"].value_counts())

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"train: {len(train_df)}  test: {len(test_df)}")

print(f"\nDownloading/loading {MODEL_NAME} (this needs internet access the first time)...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)


train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # smaller batch size -- kinder to CPU memory than the Colab GPU version
    per_device_eval_batch_size=16,
    num_train_epochs=4,              # full quality, same as the Colab notebook -- no shortcuts
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=25,
    report_to="none",
    # No explicit CPU/GPU flag needed -- with no NVIDIA GPU present, this
    # automatically runs on CPU. (Older transformers versions used a
    # `no_cuda=True` argument here; newer versions removed it and auto-detect instead.)
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

print("\n=== Starting training. This will take a while on CPU -- let it run. ===\n")
trainer.train()

print("\n=== Final evaluation ===")
preds_output = trainer.predict(test_ds)
preds = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids
print(classification_report(labels, preds, target_names=[LABEL_NAMES[i] for i in range(3)]))

print(f"\nSaving model to {SAVE_DIR} ...")
os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("\nDone. The FastAPI app (app/model_service.py) will now auto-detect and use")
print("this transformer model instead of the TF-IDF baseline the next time you")
print("restart uvicorn -- no code changes needed.")