# Amara — Explainable Transformer-Based PPD Detection Platform

An explainable NLP system that detects emotional distress in new mothers'
chat text, explains its reasoning, and connects severely distressed mothers
with doctors — plus a private chat, a mothers' community, and admin oversight.

## 1. Project structure

```
ppd_project/
├── data/
│   ├── prepare_dataset.py       # cleans raw data, adds control class + severity labels
│   └── processed_dataset.csv    # generated output (3,004 rows, 3 classes)
├── models/
│   ├── train_baseline.py        # trains the TF-IDF + Logistic Regression baseline
│   └── baseline_pipeline.joblib # trained baseline model (used by the app by default)
├── notebooks/
│   └── train_transformer_colab.ipynb   # fine-tunes BioBERT/RoBERTa on Colab (needs GPU+internet)
├── app/
│   ├── main.py            # FastAPI app: auth, chat, community, doctor, admin endpoints
│   ├── database.py        # SQLAlchemy models (SQLite)
│   ├── auth.py             # JWT auth, password hashing, role-based access control
│   ├── model_service.py    # loads model (transformer if present, else baseline) + explains predictions
│   └── templates/index.html # frontend demo UI (vanilla JS, no build step)
├── seed.py                 # creates admin + demo doctor accounts
└── requirements.txt
```

## 2. The dataset problem (read this first)

The provided `sample_dataset_for_ppd.xlsx` contained **2,254 rows, all labeled
identically as `"postpartum"`** — i.e. only a single class, all distress text,
with no non-distressed control examples and no severity levels. A classifier
trained directly on that would just always predict "postpartum" — no signal.

`data/prepare_dataset.py` addresses this with two documented, standard
weak-supervision steps, **plus a blend of a real, human-labeled dataset**
(disclose all of this clearly in your report's methodology section — normal
practice, but it must be stated, not hidden):

1. **Control class ("Normal")** — template-generated, non-distressed
   motherhood statements (750 examples) to give the model a true negative
   class to learn against.
2. **Severity split (pseudo-labeled)** — the original PPD-specific text is
   split into *Mild Distress* vs *Emotionally Distressed* using VADER
   sentiment score plus a curated lexicon of severe-distress phrases.
3. **Real, human-labeled data blend** — 965 additional distress-class rows
   from a public, peer-reviewed dataset: *"Early Identification of Depression
   Severity Levels on Reddit Using Ordinal Classification"* (The Web
   Conference '22), sourced from `usmaann/Depression_Severity_Dataset` on
   GitHub. This dataset has genuine human-annotated severity labels
   (minimum/mild/moderate/severe), mapped into our mild/severe classes. It
   is **not postpartum-specific** — general depression-severity Reddit
   posts — but it is real ground truth, not a pseudo-label, and meaningfully
   strengthens the model's ability to generalize beyond the original
   template-like PPD text. Cite this source in your references.

Final label scheme: `0=Normal, 1=Mild Distress, 2=Emotionally Distressed`.

**Before your final submission**, if you can get a properly labeled dataset
(even a few hundred real non-distressed posts), swap it in — re-run
`prepare_dataset.py` pointed at it and everything downstream (training, app)
will pick it up automatically. Mention in your report that the current
control class is bootstrapped, not naturally collected.

## 3. The two models

### Baseline (runs anywhere, no GPU/internet needed) — used by the app by default
TF-IDF (1-2 grams, English stopwords removed except negators) + Logistic
Regression, explained via SHAP's LinearExplainer (returns the exact words
that pushed a prediction towards its class).
**Test accuracy: 79%, macro F1: 0.78**, on the blended real+pseudo-labeled
dataset (3,990 rows) — a more honest number than an earlier, purely synthetic
version of this dataset which scored 86.5% but on easier, template-only text.

### Transformer (satisfies your project's "transformer-based" requirement)
`notebooks/train_transformer_colab.ipynb` fine-tunes `mental/mental-roberta-base`
(a domain-adapted RoBERTa for mental-health text; BioBERT is offered as a
swap-in alternative) on the same processed dataset, with SHAP-based
explainability. **This must be run on Google Colab** (Runtime → GPU) because
it needs internet access to download pretrained weights — something a fully
offline sandbox cannot do. Takes about 10-15 minutes on a free Colab GPU.

After training, download `transformer_model.zip` from the notebook, unzip it
into `models/transformer/` in this project, and restart the app —
`app/model_service.py` auto-detects it and switches from the baseline to the
fine-tuned transformer automatically, no code changes needed.

## 4. Running the app

```bash
pip install -r requirements.txt
python3 data/prepare_dataset.py      # regenerate processed_dataset.csv (optional, already included)
python3 models/train_baseline.py     # retrain baseline (optional, already included)
python3 seed.py                      # creates admin@ppd.com / admin123 and two demo doctors
uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser. Sign up as a mother (or
doctor), or log in as the seeded admin/doctor accounts.

Demo accounts (after running `seed.py`):
- Admin: `admin@ppd.com` / `admin123`
- Doctor: `anika@ppd.com` / `doctor123`
- Doctor: `marcus@ppd.com` / `doctor123`

## 5. Features implemented

- **Auth & sessions**: JWT-based login/signup, role-based access control
  (mother / doctor / admin), 60-minute session expiry.
- **Private chat**: mother ↔ chatbot only; messages are filtered by
  `mother_id` server-side so no mother can ever see another's chats.
- **Explainable distress detection**: every bot reply includes the predicted
  level (Normal / Mild / Emotionally Distressed), a confidence score, and the
  top words that drove the prediction.
- **Doctor recommendation**: when a mother is flagged "Emotionally
  Distressed," a referral is automatically created and surfaced on the
  doctor's dashboard; doctors can accept it.
- **Community**: a separate, shared space where mothers can post and reply —
  distinct from the private 1:1 chat.
- **Admin dashboard**: platform-wide stats (mothers, doctors, messages,
  flagged-distress count, pending referrals) and a user list.

## 6. Known limitations / what to say if asked in a viva

- The control/severity labels are **pseudo-labels** (lexicon + sentiment
  derived, or template-generated), not human-annotated ground truth — be
  upfront about this; it's a limitation of the provided dataset, not a flaw
  in the approach.
- The transformer model needs a separate Colab run because this build
  environment has no internet access to Hugging Face's model hub — the
  baseline (TF-IDF + LogisticRegression) is what actually powers the live
  app unless you complete that step.
- The community board has no moderation/reporting flow yet — a natural
  "future work" item.
- For a production system, distress detection results and referral triggers
  should be reviewed with a licensed clinician before being used for any
  real safety-critical decision. This project is a research/academic
  prototype, not a clinical diagnostic tool.