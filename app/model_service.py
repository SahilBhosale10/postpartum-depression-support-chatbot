"""
Model service used by the FastAPI app.
Loads whichever model is available:
  1. Fine-tuned transformer (models/transformer/) if present  -- from Colab notebook
  2. Baseline TF-IDF + LogisticRegression (models/baseline_pipeline.joblib) -- fallback, always available

Exposes: predict_distress(text) -> {
    "label": int, "label_name": str, "confidence": float,
    "explanation": [ {"word": str, "weight": float}, ... ],   # top contributing words
    "recommendation": str,
    "needs_doctor": bool,
}
"""
import os
import re
import random
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_pipeline.joblib")
TRANSFORMER_DIR = os.path.join(BASE_DIR, "models", "transformer")

LABEL_NAMES = {0: "Normal", 1: "Mild Distress", 2: "Emotionally Distressed"}

# Reply text is organized as: {label: {tone: [variant, variant, ...]}} for
# labels 0 and 1 -- a mother can set a tone_preference ("warm", "practical",
# or "brief") and get repeat-avoiding variety within that tone, instead of
# the same fixed sentence every time.
#
# Label 2 (Emotionally Distressed) is intentionally tone-INVARIANT: this is
# the safety-critical doctor-referral message, so it does not follow the
# mother's cosmetic tone preference. It DOES still rotate through several
# differently-worded variants (so it doesn't feel robotic), all of which are
# firm, direct, and serious about the referral -- but never angry, judgmental,
# or shaming. A mother disclosing self-harm ideation, harm-to-baby thoughts,
# or abuse needs to feel safe continuing to talk, not scolded; anger from the
# bot at exactly that moment would discourage disclosure and works directly
# against the point of the safety net. "Firm" here means clear and serious
# about the referral, not hostile toward her.
RECOMMENDATIONS = {
    0: {
        "warm": [
            "You seem to be doing okay. Keep up the small routines that are helping you cope, "
            "and don't hesitate to check back in whenever you'd like to talk.",
            "It's good to hear things feel steady right now. Keep leaning on whatever's been "
            "working for you, and I'm here any time you want to talk something through.",
            "Glad to hear you're doing alright. Keep taking care of yourself the way you have "
            "been -- and I'm always here if that changes.",
        ],
        "practical": [
            "Things sound stable right now. Keep whatever routine has been working, and check "
            "back in if that changes.",
            "Sounds like you're managing well. Stick with what's been effective, and flag it "
            "here if anything shifts.",
            "No concerns from what you've shared. Keep your current routine going, and come "
            "back any time you want to check in.",
        ],
        "brief": [
            "Sounds like you're doing okay. Keep it up.",
            "Good to hear. Keep going the way you have been.",
            "You seem steady right now. I'm here if that changes.",
        ],
    },
    1: {
        "warm": [
            "It sounds like things have been a bit heavy lately. Try to carve out small moments "
            "of rest, lean on people around you, and consider light activities like a short walk "
            "or journaling. We're here if you want to talk more.",
            "That sounds like a tiring stretch. Be gentle with yourself, ask for help where you "
            "can, and try to fit in even a small break today. I'm here whenever you want to talk.",
            "I hear you -- this part is hard. Small things like a short walk, a bit of rest, or "
            "leaning on someone close to you can help take the edge off. Talk to me any time.",
        ],
        "practical": [
            "This sounds like a heavier stretch than usual. A short walk, a proper break, or "
            "asking someone for help with the baby for even half an hour can make a real "
            "difference. Check back in and let me know how it's going.",
            "A few things that tend to help here: getting a short break, some sleep if you can, "
            "and delegating a task or two today. Let me know how you're doing.",
        ],
        "brief": [
            "Sounds tough right now. Try to get a short break today if you can. I'm here.",
            "That's a hard patch. A bit of rest or help from someone nearby could help. Check "
            "back in.",
        ],
    },
    2: [
        "It sounds like you're going through something really difficult right now, and I want "
        "you to know you're not alone. It may help to talk to one of the doctors available on "
        "this platform. Would you like me to connect you with one?",
        "What you've shared sounds serious, and I want to make sure you get real support for "
        "it -- not just from me. I'd strongly recommend talking to one of the doctors here. "
        "Would you like me to connect you with one now?",
        "I'm taking this seriously, and I think it's important you talk to a doctor about this "
        "soon, not just me. Would you like me to connect you with someone on the platform right now?",
    ],
}


def get_recommendation(label: int, tone: str = None, previous_text: str = None) -> str:
    """
    Picks reply text for the given label, avoiding an exact repeat of the
    mother's last bot message where possible. `tone` only affects labels 0
    and 1 -- label 2 always uses its own fixed, serious variant pool
    regardless of tone preference (see module docstring above for why).
    """
    if label == 2:
        pool = RECOMMENDATIONS[2]
    else:
        tone_pools = RECOMMENDATIONS.get(label, RECOMMENDATIONS[0])
        pool = tone_pools.get(tone) or tone_pools["warm"]

    choices = [v for v in pool if v != previous_text] or pool
    return random.choice(choices)

USE_TRANSFORMER = os.path.isdir(TRANSFORMER_DIR) and os.path.exists(
    os.path.join(TRANSFORMER_DIR, "config.json")
)

_baseline_pipeline = None
_explainer = None


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOP_WORDS
_EXPLANATION_STOPWORDS = _SK_STOP_WORDS  # filter these out of "why" explanations even if the model scored them


def _load_baseline():
    global _baseline_pipeline, _explainer
    if _baseline_pipeline is None:
        _baseline_pipeline = joblib.load(BASELINE_MODEL_PATH)
        import shap
        tfidf = _baseline_pipeline.named_steps["tfidf"]
        clf = _baseline_pipeline.named_steps["clf"]
        # Linear explainer works directly on the TF-IDF sparse feature space
        background = tfidf.transform([""])  # cheap empty background
        _explainer = shap.LinearExplainer(clf, background)
    return _baseline_pipeline, _explainer


def _predict_baseline(text: str):
    pipeline, explainer = _load_baseline()
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = tfidf.transform([text])
    probs = clf.predict_proba(X)[0]
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])

    shap_values = explainer.shap_values(X)
    # shap_values shape: (1, n_features, n_classes) or list per class depending on version
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_label][0]
    else:
        class_shap = shap_values[0][:, pred_label] if shap_values.ndim == 3 else shap_values[0]

    feature_names = np.array(tfidf.get_feature_names_out())
    nonzero_idx = X.nonzero()[1]
    contributions = [(feature_names[i], float(class_shap[i])) for i in nonzero_idx]
    # drop pure-stopword unigrams from the explanation (a bigram containing a
    # stopword, e.g. "feel so", can still carry meaning, so only filter unigrams)
    contributions = [
        (w, v) for w, v in contributions
        if not (" " not in w and w in _EXPLANATION_STOPWORDS)
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_words = [{"word": w, "weight": round(v, 4)} for w, v in contributions[:8]]

    return pred_label, confidence, top_words


def _predict_transformer(text: str):
    """
    Uses the fine-tuned transformer from models/transformer/ (produced by
    notebooks/train_transformer_colab.ipynb). Explanation is computed with a
    leave-one-word-out (LOO / erasure) method: each word is removed in turn
    and we measure how much the predicted class's confidence drops -- words
    that matter most cause the biggest drop. This needs no extra dependency
    at inference time (SHAP is used for the richer analysis in the notebook).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    global _tf_tokenizer, _tf_model
    if "_tf_tokenizer" not in globals():
        globals()["_tf_tokenizer"] = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        globals()["_tf_model"] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
        globals()["_tf_model"].eval()

    tokenizer = globals()["_tf_tokenizer"]
    model = globals()["_tf_model"]

    def _probs(t: str):
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=-1)[0].numpy()

    base_probs = _probs(text)
    pred_label = int(np.argmax(base_probs))
    confidence = float(base_probs[pred_label])

    words = text.split()
    contributions = []
    for i in range(len(words)):
        ablated = " ".join(words[:i] + words[i + 1:])
        if not ablated.strip():
            continue
        probs_wo = _probs(ablated)
        drop = confidence - float(probs_wo[pred_label])  # positive => word supports the prediction
        contributions.append((words[i], drop))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_words = [{"word": w, "weight": round(v, 4)} for w, v in contributions[:8]]
    return pred_label, confidence, top_words


import re

DOCTOR_ACTION_WORDS = [
    "connect", "talk", "speak", "see", "need", "want", "book", "find",
    "show", "give", "appointment", "list", "referral", "refer", "meet",
]
AFFIRMATIVE_REPLIES = {
    "yes", "yes please", "yeah", "yep", "yup", "sure", "ok", "okay",
    "please", "sure please", "yes pls", "pls", "plz",
}


def wants_doctor(text: str, previous_bot_offered_referral: bool = False) -> bool:
    """Detects an explicit request to see/connect with a doctor, independent of
    the distress classifier. Two paths:
      1. The message mentions "doctor" alongside an action/intent word --
         matched flexibly (word presence, not rigid full-phrase matching) so
         small wording differences ("connect me with doctor" vs "...with a
         doctor") don't silently fail.
      2. A short affirmative reply ("yes", "sure", "okay") immediately after
         the bot itself offered a referral in the previous turn -- otherwise
         a mother saying "yes" to the bot's own question gets ignored.
    """
    t = (text or "").lower().strip()
    if not t:
        return False

    if re.search(r"\bdoctor(s)?\b", t) and any(w in t for w in DOCTOR_ACTION_WORDS):
        return True

    if previous_bot_offered_referral and t.strip(" .!") in AFFIRMATIVE_REPLIES:
        return True

    return False


# --- Safety net: this check CANNOT be silently overridden by anything else
# in this file (negation override, greeting shortcut, or the statistical
# classifier itself). If a message contains a clear disclosure of physical
# abuse, violence, or self-harm, it is ALWAYS treated as Emotionally
# Distressed with a doctor referral, regardless of what any other heuristic
# or model concludes. Getting this wrong in the "safe" direction (flagging
# something that turns out to be milder) is an acceptable cost; getting it
# wrong in the unsafe direction (missing a real disclosure) is not.
_SAFETY_PATTERN = re.compile(
    r"\b(beats?|hits?|hurts?|hurting|abus\w*|assault\w*|violent|violence|"
    r"strangl\w*|chok\w*|threatens?\w*)\b.{0,20}\b(me|myself)\b"
    r"|"
    r"\b(me|myself)\b.{0,20}\b(beats?|hits?|hurts?|abus\w*|assault\w*)\b"
    r"|"
    r"\b(husband|partner|boyfriend|spouse|wife|girlfriend)\b.{0,25}"
    r"\b(abus\w*|violent|violence|hits?|beats?|hurts?|threatens?\w*|controlling)\b"
    r"|"
    r"\b(abus\w*|violent|violence|hits?|beats?|hurts?|threatens?\w*)\b.{0,25}"
    r"\b(husband|partner|boyfriend|spouse|wife|girlfriend)\b"
    r"|"
    r"\b(suicidal|kill myself|end my life|end it all|want to die|"
    r"hurt (my baby|the baby)|harm (my baby|the baby))\b",
    re.IGNORECASE,
)


def _check_safety(text: str):
    """Runs against the RAW current message only -- never context-padded
    text -- so a safety signal is never diluted or masked by older turns."""
    match = _SAFETY_PATTERN.search(text or "")
    if not match:
        return None
    return {
        "label": 2,
        "confidence": 0.99,
        "explanation": [{"word": match.group(0), "weight": 1.0}],
    }


# Bag-of-words style classifiers (and even transformers, on short inputs) can
# be fooled by messages that NEGATE a distress term rather than express it --
# e.g. "I don't have any distress" contains the word "distress", which is a
# strong signal for the classifier, but the negation flips its actual meaning.
# This catches the common, direct denial pattern (negation word closely
# followed by a distress/emotion term) and overrides the classification to
# Normal. It is a narrow heuristic patch, not a general negation-handling
# system -- it only fires on a fairly tight word window to avoid overriding
# genuine distress messages that happen to also contain an unrelated "no".
#
# IMPORTANT: this must only ever run against the RAW current message, never
# a context-padded version of it -- otherwise a negation phrase from an
# OLDER turn ("I have no stress" two messages ago) can silently suppress the
# classification of a completely unrelated, serious NEW message. That was a
# real bug caught in testing, not a hypothetical.
_NEGATION_OVERRIDE_PATTERN = re.compile(
    r"\b(no|not|don'?t|never|zero|nothing wrong with)\b"
    r"(?:\s+\w+){0,3}\s+"
    r"\b(distress(ed)?|depress(ed|ion)?|hopeless|anxious|sad|struggl\w*|overwhelm\w*|stress(ed)?)\b",
    re.IGNORECASE,
)


def _check_negation_override(text: str, pred_label: int):
    """Returns an override dict if this message denies distress, else None."""
    if pred_label == 0:
        return None  # already Normal, nothing to override
    match = _NEGATION_OVERRIDE_PATTERN.search(text)
    if not match:
        return None
    return {
        "label": 0,
        "confidence": 0.75,
        "explanation": [
            {"word": match.group(1), "weight": 1.0},
            {"word": match.group(2), "weight": 1.0},
        ],
    }


# Short greetings ("hi", "good morning") carry almost no signal for a
# TF-IDF/transformer classifier trained on emotionally expressive text, so
# they tend to fall back to whatever the majority training class is (Mild
# Distress here) rather than being read as the neutral small talk they are.
# This only matches when the ENTIRE message is just a greeting -- a message
# like "hi, I've been feeling hopeless lately" must NOT be short-circuited,
# since the real content after the greeting still needs to be classified.
_GREETING_ONLY_PATTERN = re.compile(
    r"^\s*(hi+|hello+|hey+|yo+|howdy|greetings|sup|hiya|"
    r"good\s+(morning|afternoon|evening|night)|morning|evening|"
    r"hi\s+there|hey\s+there)\s*[!.,]*\s*$",
    re.IGNORECASE,
)


def _check_greeting(text: str):
    if _GREETING_ONLY_PATTERN.match(text):
        return {
            "label": 0, "confidence": 0.95,
            "explanation": [{"word": text.strip().lower(), "weight": 1.0}],
        }
    return None


# Questions asking what the system knows about the mother ("what do you know
# about my hobbies?", "do you know my background?") are meta/small-talk, not
# distress -- but they carry almost no signal for the classifier (same root
# problem as bare greetings), so it tends to fall back to Mild Distress. This
# short-circuits them to Normal and tags which topic was asked about, so the
# chat endpoint (which has access to the mother's stored profile fields, unlike
# this module) can answer the question directly instead of giving a generic
# distress-support reply.
_ABOUT_ME_HOBBIES_PATTERN = re.compile(
    r"\b(what do you know|do you know|remember|recall|what are)\b"
    r".{0,20}\b(about )?(my )?hobb(y|ies)\b",
    re.IGNORECASE,
)
_ABOUT_ME_BACKGROUND_PATTERN = re.compile(
    r"\b(what do you know|do you know|remember|recall|what is|what's)\b"
    r".{0,20}\b(about )?(my )?background\b",
    re.IGNORECASE,
)
_ABOUT_ME_GENERAL_PATTERN = re.compile(
    r"\bwhat (do you|did i tell you|have i told you)\b.{0,25}\babout (me|myself)\b"
    r"|\bdo you know (anything )?about me\b",
    re.IGNORECASE,
)


def _check_about_me(text: str):
    for pattern, topic in (
        (_ABOUT_ME_HOBBIES_PATTERN, "hobbies"),
        (_ABOUT_ME_BACKGROUND_PATTERN, "background"),
        (_ABOUT_ME_GENERAL_PATTERN, "general"),
    ):
        m = pattern.search(text or "")
        if m:
            return {
                "label": 0, "confidence": 0.9,
                "explanation": [{"word": m.group(0).strip(), "weight": 1.0}],
                "topic": topic,
            }
    return None


# Questions about the app/platform itself ("how does this app work?", "what
# can you do?", "what is this platform for?") are a distinct, common category
# of meta-question, separate from about-me questions. Relying purely on the
# retrained classifier to generalize to every possible phrasing of this isn't
# realistic with a dataset this size -- new wordings keep surfacing that
# weren't in the ~70 augmented examples (e.g. "how does this app work" vs.
# the trained "how does this chatbot work"). A direct pattern match is far
# more reliable for this well-defined, low-risk category than hoping the
# statistical model has seen that exact phrasing.
_APP_FAQ_PATTERN = re.compile(
    r"\bhow (does|do) (this|the) (app|platform|chatbot|bot|site|website)\b.{0,20}\bwork\b"
    r"|\bwhat does (this|the) (app|platform|chatbot|bot) do\b"
    r"|\bwhat is (this|the) (app|platform|website|site) for\b"
    r"|\bhow do i use (this|the) (app|platform|site)\b"
    r"|\bwhat can you do\b"
    r"|\bwhat is amara\b",
    re.IGNORECASE,
)


def _check_app_faq(text: str):
    m = _APP_FAQ_PATTERN.search(text or "")
    if not m:
        return None
    return {
        "label": 0, "confidence": 0.9,
        "explanation": [{"word": m.group(0).strip(), "weight": 1.0}],
    }


_APP_FAQ_ANSWERS = [
    "Amara is a private space for new mothers. You can talk to me any time about how you're "
    "feeling -- I'll check in gently, and if things ever feel like too much, I can connect you "
    "with a real doctor here. There's also a community space where mothers share and support "
    "each other. Your chat with me is private -- no one else can see it.",
    "This is a support space built for new mothers: you can chat with me about how you're doing, "
    "I'll flag it and offer to connect you with a doctor if things sound serious, and there's a "
    "community area separate from our private chat where mothers talk to each other. Everything "
    "in our chat stays between us.",
]


def answer_app_faq(previous_text: str = None) -> str:
    choices = [a for a in _APP_FAQ_ANSWERS if a != previous_text] or _APP_FAQ_ANSWERS
    return random.choice(choices)


def answer_about_me(topic: str, hobbies: str = None, background_info: str = None) -> str:
    """Builds a direct answer to an about-me meta-question using the mother's
    stored profile fields. Called from the chat endpoint (not predict_distress
    itself), since this module has no access to the database/user object."""
    hobbies = (hobbies or "").strip()
    background_info = (background_info or "").strip()

    if topic == "hobbies":
        if hobbies:
            return (f"You told me you enjoy {hobbies}. I use that to make a few of "
                     f"my suggestions feel a little more like you.")
        return "I don't have any hobbies on file for you yet -- you can add them to your profile any time."

    if topic == "background":
        if background_info:
            return (f'Here\'s what you shared with me: "{background_info}". '
                     f"It helps me understand your context a bit better.")
        return "I don't have any background info on file for you yet -- you can add it to your profile any time."

    # general "what do you know about me"
    parts = []
    if hobbies:
        parts.append(f"you enjoy {hobbies}")
    if background_info:
        parts.append(f'a bit about your background: "{background_info}"')
    if not parts:
        return ("I don't have anything extra on file about you beyond your account "
                 "details -- you can add your hobbies and background any time from your profile.")
    return ("Here's what I have on file: " + " and ".join(parts) +
            ". I use this to make my suggestions feel a bit more personal, but it's "
            "never used for anything beyond that.")


_HOBBY_NUDGE_TEMPLATES = {
    0: [
        "Since you enjoy {hobby}, maybe fit in a little time for that today too.",
        "Maybe find a few minutes for {hobby} today, alongside everything else.",
        "You mentioned {hobby} -- a little time with that today could be a nice addition.",
    ],
    1: [
        "You mentioned enjoying {hobby} -- even a few minutes of that might help take the edge off.",
        "If you can, a short bit of {hobby} today might help you breathe a little easier.",
        "Even a small dose of {hobby} today could take a bit of the pressure off.",
    ],
}


def personalize_recommendation(label: int, base_recommendation: str, hobbies: str = None,
                                previous_nudge: str = None) -> str:
    """
    Lightly tailors the recommendation text using a hobby/interest the mother
    gave at signup (e.g. "reading, painting"). This is cosmetic personalization
    only -- it never touches classification, confidence, or the safety/referral
    logic, and is deliberately skipped for Emotionally Distressed (label 2) so
    the doctor-referral message stays focused and uncluttered.

    `previous_nudge`, if given, is the exact nudge sentence used last time (not
    the whole message) -- lets the caller avoid repeating the same hobby line
    back-to-back, same idea as get_recommendation()'s repeat-avoidance.
    """
    hobby = (hobbies or "").strip()
    if not hobby or label not in _HOBBY_NUDGE_TEMPLATES:
        return base_recommendation

    # If multiple hobbies were given comma-separated, just use the first for
    # a natural-sounding single suggestion.
    first_hobby = hobby.split(",")[0].strip()
    if not first_hobby:
        return base_recommendation

    templates = _HOBBY_NUDGE_TEMPLATES[label]
    candidates = [t.format(hobby=first_hobby) for t in templates]
    choices = [c for c in candidates if c != previous_nudge] or candidates
    nudge = random.choice(choices)
    return f"{base_recommendation} {nudge}"


def predict_distress(text: str, raw_text: str = None) -> dict:
    """
    text     -- what actually gets fed to the classifier (may include
                prepended context for short/ambiguous messages).
    raw_text -- the mother's actual current message, unmodified. Defaults to
                `text` when not given separately. Safety and negation checks
                always run against raw_text, never the context-padded text,
                so older turns can never mask or dilute the current message.
    """
    text = (text or "").strip()
    raw_text = (raw_text if raw_text is not None else text).strip()
    if not text:
        return {
            "label": 0, "label_name": LABEL_NAMES[0], "confidence": 0.0,
            "explanation": [], "recommendation": "Please share a bit about how you're feeling.",
            "needs_doctor": False, "meta_topic": None,
        }

    greeting = _check_greeting(raw_text)
    about_me = None if greeting else _check_about_me(raw_text)
    app_faq = None if (greeting or about_me) else _check_app_faq(raw_text)
    if greeting:
        pred_label, confidence, top_words = greeting["label"], greeting["confidence"], greeting["explanation"]
    elif about_me:
        pred_label, confidence, top_words = about_me["label"], about_me["confidence"], about_me["explanation"]
    elif app_faq:
        pred_label, confidence, top_words = app_faq["label"], app_faq["confidence"], app_faq["explanation"]
    elif USE_TRANSFORMER:
        pred_label, confidence, top_words = _predict_transformer(text)
    else:
        pred_label, confidence, top_words = _predict_baseline(text)

    override = _check_negation_override(raw_text, pred_label)
    if override:
        pred_label = override["label"]
        confidence = override["confidence"]
        top_words = override["explanation"]

    # Safety check runs LAST and against raw_text only, so it can never be
    # suppressed by the greeting shortcut, the negation override, or the
    # classifier itself -- see _check_safety for why this matters.
    safety = _check_safety(raw_text)
    if safety:
        pred_label = safety["label"]
        confidence = safety["confidence"]
        top_words = safety["explanation"]

    if about_me and pred_label == 0:
        meta_topic = about_me["topic"]
    elif app_faq and pred_label == 0:
        meta_topic = "app_faq"
    else:
        meta_topic = None

    return {
        "label": pred_label,
        "label_name": LABEL_NAMES[pred_label],
        "confidence": round(confidence, 3),
        "explanation": top_words,
        "recommendation": get_recommendation(pred_label),
        "needs_doctor": pred_label == 2,
        "meta_topic": meta_topic,
    }