"""
Data preparation pipeline for the PPD (Postpartum Depression) detection project.

Input : sample_dataset_for_ppd.xlsx  (2254 rows, ALL labeled "postpartum" -> distress text only)
Output: processed_dataset.csv with columns [text, label, label_name]

label mapping:
    0 -> Normal            (no distress)
    1 -> Mild Distress      (some stress but coping)
    2 -> Emotionally Distressed (severe distress, needs support)

Because the raw dataset had a SINGLE label for every row, this script:
  1. Cleans and deduplicates the raw distress text.
  2. Generates a "Normal" (non-distressed) control class from template-based,
     diverse, realistic postpartum statements that reflect a mother coping well.
     NOTE: this is a documented bootstrapping step (common when only a positive
     class exists) -- clearly disclose this in your report/methodology section.
  3. Splits the raw distress text into "Mild" vs "Emotionally Distressed" using
     VADER sentiment score + a curated PPD severe-distress keyword lexicon
     (hopelessness, self-worth, intrusive/dark thoughts, inability to function).
     This is a standard weak-supervision / pseudo-labeling approach used when
     graded severity labels are unavailable.
"""
import os
import re
import random
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

random.seed(42)

# Paths are resolved relative to this file's location, so this script works
# no matter where the project folder is placed on disk.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "sample_dataset_for_ppd.xlsx")  # place your raw file here to regenerate
REAL_LABELED_PATH = os.path.join(BASE_DIR, "data", "real_labeled_dataset.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed_dataset.csv")

# ---------------------------------------------------------------------------
# 1. Load + clean raw distress text
# ---------------------------------------------------------------------------
df = pd.read_excel(RAW_PATH)
df.columns = ["text", "label"]
df = df[["text"]].copy()

def clean_text(t: str) -> str:
    t = str(t).strip()
    t = re.sub(r"http\S+|www\.\S+", " ", t)          # urls
    t = re.sub(r"@\w+", " ", t)                        # mentions
    t = re.sub(r"\s+", " ", t)                          # extra whitespace
    return t.strip()

df["text"] = df["text"].apply(clean_text)
df = df[df["text"].str.len() > 15]
df = df.drop_duplicates(subset="text").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 1b. Augment with common, direct self-report phrasings that real users type
#     (e.g. "I feel depressed") but that rarely appear verbatim in forum/Reddit
#     style writing, which tends toward descriptive rather than diagnostic language.
#     Without these, the model has literally zero signal for very common words
#     like "depressed", "sad", "crying", "anxious" -- a real gap worth closing.
# ---------------------------------------------------------------------------
direct_phrasings = [
    "I am feeling very depressed", "I feel so depressed lately", "I think I am depressed",
    "I've been feeling depressed since the baby was born", "I feel really sad all the time",
    "I can't stop crying today", "I cry every day and I don't know why",
    "I feel so anxious and depressed", "I am so sad and I don't know how to fix it",
    "I feel numb and depressed most days", "I think I might have postpartum depression",
    "I feel very low and depressed", "I'm struggling with depression after having my baby",
    "I feel sad, tired, and depressed all the time", "I am crying again, I feel so depressed",
    "I feel anxious all the time and it's exhausting", "I feel so sad I don't want to get out of bed",
    "I'm scared I might have postpartum depression", "I feel depressed and don't enjoy anything anymore",
    "Everything feels sad and heavy lately", "I feel so low, nothing makes me happy anymore",
]
augment_df = pd.DataFrame({"text": direct_phrasings})
df = pd.concat([df, augment_df], ignore_index=True).drop_duplicates(subset="text").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Severity split for the distress class (Mild vs Emotionally Distressed)
# ---------------------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

SEVERE_LEXICON = [
    "hopeless", "worthless", "no point", "can't cope", "cant cope",
    "dark thoughts", "hurt myself", "hurt my baby", "want to disappear",
    "can't go on", "cant go on", "giving up", "no one would notice",
    "empty inside", "drowning", "failure as a mother", "bad mother",
    "don't want to be here", "dont want to be here", "numb", "trapped",
    "suicidal", "end it", "can't take it anymore", "cant take it anymore",
    "isolated", "losing myself", "no support", "alone in this",
    "depressed", "depression", "crying every day", "cry every day",
    "can't stop crying", "cant stop crying", "so sad", "very low",
]

def severe_hits(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in SEVERE_LEXICON if kw in t)

rows = []
for t in df["text"]:
    vs = analyzer.polarity_scores(t)
    compound = vs["compound"]
    hits = severe_hits(t)
    # Severe: strongly negative sentiment AND/OR explicit severe-distress language
    if compound <= -0.55 or hits >= 2:
        label, name = 2, "Emotionally Distressed"
    else:
        label, name = 1, "Mild Distress"
    rows.append({"text": t, "label": label, "label_name": name})

distress_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. Generate "Normal" (non-distressed) control class
# ---------------------------------------------------------------------------
subjects = ["my baby", "the baby", "my little one", "our son", "our daughter", "my newborn"]

openers = [
    "I'm exhausted some days, but honestly {subj} makes me smile every single morning.",
    "Motherhood has been challenging to adjust to, but I feel supported and we're figuring it out together.",
    "Today {subj} laughed for the first time and it made all the sleepless nights worth it.",
    "I was nervous about becoming a mom, but I'm settling into a routine and feeling more confident.",
    "We had a rough night with feeding, but my partner and I tag-teamed it and I still feel like myself.",
    "I joined a local moms' group and it's been so nice to share tips and laugh about newborn life.",
    "I'm tired but genuinely happy - {subj} is healthy and I'm getting the hang of this.",
    "Some days are harder than others, but overall I feel grateful and connected to {subj}.",
    "I took a walk outside with the stroller today and it really lifted my mood.",
    "My mother-in-law came to help this week and it gave me time to rest and recharge.",
    "I finally got a full night's sleep and I feel like a completely different person today.",
    "Balancing work and {subj} is tricky, but my manager has been understanding and it's manageable.",
    "I cried happy tears watching {subj} sleep peacefully last night.",
    "We started a bedtime routine and it's made evenings so much calmer for all of us.",
    "I felt overwhelmed the first few weeks, but things are steadily getting easier for me.",
    "My friends threw me a small get-together and it reminded me I still have a life outside motherhood.",
    "I've been journaling every night and it helps me process the changes; I feel okay overall.",
    "Breastfeeding was hard to figure out at first, but we found our rhythm and I feel proud.",
    "I still get tired, but I've learned to ask for help and it's made a huge difference.",
    "Watching {subj} grow every day fills me with so much joy, even on the messier days.",
    "I talked to my doctor about my worries and she reassured me everything is normal.",
    "My partner has been amazing with night feeds, and I actually feel rested this week.",
    "I went back to my yoga class and it's been great for clearing my head.",
    "I was worried about bonding with {subj}, but it happened naturally and I feel connected now.",
    "Life is busier than ever, but I feel like I'm handling it well with support around me.",
    "I treated myself to a coffee alone today while my mom watched {subj}, and it recharged me.",
    "The first month was a blur, but now I feel like I've found my footing as a mom.",
    "I love this new chapter of my life, even with the chaos - I feel content most days.",
    "My sister checks in on me every day and it makes such a big difference in how I feel.",
    "I used to worry constantly, but now I trust myself more as a parent and feel calmer.",
]

closers = [
    "It's not always easy, but I feel like I'm coping well.",
    "Overall I'd say I'm doing okay and enjoying this time.",
    "I feel positive about how things are going.",
    "I'm grateful for the support system I have around me.",
    "Some days are tiring, but I genuinely feel happy.",
    "I feel like myself again, just with a little less sleep.",
    "I'm proud of how far we've come as a family.",
    "It feels manageable, and I'm not doing this alone.",
    "I feel hopeful and excited about what's ahead.",
    "",  # allow some without a closer for length variety
]

normal_rows = []
n_needed = int(len(distress_df) * 0.75)  # keep dataset reasonably balanced
seen = set()
attempts = 0
while len(normal_rows) < n_needed and attempts < n_needed * 20:
    attempts += 1
    subj = random.choice(subjects)
    opener = random.choice(openers).format(subj=subj)
    closer = random.choice(closers)
    text = f"{opener} {closer}".strip()
    if text in seen:
        continue
    seen.add(text)
    normal_rows.append({"text": text, "label": 0, "label_name": "Normal"})

normal_df = pd.DataFrame(normal_rows).drop_duplicates(subset="text").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. Combine, shuffle, save
# ---------------------------------------------------------------------------
final_df = pd.concat([normal_df, distress_df], ignore_index=True)

# ---------------------------------------------------------------------------
# 5. Blend in a REAL, human-labeled dataset (not pseudo-labeled) for extra
#    ground-truth signal and better generalization.
#
#    Source: "Early Identification of Depression Severity Levels on Reddit
#    Using Ordinal Classification" (The Web Conference '22), Zirikly et al.
#    (usmaann/Depression_Severity_Dataset on GitHub). 3,553 Reddit posts,
#    human-labeled across 4 clinical severity levels: minimum / mild /
#    moderate / severe -- analogous to PHQ-9 depression severity banding.
#
#    Mapped to our 3-class scheme:
#      minimum            -> 0 Normal
#      mild                -> 1 Mild Distress
#      moderate            -> 1 Mild Distress  (still below clinical threshold needing urgent referral)
#      severe              -> 2 Emotionally Distressed
#
#    This is NOT postpartum-specific (general depression-severity text), but
#    it is real, human-annotated ground truth, which is a meaningfully
#    stronger signal than the pseudo-labels above -- disclose this blend
#    explicitly in your report's methodology/dataset section.
# ---------------------------------------------------------------------------
if os.path.exists(REAL_LABELED_PATH):
    real_df = pd.read_csv(REAL_LABELED_PATH)
    real_df["text"] = real_df["text"].apply(clean_text)
    real_df = real_df[real_df["text"].str.len() > 15].drop_duplicates(subset="text")

    severity_map = {"minimum": 0, "mild": 1, "moderate": 1, "severe": 2}
    real_df["label"] = real_df["label"].map(severity_map)
    real_df = real_df.dropna(subset=["label"])
    real_df["label"] = real_df["label"].astype(int)

    # Only take the DISTRESS classes (mild/moderate/severe) from the real dataset.
    # Its "minimum" (non-depressed) class is generic Reddit content, not
    # maternal-specific, and diluted the model's ability to recognize normal,
    # happy postpartum language when tested -- so for the "Normal" class we
    # keep only the maternal-specific synthetic data above, and use this real
    # dataset purely to strengthen the harder mild/moderate/severe boundary
    # with genuine, human-labeled severity variety.
    real_df = real_df[real_df["label"] != 0]
    real_df["label_name"] = real_df["label"].map({1: "Mild Distress", 2: "Emotionally Distressed"})

    final_df = pd.concat([final_df, real_df[["text", "label", "label_name"]]], ignore_index=True)
    print(f"Blended in {len(real_df)} real, human-labeled distress rows from the public severity dataset.")
else:
    print("NOTE: real_labeled_dataset.csv not found -- skipping the real-data blend "
          "(dataset will use only pseudo-labeled/synthetic data). See README for how to get it.")

final_df = final_df.drop_duplicates(subset="text").reset_index(drop=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv(OUT_PATH, index=False)

print("Final dataset shape:", final_df.shape)
print(final_df["label_name"].value_counts())
print("\nSaved to:", OUT_PATH)