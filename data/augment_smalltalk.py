"""
Augments processed_dataset.csv with real conversational/meta/small-talk
examples labeled Normal (0).

WHY THIS EXISTS: testing found the baseline classifier misreading neutral,
low-emotional-content messages (e.g. "what do you know about my hobbies?",
"how does this app work?") as Mild Distress, because the training set had
almost no examples of this kind of text -- only 750/3025 rows (25%) were
Normal at all, and none of those were conversational/meta text; they were
all first-person motherhood narrative statements (see prepare_dataset.py's
template list). The app currently patches around this with hand-written
regexes in model_service.py (_check_greeting, _check_about_me), which work
but only catch the exact phrasings anticipated -- a new phrasing the mother
happens to use will fall right back to classifier guesswork.

This script adds ~70 more Normal-labeled examples covering: small talk,
questions about the platform/bot itself, gratitude/acknowledgement replies,
and generic off-topic queries -- so the model itself learns "text with no
distress signal in it" is Normal, rather than relying purely on regex
shortcuts. The regex checks in model_service.py should stay as a safety-net
layer regardless (defense in depth), but the classifier should get better
at the general case too.

Usage:
    python3 data/augment_smalltalk.py
Reads and overwrites: data/processed_dataset.csv (a .bak copy is kept)
"""
import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_dataset.csv")
BACKUP_PATH = DATA_PATH + ".bak"

SMALLTALK_NORMAL_EXAMPLES = [
    # --- Meta-questions about the bot / stored profile info ---
    "What do you know about my hobbies?",
    "Do you know my background?",
    "What do you know about me?",
    "What are my hobbies?",
    "Remember my hobbies?",
    "What did I tell you about myself?",
    "Do you remember what I said about my background?",
    "Can you tell me what you have on file for me?",
    "What information do you store about me?",
    "Is my chat with you private?",
    "Who can see my messages here?",
    "How does this chatbot work?",
    "What can you do?",
    "What is Amara?",
    "What is this platform for?",
    "How accurate are you?",
    "Are you a real doctor?",
    "Do you save my data?",
    "Is this confidential?",
    "What happens if I get flagged as distressed?",
    "How do doctors get assigned to me?",
    "Can I choose which doctor I see?",
    "How do I edit my profile?",
    "How do I add my hobbies to my profile?",
    "How do I change my password?",
    "Is this app free to use?",
    "How do I log out?",
    "What's your name?",
    "Are you a person or a bot?",
    "How were you trained?",
    # --- How-to / navigation questions (functional, not emotional) ---
    "How do I book an appointment with a doctor?",
    "How do I post something in the community?",
    "Where can I see my past appointments?",
    "How do I reply to someone in the community section?",
    "Can I delete a message I sent?",
    "How do I update my email address?",
    "What's on the community page?",
    "How is my privacy protected here?",
    "Can I use this app on my phone?",
    # --- Small talk / gratitude / acknowledgement ---
    "Thanks, that helps.",
    "Thank you so much.",
    "Okay, got it.",
    "That makes sense, thanks.",
    "Sounds good, thank you.",
    "Cool, thanks for explaining.",
    "Appreciate it.",
    "Got it, thank you.",
    "Nice, thanks for the tip.",
    "That's helpful, thank you.",
    "Alright, will do.",
    "Okay sounds good.",
    "Sure, that works.",
    "Perfect, thanks.",
    "Great, appreciate the help.",
    # --- Generic off-topic small talk ---
    "How was your day?",
    "Nice weather today isn't it.",
    "I like using this app.",
    "This app is quite helpful.",
    "I'm just testing this out.",
    "Just checking how this works.",
    "Just saying hi to see how you respond.",
    "Can you tell me a joke?",
    "Can you recommend a good book?",
    "What day is it today?",
    "Do you know what time it is?",
    "What's the weather like today?",
    "Can you set a reminder for me?",
    "What should I say to start?",
    "This is my first time using an app like this.",
    "I found this app through a friend.",
    "Just wanted to see what this chat does.",
    "Testing, testing, one two three.",
    "Hello, is anyone there?",
    "Can you hear me?",
]


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} existing rows.")
    print(df["label_name"].value_counts())

    if not os.path.exists(BACKUP_PATH):
        shutil.copy(DATA_PATH, BACKUP_PATH)
        print(f"Backed up original to {BACKUP_PATH}")

    aug_df = pd.DataFrame({
        "text": SMALLTALK_NORMAL_EXAMPLES,
        "label": 0,
        "label_name": "Normal",
    })

    before = len(df)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="text").reset_index(drop=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    added = len(combined) - before

    combined.to_csv(DATA_PATH, index=False)
    print(f"\nAdded {added} new small-talk/meta Normal examples (some may have deduped).")
    print(f"New shape: {combined.shape}")
    print(combined["label_name"].value_counts())
    print(f"\nSaved to {DATA_PATH}")


if __name__ == "__main__":
    main()
