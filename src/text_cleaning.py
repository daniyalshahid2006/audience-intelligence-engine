import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles can choke on Urdu script and emoji; make printing safe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

import config

# Roman Urdu: many spellings of the same word -> one canonical word.
CANONICAL = {
    "kitnay": "kitne", "kitnai": "kitne", "kitnaa": "kitna",
    "hy": "hai", "h": "hai", "he": "hai",
    "bohot": "bohat", "boht": "bohat", "bht": "bohat",
    "kia": "kya", "kaya": "kya",
    "me": "mein", "mai": "mein",
    "yeh": "ye",
    "prize": "price", "rate": "price",
    "kaprra": "kapra", "kaisa": "kesa", "bta": "bata",
    "chahiyay": "chahiye", "chaheye": "chahiye",
    "kb": "kab", "den": "dein", "dain": "dein",
    "krna": "karna", "hoga": "hogi",
}


def squeeze(word):
    """Collapse repeated letters: 'haii' -> 'hai', 'kitnaa' -> 'kitna'."""
    return re.sub(r"(.)\1+", r"\1", word)


# Look words up with repeated letters squeezed, so 'hogii' still matches 'hogi'.
LOOKUP = {squeeze(k): v for k, v in CANONICAL.items()}

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
URDU_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")
ARABIC_PUNCT = re.compile(r"[\u060c\u061b\u061f\u066a-\u066d\u06d4]")
NOT_TEXT = re.compile(r"[^a-z0-9\u0600-\u06ff\s]")
URDU_LETTER_FIX = str.maketrans({"\u064a": "\u06cc", "\u0643": "\u06a9"})


def clean_text(text, language):
    """Clean one comment. Emoji and punctuation are removed; Urdu script is kept."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = URL.sub(" link ", text)
    text = MENTION.sub(" mention ", text)
    text = text.translate(URDU_LETTER_FIX)
    text = URDU_DIACRITICS.sub("", text)
    text = ARABIC_PUNCT.sub(" ", text)
    text = NOT_TEXT.sub(" ", text)          # removes emoji and punctuation
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # 'pleaseee' -> 'pleasee'

    words = text.split()
    if language == "Roman Urdu":
        words = [LOOKUP.get(squeeze(w), w) for w in words]

    cleaned = " ".join(words)
    return cleaned if cleaned else "emoji_only"


def clean_comments():
    src = config.RAW_DIR / "comments.csv"
    if not src.exists():
        raise FileNotFoundError(f"Run generate_comments.py first: {src}")

    df = pd.read_csv(src, encoding="utf-8-sig")
    df["clean_text"] = [clean_text(t, l) for t, l in zip(df["comment_text"], df["language_type"])]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "comments_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(f"Saved {len(df):,} cleaned comments to {out}")

    print("\nDistinct texts before and after cleaning:")
    rows = []
    for lang in ["English", "Roman Urdu", "Urdu"]:
        part = df[df["language_type"] == lang]
        before = part["comment_text"].str.lower().str.strip().nunique()
        after = part["clean_text"].nunique()
        rows.append((lang, len(part), before, after))
    summary = pd.DataFrame(rows, columns=["language", "comments", "distinct_before", "distinct_after"])
    print(summary.to_string(index=False))

    has_urdu = df["clean_text"].str.contains(r"[\u0600-\u06ff]", regex=True)
    urdu_kept = bool((has_urdu == (df["language_type"] == "Urdu")).all())
    print(f"\nUrdu script kept in every Urdu comment (and nowhere else): {urdu_kept}")
    print(f"Emoji-only comments given the placeholder: {int((df['clean_text'] == 'emoji_only').sum()):,}")
    print(f"Empty cleaned texts: {int((df['clean_text'].str.strip() == '').sum())}")

    print("\nBefore -> after examples (English / Roman Urdu):")
    sample = df[df["language_type"] != "Urdu"].sample(10, random_state=3)
    print(sample[["comment_text", "clean_text"]].to_string(index=False))
    return df


if __name__ == "__main__":
    clean_comments()