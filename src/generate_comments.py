import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles can choke on Urdu script and emoji; make printing safe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

import config

# Table B holds a SAMPLE of each post's comments (posts.csv holds the true total).
MAX_COMMENTS_PER_POST = 15

COLORS = ["blue", "green", "maroon", "white", "black", "pink", "yellow", "mustard"]
EMOJIS = ["\U0001f60d", "\U0001f525", "\u2764\ufe0f", "\U0001f44c", "\U0001f62d", "\U0001f64f", "\u2728"]

TOPICS = [
    "Price inquiry", "Fabric quality", "Sizing", "Delivery", "Discount",
    "Availability", "Purchase intent", "Praise", "Complaint", "Tagging / giveaway",
]

# topic -> (sentiment, purchase_intent)
TOPIC_META = {
    "Price inquiry": ("Neutral", "Possible inquiry"),
    "Fabric quality": ("Neutral", "Possible inquiry"),
    "Sizing": ("Neutral", "Possible inquiry"),
    "Delivery": ("Neutral", "Possible inquiry"),
    "Discount": ("Neutral", "Possible inquiry"),
    "Availability": ("Neutral", "Possible inquiry"),
    "Purchase intent": ("Positive", "Strong intent"),
    "Praise": ("Positive", "None"),
    "Complaint": ("Negative", "None"),
    "Tagging / giveaway": ("Neutral", "None"),
}

# Topic mix per content type, in TOPICS order (normalised in code).
TOPIC_MIX = {
    "Reel":              [0.10, 0.04, 0.04, 0.05, 0.06, 0.12, 0.08, 0.38, 0.03, 0.10],
    "Product photo":     [0.22, 0.14, 0.14, 0.07, 0.08, 0.10, 0.07, 0.12, 0.03, 0.03],
    "Influencer post":   [0.10, 0.14, 0.05, 0.05, 0.07, 0.08, 0.07, 0.30, 0.04, 0.10],
    "Styling video":     [0.08, 0.18, 0.14, 0.04, 0.05, 0.08, 0.07, 0.28, 0.02, 0.06],
    "Sale announcement": [0.20, 0.03, 0.04, 0.06, 0.32, 0.12, 0.12, 0.04, 0.04, 0.03],
    "Eid campaign":      [0.10, 0.05, 0.06, 0.15, 0.08, 0.16, 0.14, 0.20, 0.03, 0.03],
}

# Language mix per platform: [English, Roman Urdu, Urdu]
LANGUAGES = ["English", "Roman Urdu", "Urdu"]
LANGUAGE_MIX = {
    "Instagram": [0.40, 0.50, 0.10],
    "TikTok":    [0.20, 0.65, 0.15],
    "Facebook":  [0.30, 0.40, 0.30],
}

TEMPLATES = {
    "Price inquiry": {
        "English": [
            "Price please?", "What is the price of this suit?", "How much for the 3 piece?",
            "Price?", "Can you tell me the price in DM?", "What's the price of this lawn?",
        ],
        "Roman Urdu": [
            "Price kya hai?", "Is ka price btayen", "Rate kya hai iska?",
            "Price bata dein please", "Ye kitne ka hai?", "Kitne ka hai ye suit?",
            "Price kitna hai 3 piece ka?", "Dm price plz",
        ],
        "Urdu": [
            "\u0642\u06cc\u0645\u062a \u06a9\u06cc\u0627 \u06c1\u06d2\u061f", "\u0627\u0633 \u06a9\u06cc \u0642\u06cc\u0645\u062a \u0628\u062a\u0627\u0626\u06cc\u06ba", "\u06cc\u06c1 \u06a9\u062a\u0646\u06d2 \u06a9\u0627 \u06c1\u06d2\u061f", "\u0631\u06cc\u0679 \u0628\u062a\u0627 \u062f\u06cc\u06ba \u067e\u0644\u06cc\u0632",
        ],
    },
    "Fabric quality": {
        "English": [
            "Is this pure lawn?", "How is the fabric quality?",
            "Is the fabric soft? Does it shrink after wash?",
            "Is it see-through? The fabric looks thin", "What is the fabric of the dupatta?",
        ],
        "Roman Urdu": [
            "Fabric kaisa hai?", "Kapra kesa hai is ka?", "Kya ye pure lawn hai?",
            "Quality kaisi hai? Rang toh nahi jata?", "Kapra patla toh nahi?",
            "Dupatta ka fabric kya hai?",
        ],
        "Urdu": [
            "\u06a9\u067e\u0691\u0627 \u06a9\u06cc\u0633\u0627 \u06c1\u06d2\u061f", "\u06a9\u06cc\u0627 \u06cc\u06c1 \u062e\u0627\u0644\u0635 \u0644\u0627\u0646 \u06c1\u06d2\u061f", "\u06a9\u0648\u0627\u0644\u0679\u06cc \u06a9\u06cc\u0633\u06cc \u06c1\u06d2\u061f", "\u06a9\u067e\u0691\u0627 \u067e\u062a\u0644\u0627 \u062a\u0648 \u0646\u06c1\u06cc\u06ba \u06c1\u06d2\u061f",
        ],
    },
    "Sizing": {
        "English": [
            "Which sizes are available?", "Is this size chart accurate?", "Do you have XL size?",
            "Is it available in large?", "Can it be stitched in custom size?",
        ],
        "Roman Urdu": [
            "Size chart bhej dein", "XL size mil jayega?", "Ye stitched hai ya unstitched?",
            "Size kya kya available hain?", "Large size hai?", "Custom size mein ban jayega?",
        ],
        "Urdu": [
            "\u0633\u0627\u0626\u0632 \u06a9\u0648\u0646 \u06a9\u0648\u0646 \u0633\u06d2 \u06c1\u06cc\u06ba\u061f", "\u06a9\u06cc\u0627 \u0627\u06cc\u06a9\u0633\u0679\u0631\u0627 \u0644\u0627\u0631\u062c \u062f\u0633\u062a\u06cc\u0627\u0628 \u06c1\u06d2\u061f", "\u0633\u0644\u0627 \u06c1\u0648\u0627 \u06c1\u06d2 \u06cc\u0627 \u0627\u0646 \u0633\u0644\u0627\u061f",
        ],
    },
    "Delivery": {
        "English": [
            "How many days for delivery?", "Do you deliver to Karachi?", "Delivery charges?",
            "Is cash on delivery available?", "How long does delivery take to Islamabad?",
        ],
        "Roman Urdu": [
            "Delivery kitne din mein hogi?", "Karachi delivery hoti hai?",
            "Delivery charges kitne hain?", "COD available hai?",
            "Lahore mein delivery kab tak hogi?", "Order kitne din mein pohnchega?",
        ],
        "Urdu": [
            "\u0688\u06cc\u0644\u06cc\u0648\u0631\u06cc \u06a9\u062a\u0646\u06d2 \u062f\u0646\u0648\u06ba \u0645\u06cc\u06ba \u06c1\u0648\u06af\u06cc\u061f", "\u06a9\u06cc\u0627 \u06a9\u0631\u0627\u0686\u06cc \u0688\u06cc\u0644\u06cc\u0648\u0631\u06cc \u06c1\u0648\u062a\u06cc \u06c1\u06d2\u061f", "\u06a9\u06cc\u0634 \u0622\u0646 \u0688\u06cc\u0644\u06cc\u0648\u0631\u06cc \u06c1\u06d2\u061f",
        ],
    },
    "Discount": {
        "English": [
            "Any discount on this?", "When is the sale starting?", "Is there a discount code?",
            "Please give a discount, too expensive", "Will this be part of the 11.11 sale?",
        ],
        "Roman Urdu": [
            "Discount milega?", "Sale kab lag rahi hai?", "Koi discount code hai?",
            "Sale mein ye kitne ka hoga?", "Thora discount kar dein",
            "Kya ye 11.11 sale mein hai?",
        ],
        "Urdu": [
            "\u06a9\u06cc\u0627 \u0688\u0633\u06a9\u0627\u0624\u0646\u0679 \u0645\u0644\u06d2 \u06af\u0627\u061f", "\u0633\u06cc\u0644 \u06a9\u0628 \u0644\u06af\u06d2 \u06af\u06cc\u061f", "\u062a\u06be\u0648\u0691\u06cc \u0631\u0639\u0627\u06cc\u062a \u06a9\u0631 \u062f\u06cc\u06ba",
        ],
    },
    "Availability": {
        "English": [
            "Is this still available?", "Do you have this in stock?",
            "When will this be restocked?", "Is this available in {color}?",
            "Is this available in stores?",
        ],
        "Roman Urdu": [
            "Ye available hai?", "Stock mein hai ye?", "Restock kab hoga?",
            "{color} colour mein hai?", "Store mein mil jayega?",
            "Sold out ho gaya? Dobara kab aayega",
        ],
        "Urdu": [
            "\u06a9\u06cc\u0627 \u06cc\u06c1 \u062f\u0633\u062a\u06cc\u0627\u0628 \u06c1\u06d2\u061f", "\u0627\u0633\u0679\u0627\u06a9 \u0645\u06cc\u06ba \u06c1\u06d2\u061f", "\u062f\u0648\u0628\u0627\u0631\u06c1 \u06a9\u0628 \u0622\u0626\u06d2 \u06af\u0627\u061f",
        ],
    },
    "Purchase intent": {
        "English": [
            "I want to order this", "Placing an order now", "Please send me the order link",
            "I need 2 of these in {color}", "Ordering today, how do I pay?",
        ],
        "Roman Urdu": [
            "Mujhe ye chahiye", "Order karna hai, link bhejein", "Main ne order kar diya",
            "2 suits chahiye {color} mein", "Kaise order karun?",
            "Inbox mein details bhej dein, order karna hai",
        ],
        "Urdu": [
            "\u0645\u062c\u06be\u06d2 \u06cc\u06c1 \u0686\u0627\u06c1\u06cc\u06d2", "\u0622\u0631\u0688\u0631 \u06a9\u0631\u0646\u0627 \u06c1\u06d2", "\u0622\u0631\u0688\u0631 \u06a9\u0627 \u0644\u0646\u06a9 \u0628\u06be\u06cc\u062c\u06cc\u06ba",
        ],
    },
    "Praise": {
        "English": [
            "So beautiful!", "Love this design", "Gorgeous collection", "Stunning colour",
            "Best lawn collection this season", "Absolutely love it",
        ],
        "Roman Urdu": [
            "Bohat khoobsurat", "Kya baat hai, bohat pyara", "Mashallah bohat acha design",
            "Color bohat zabardast hai", "Love this collection", "Wah bohat sundar",
        ],
        "Urdu": [
            "\u0628\u06c1\u062a \u062e\u0648\u0628\u0635\u0648\u0631\u062a", "\u0645\u0627\u0634\u0627\u0621\u0627\u0644\u0644\u06c1 \u0628\u06c1\u062a \u067e\u06cc\u0627\u0631\u0627", "\u06a9\u0645\u0627\u0644 \u06a9\u0627 \u0688\u06cc\u0632\u0627\u0626\u0646",
        ],
    },
    "Complaint": {
        "English": [
            "Ordered 10 days ago, still not received", "Quality was not as shown in the pictures",
            "Customer service never replies", "Colour faded after one wash",
            "Received the wrong size", "Price is too high for this quality",
        ],
        "Roman Urdu": [
            "Order kiya tha 10 din ho gaye abhi tak nahi aya",
            "Quality pictures jesi nahi thi", "Customer service reply hi nahi karti",
            "Ek wash mein rang utar gaya", "Galat size bhej diya",
            "Price bohat zyada hai quality kuch khas nahi",
        ],
        "Urdu": [
            "\u0622\u0631\u0688\u0631 \u06a9\u06cc\u06d2 \u062f\u0633 \u062f\u0646 \u06c1\u0648 \u06af\u0626\u06d2 \u0627\u0628\u06be\u06cc \u062a\u06a9 \u0646\u06c1\u06cc\u06ba \u0622\u06cc\u0627", "\u06a9\u0648\u0627\u0644\u0679\u06cc \u062a\u0635\u0627\u0648\u06cc\u0631 \u062c\u06cc\u0633\u06cc \u0646\u06c1\u06cc\u06ba \u062a\u06be\u06cc",
            "\u0642\u06cc\u0645\u062a \u0628\u06c1\u062a \u0632\u06cc\u0627\u062f\u06c1 \u06c1\u06d2",
        ],
    },
    "Tagging / giveaway": {
        "English": [
            "@friend look at this!", "Tag your bestie", "Giveaway entry done",
            "Done! Following and tagged 3 friends", "Check this out", "\U0001f60d\U0001f60d\U0001f60d",
        ],
        "Roman Urdu": [
            "Dekho ye kitna pyara hai", "Giveaway ke liye comment",
            "Done ji, friends ko tag kar diya", "Dekho yar ye acha hai",
        ],
        "Urdu": [
            "\u062f\u06cc\u06a9\u06be\u0648 \u06cc\u06c1 \u06a9\u062a\u0646\u0627 \u067e\u06cc\u0627\u0631\u0627 \u06c1\u06d2", "\u06af\u06cc\u0648 \u0627\u0648\u06d2 \u06a9\u06d2 \u0644\u06cc\u06d2 \u06a9\u0645\u0646\u0679",
        ],
    },
}

# Common Roman Urdu spelling variants (people spell the same word many ways).
VARIANTS = {
    "kitne": ["kitne", "kitnay", "kitnai"],
    "kitna": ["kitna", "kitnaa"],
    "hai": ["hai", "hy", "h", "he"],
    "bohat": ["bohat", "bohot", "boht", "bht"],
    "kya": ["kya", "kia", "kaya"],
    "mein": ["mein", "main", "me", "mai"],
    "ye": ["ye", "yeh", "ya"],
    "price": ["price", "prize", "rate"],
    "kapra": ["kapra", "kaprra"],
    "kesa": ["kesa", "kaisa"],
    "bata": ["bata", "bta"],
    "chahiye": ["chahiye", "chahiyay", "chaheye"],
    "kab": ["kab", "kb"],
    "dein": ["dein", "den", "dain"],
    "karna": ["karna", "krna"],
    "hogi": ["hogi", "hogii", "hoga"],
}


def vary_roman(text, rng):
    """Randomly swap some Roman Urdu words for alternative spellings."""
    out = []
    for word in text.split(" "):
        core = word.rstrip("?!.,")
        suffix = word[len(core):]
        key = core.lower()
        if key in VARIANTS and rng.random() < 0.5:
            alternatives = VARIANTS[key]
            word = alternatives[rng.integers(len(alternatives))] + suffix
        out.append(word)
    return " ".join(out)


def add_noise(text, language, rng):
    """Make the text look like real social media comments."""
    if language != "Urdu":
        if rng.random() < 0.6:
            text = text.lower()
        if rng.random() < 0.15:
            text = text.rstrip("?!.")
        if rng.random() < 0.06 and text and text[-1].isalpha():
            text = text + text[-1] * 2
    if rng.random() < 0.25:
        text = text + " " + EMOJIS[rng.integers(len(EMOJIS))]
    return text


def topic_weights(content_type, campaign, anomaly):
    """Topic probabilities for one post."""
    w = np.array(TOPIC_MIX[content_type], dtype=float)
    if campaign in ("11.11 Collection", "12.12 Collection"):
        w[TOPICS.index("Discount")] *= 1.8
    if campaign == "Eid Collection":
        w[TOPICS.index("Delivery")] *= 1.6
        w[TOPICS.index("Availability")] *= 1.3
    w = w / w.sum()
    if anomaly == "comment_spike":
        # A giveaway-style post: most comments are tags and entries.
        giveaway = np.zeros(len(TOPICS))
        giveaway[TOPICS.index("Tagging / giveaway")] = 1.0
        w = 0.35 * w + 0.65 * giveaway
    return w


def generate_comments(seed=config.RANDOM_SEED):
    posts_file = config.RAW_DIR / "posts.csv"
    if not posts_file.exists():
        raise FileNotFoundError(f"Run generate_posts.py first: {posts_file}")

    posts = pd.read_csv(posts_file)
    rng = np.random.default_rng(seed + 1)

    rows = []
    for post in posts.itertuples(index=False):
        n_comments = int(min(post.comments_count, MAX_COMMENTS_PER_POST))
        if n_comments <= 0:
            continue

        weights = topic_weights(post.content_type, post.campaign, post.sim_anomaly)
        topics = rng.choice(len(TOPICS), size=n_comments, p=weights)
        languages = rng.choice(len(LANGUAGES), size=n_comments,
                               p=LANGUAGE_MIX[post.platform])

        for t_idx, l_idx in zip(topics, languages):
            topic = TOPICS[t_idx]
            language = LANGUAGES[l_idx]
            options = TEMPLATES[topic][language]
            text = options[rng.integers(len(options))]
            text = text.replace("{color}", COLORS[rng.integers(len(COLORS))])
            if language == "Roman Urdu":
                text = vary_roman(text, rng)
            text = add_noise(text, language, rng)

            sentiment, intent = TOPIC_META[topic]
            rows.append((post.post_id, text, language, topic, sentiment, intent))

    df = pd.DataFrame(rows, columns=["post_id", "comment_text", "language_type",
                                     "topic", "sentiment", "purchase_intent"])
    df.insert(0, "comment_id", [f"C{i + 1:06d}" for i in range(len(df))])
    df["data_source"] = config.DATA_SOURCE

    out = config.RAW_DIR / "comments.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Saved {len(df):,} comments for {df['post_id'].nunique():,} posts to {out}")
    print(f"File size: {out.stat().st_size / 1e6:.1f} MB")
    print("\nLanguage mix (%):")
    print((df["language_type"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nTopic mix (%):")
    print((df["topic"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nSentiment (%):")
    print((df["sentiment"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nPurchase intent (%):")
    print((df["purchase_intent"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nSample comments (English / Roman Urdu):")
    sample = df[df["language_type"] != "Urdu"].sample(12, random_state=1)
    print(sample[["comment_text", "language_type", "topic"]].to_string(index=False))
    return df


if __name__ == "__main__":
    generate_comments()