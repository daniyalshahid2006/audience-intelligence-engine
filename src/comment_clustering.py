import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config
from text_cleaning import clean_comments

K_RANGE = range(6, 15)

# A small bilingual domain lexicon (English, Roman Urdu, Urdu). It does two jobs:
# it links the same need across the three languages (concept features), and it names
# the clusters afterwards. It never sees the hidden topic label.
NEEDS = {
    "Price questions": ["price", "kitne", "cost", "much", "dm",
                        "\u0642\u06cc\u0645\u062a", "\u06a9\u062a\u0646\u06d2", "\u0631\u06cc\u0679"],
    "Fabric and quality": ["fabric", "kapra", "kesa", "pure", "soft", "shrink", "thin", "dupatta",
                           "patla", "quality", "rang", "see", "through",
                           "\u06a9\u067e\u0691\u0627", "\u0644\u0627\u0646", "\u06a9\u0648\u0627\u0644\u0679\u06cc", "\u067e\u062a\u0644\u0627", "\u062e\u0627\u0644\u0635"],
    "Sizing and fit": ["size", "sizes", "xl", "large", "stitched", "unstitched", "chart", "custom",
                       "\u0633\u0627\u0626\u0632", "\u0633\u0644\u0627", "\u0644\u0627\u0631\u062c"],
    "Delivery and shipping": ["delivery", "deliver", "cod", "charges", "pohnchega", "cash", "islamabad",
                              "karachi", "lahore", "\u0688\u06cc\u0644\u06cc\u0648\u0631\u06cc"],
    "Discounts and sales": ["discount", "sale", "code", "thora", "11",
                            "\u0631\u0639\u0627\u06cc\u062a", "\u0633\u06cc\u0644", "\u0688\u0633\u06a9\u0627\u0624\u0646\u0679"],
    "Availability and restock": ["available", "stock", "restock", "sold", "store", "stores", "dobara",
                                 "colour", "\u062f\u0633\u062a\u06cc\u0627\u0628", "\u0627\u0633\u0679\u0627\u06a9", "\u062f\u0648\u0628\u0627\u0631\u06c1"],
    "Ready to buy": ["order", "chahiye", "want", "need", "link", "placing", "ordering", "pay", "karna",
                     "\u0622\u0631\u0688\u0631", "\u0686\u0627\u06c1\u06cc\u06d2"],
    "Praise": ["beautiful", "love", "gorgeous", "stunning", "khoobsurat", "pyara", "zabardast",
               "sundar", "mashallah", "acha", "design", "best", "absolutely", "wah", "baat", "collection", "color",
               "\u062e\u0648\u0628\u0635\u0648\u0631\u062a", "\u067e\u06cc\u0627\u0631\u0627", "\u06a9\u0645\u0627\u0644"],
    "Complaints": ["ago", "received", "faded", "wrong", "never", "replies", "galat", "utar", "service",
                   "customer", "shown", "pictures", "still", "zyada", "high", "khas", "gaye", "abhi",
                   "reply", "jesi", "tha", "thi", "wash",
                   "\u062a\u0635\u0627\u0648\u06cc\u0631", "\u0632\u06cc\u0627\u062f\u06c1", "\u0627\u0628\u06be\u06cc", "\u06af\u0626\u06d2"],
    "Tagging and giveaways": ["tag", "tagged", "giveaway", "friends", "mention", "bestie", "dekho", "done",
                              "following", "emoji_only", "check", "look",
                              "\u06af\u06cc\u0648", "\u062f\u06cc\u06a9\u06be\u0648", "\u06a9\u0645\u0646\u0679"],
}

# word -> concept token, so the same need in English, Roman Urdu and Urdu shares a feature
CONCEPT_OF = {w: "cx" + str(i) for i, words in enumerate(NEEDS.values()) for w in words}


def add_concepts(text):
    """Append a shared concept token for every lexicon word found in the text."""
    found = [CONCEPT_OF[w] for w in text.split() if w in CONCEPT_OF]
    return text + " " + " ".join(found) if found else text


def build_features(texts):
    plain = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    concept = TfidfVectorizer(analyzer="word", sublinear_tf=True, token_pattern=r"cx\d+")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    Xw = plain.fit_transform(texts)
    Xk = concept.fit_transform([add_concepts(t) for t in texts])
    Xc = char.fit_transform(texts)
    # Concepts link the three languages; words and characters keep the finer detail.
    return hstack([1.5 * Xk, 0.6 * Xw, 0.2 * Xc]).tocsr(), Xw, plain


def name_cluster(tokens_in_cluster, overall):
    """Pick the need whose lexicon words best match this cluster's wording."""
    total = sum(tokens_in_cluster.values()) or 1
    best, best_score = "Mixed comments", 0.0
    for need, words in NEEDS.items():
        score = 0.0
        for w in words:
            share = tokens_in_cluster.get(w, 0) / total
            rarity = np.log(1 + 1 / max(overall.get(w, 1e-6), 1e-6))
            score += share * rarity
        if score > best_score:
            best, best_score = need, score
    return best


def cluster_comments(k=None):
    clean_path = config.PROCESSED_DIR / "comments_clean.csv"
    if not clean_path.exists():
        clean_comments()
    df = pd.read_csv(clean_path, encoding="utf-8-sig")

    # Many comments share the same cleaned wording, so cluster the distinct texts
    # and weight each by how often it occurs.
    uniq = df.groupby("clean_text").size().reset_index(name="count")
    texts = uniq["clean_text"].tolist()
    weights = uniq["count"].to_numpy(dtype=float)
    X, Xw, word_vec = build_features(texts)

    scores = {}
    for kk in K_RANGE:
        km = KMeans(n_clusters=kk, n_init=10, random_state=config.RANDOM_SEED)
        labels = km.fit_predict(X, sample_weight=weights)
        scores[kk] = silhouette_score(X, labels)
    best_k = k or max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, n_init=20, random_state=config.RANDOM_SEED)
    uniq["cluster"] = km.fit_predict(X, sample_weight=weights)

    df = df.merge(uniq[["clean_text", "cluster"]], on="clean_text", how="left")

    # Describe every cluster: size, top words, a need name, sample comments.
    vocab = np.array(word_vec.get_feature_names_out())
    overall_tokens = {}
    for text, cnt in zip(uniq["clean_text"], uniq["count"]):
        for tok in text.split():
            overall_tokens[tok] = overall_tokens.get(tok, 0) + cnt
    grand = sum(overall_tokens.values())
    overall_share = {t: c / grand for t, c in overall_tokens.items()}

    rows = []
    for c in sorted(uniq["cluster"].unique()):
        part = uniq[uniq["cluster"] == c]
        idx = part.index.to_numpy()
        centroid = np.asarray(Xw[idx].multiply(part["count"].to_numpy()[:, None]).sum(axis=0)).ravel()
        top_terms = [t for t in vocab[np.argsort(centroid)[::-1][:12]] if " " not in t][:5]
        tokens = {}
        for text, cnt in zip(part["clean_text"], part["count"]):
            for tok in text.split():
                tokens[tok] = tokens.get(tok, 0) + cnt
        need = name_cluster(tokens, overall_share)
        members = df[df["cluster"] == c]
        examples = (members[members["language_type"] != "Urdu"]["comment_text"]
                    .drop_duplicates().head(3).tolist())
        top_hidden = members["topic"].value_counts()
        rows.append({
            "cluster": int(c),
            "need": need,
            "comments": len(members),
            "share_pct": round(100 * len(members) / len(df), 1),
            "top_terms": ", ".join(top_terms),
            "examples": " | ".join(examples),
            "pct_english": round(100 * (members["language_type"] == "English").mean(), 1),
            "pct_roman_urdu": round(100 * (members["language_type"] == "Roman Urdu").mean(), 1),
            "pct_urdu": round(100 * (members["language_type"] == "Urdu").mean(), 1),
            "pct_strong_intent": round(100 * (members["purchase_intent"] == "Strong intent").mean(), 1),
            "pct_possible_inquiry": round(100 * (members["purchase_intent"] == "Possible inquiry").mean(), 1),
            "pct_negative": round(100 * (members["sentiment"] == "Negative").mean(), 1),
            # Validation only: the hidden label the model never saw.
            "validation_majority_topic": top_hidden.index[0],
            "validation_purity_pct": round(100 * top_hidden.iloc[0] / len(members), 1),
        })
    summary = pd.DataFrame(rows).sort_values("comments", ascending=False)

    ari = adjusted_rand_score(df["topic"], df["cluster"])
    purity = (df.groupby("cluster")["topic"].agg(lambda s: s.value_counts().iloc[0]).sum()) / len(df)

    df.to_csv(config.PROCESSED_DIR / "comments_clustered.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(config.OUTPUT_DIR / "q2_audience_clusters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"k": list(scores), "silhouette": [round(v, 4) for v in scores.values()]}).to_csv(
        config.OUTPUT_DIR / "q2_silhouette_by_k.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.max_columns", 20)
    print("Silhouette score by number of clusters:")
    print("  " + ", ".join(f"k={kk}: {v:.3f}" for kk, v in scores.items()))
    print(f"Chosen k = {best_k}")
    print(f"\nValidation against the hidden topic labels: purity {100 * purity:.1f}%, adjusted Rand index {ari:.3f}")
    print("\nAudience clusters (ASCII view):")
    show = summary[["cluster", "need", "comments", "share_pct", "validation_majority_topic", "validation_purity_pct"]]
    print(show.to_string(index=False))
    return df, summary


if __name__ == "__main__":
    cluster_comments()