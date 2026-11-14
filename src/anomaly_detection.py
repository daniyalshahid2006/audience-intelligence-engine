import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from comment_clustering import cluster_comments

FLAG_SHARE = 0.03   # flag the 3% most unusual posts


def residual_z(y, design):
    """Compare each post with what we would EXPECT given its platform, content type,
    brand and age, then express the gap in robust standard deviations."""
    X = design.to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y.to_numpy(dtype=float), rcond=None)
    resid = y.to_numpy(dtype=float) - X @ coef
    mad = np.median(np.abs(resid - np.median(resid)))
    return pd.Series(np.clip((resid - np.median(resid)) / (1.4826 * mad), -8, 8), index=y.index)


def detect_anomalies():
    posts = pd.read_csv(config.RAW_DIR / "posts.csv")
    clustered_path = config.PROCESSED_DIR / "comments_clustered.csv"
    summary_path = config.OUTPUT_DIR / "q2_audience_clusters.csv"
    if not clustered_path.exists() or not summary_path.exists():
        cluster_comments()
    comments = pd.read_csv(clustered_path, encoding="utf-8-sig")
    need_of = pd.read_csv(summary_path, encoding="utf-8-sig").set_index("cluster")["need"]
    comments["need"] = comments["cluster"].map(need_of)

    # What each post's conversation is made of (share of each audience need).
    mix = pd.crosstab(comments["post_id"], comments["need"], normalize="index")
    posts = posts.merge(mix, left_on="post_id", right_index=True, how="left").fillna({c: 0 for c in mix.columns})

    posts["age_group"] = pd.cut(posts["days_since_post"], [0, 3, 7, 14, 90], labels=["1-3", "4-7", "8-14", "15+"])
    posts["log_likes"] = np.log1p(posts["likes"])
    posts["log_comments"] = np.log1p(posts["comments_count"])
    posts["comment_to_like"] = np.log1p(posts["comments_count"] / posts["likes"].clip(lower=1))
    design = pd.get_dummies(posts[["platform", "content_type", "brand", "age_group"]].astype(str), drop_first=True)
    design.insert(0, "intercept", 1.0)
    for col in ["log_likes", "log_comments", "comment_to_like"]:
        posts["z_" + col] = residual_z(posts[col], design)

    giveaway = posts.get("Tagging and giveaways", pd.Series(0.0, index=posts.index))
    complaints = posts.get("Complaints", pd.Series(0.0, index=posts.index))
    features = pd.DataFrame({
        "z_likes": posts["z_log_likes"],
        "z_comments": posts["z_log_comments"],
        "z_ratio": posts["z_comment_to_like"],
        "giveaway_share": giveaway,
        "complaint_share": complaints,
    })

    model = IsolationForest(n_estimators=300, contamination=FLAG_SHARE, random_state=config.RANDOM_SEED)
    model.fit(features)
    posts["anomaly_score"] = -model.score_samples(features)   # higher = more unusual
    posts["flagged"] = model.predict(features) == -1

    def explain(row):
        reasons = []
        if row["z_log_likes"] > 3:
            reasons.append("far more likes than similar posts")
        if row["z_log_likes"] < -3:
            reasons.append("far fewer likes than similar posts")
        if row["z_comment_to_like"] > 3:
            reasons.append("unusually many comments for its likes")
        if giveaway[row.name] > 0.4:
            reasons.append("conversation dominated by tagging/giveaway entries")
        if complaints[row.name] > 0.25:
            reasons.append("many complaints")
        return "; ".join(reasons) if reasons else "unusual mix of signals"

    def label(row):
        if row["z_log_likes"] > 3:
            return "Outperformer"
        if row["z_log_likes"] < -3:
            return "Underperformer"
        if row["z_comment_to_like"] > 3 or giveaway[row.name] > 0.4:
            return "Comment spike"
        return "Other unusual"

    flagged = posts[posts["flagged"]].copy()
    flagged["type"] = flagged.apply(label, axis=1)
    flagged["why"] = flagged.apply(explain, axis=1)
    flagged = flagged.sort_values("anomaly_score", ascending=False)

    cols = ["post_id", "brand", "platform", "content_type", "campaign", "likes", "comments_count",
            "views", "days_since_post", "type", "why", "anomaly_score"]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flagged[cols].round(3).to_csv(config.OUTPUT_DIR / "q3_unusual_posts.csv", index=False)
    posts[["post_id", "anomaly_score", "flagged"]].round(4).to_csv(
        config.PROCESSED_DIR / "post_anomaly_scores.csv", index=False)

    # Validation only: compare with the anomalies planted in the simulation.
    planted = posts["sim_anomaly"] != "none"
    hit = flagged["post_id"].isin(posts.loc[planted, "post_id"])
    precision = hit.mean()
    recall = (planted & posts["flagged"]).sum() / planted.sum()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print(f"Flagged {len(flagged)} of {len(posts):,} posts ({100 * FLAG_SHARE:.0f}% most unusual)")
    print("\nFlagged posts by type:")
    print(flagged["type"].value_counts().to_string())
    print("\nValidation against the planted anomalies:")
    print(f"  precision {100 * precision:.0f}%  (flagged posts that were really planted)")
    print(f"  recall    {100 * recall:.0f}%  (planted anomalies that were found)")
    print("  recall by planted type:")
    for t in ["viral", "comment_spike", "flop"]:
        m = posts["sim_anomaly"] == t
        print(f"    {t:14s} {100 * (m & posts['flagged']).sum() / m.sum():.0f}%  ({int((m & posts['flagged']).sum())} of {int(m.sum())})")
    print("\nTop 5 unusual posts:")
    print(flagged[["post_id", "platform", "content_type", "likes", "comments_count", "type", "why"]].head(5).to_string(index=False))
    return flagged


if __name__ == "__main__":
    detect_anomalies()