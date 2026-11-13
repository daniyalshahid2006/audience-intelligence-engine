import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config


def generate_posts(n_posts=config.N_POSTS, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = n_posts

    brand = rng.choice(list(config.BRANDS), size=n)
    size = np.array([config.BRANDS[b] for b in brand])
    platform = rng.choice(config.PLATFORMS, size=n, p=config.PLATFORM_WEIGHTS)
    content = rng.choice(config.CONTENT_TYPES, size=n, p=config.CONTENT_WEIGHTS)
    campaign = rng.choice(config.CAMPAIGNS, size=n, p=config.CAMPAIGN_WEIGHTS)
    days = rng.integers(1, 91, size=n)

    # Behaviour built into the data (the analysis should rediscover it)
    base_views = {"Instagram": 12000, "TikTok": 30000, "Facebook": 6000}
    like_rate = {"Instagram": 0.06, "TikTok": 0.03, "Facebook": 0.03}
    share_rate = {"Instagram": 0.04, "TikTok": 0.08, "Facebook": 0.05}
    view_mult = {"Reel": 1.8, "Product photo": 0.7, "Influencer post": 1.6,
                 "Styling video": 1.4, "Sale announcement": 0.9, "Eid campaign": 1.3}
    comment_mult = {"Reel": 1.0, "Product photo": 0.9, "Influencer post": 1.5,
                    "Styling video": 1.0, "Sale announcement": 2.2, "Eid campaign": 1.2}

    # Views build up quickly in the first days, then level off
    maturity = 1 - np.exp(-days / 4)

    views = (np.array([base_views[p] for p in platform]) * size
             * np.array([view_mult[c] for c in content])
             * maturity * rng.lognormal(0, 0.6, n))
    likes = (views * np.array([like_rate[p] for p in platform])
             * rng.lognormal(0, 0.25, n))
    comments = (likes * 0.08 * np.array([comment_mult[c] for c in content])
                * rng.lognormal(0, 0.3, n))
    shares = (likes * np.array([share_rate[p] for p in platform])
              * rng.lognormal(0, 0.4, n))

    # Planted anomalies (hidden label, used only to validate detection later)
    anomaly = np.full(n, "none", dtype=object)
    idx = rng.choice(n, size=int(n * config.ANOMALY_RATE), replace=False)
    kinds = rng.choice(["viral", "comment_spike", "flop"], size=len(idx))
    for i, k in zip(idx, kinds):
        anomaly[i] = k
        if k == "viral":
            views[i] *= 8
            likes[i] *= 8
            comments[i] *= 6
            shares[i] *= 10
        elif k == "comment_spike":
            comments[i] *= 6
        else:
            views[i] *= 0.1
            likes[i] *= 0.1
            comments[i] *= 0.1
            shares[i] *= 0.1

    views = np.rint(views).astype(float)
    shares = np.rint(shares).astype(float)

    # "If available": photos/sale posts often show no view count on Instagram/Facebook,
    # and Instagram often hides shares
    no_views = (np.isin(content, ["Product photo", "Sale announcement"])
                & np.isin(platform, ["Instagram", "Facebook"])
                & (rng.random(n) < 0.7))
    views[no_views] = np.nan
    no_shares = (platform == "Instagram") & (rng.random(n) < 0.5)
    shares[no_shares] = np.nan

    df = pd.DataFrame({
        "post_id": [f"P{i + 1:04d}" for i in range(n)],
        "brand": brand,
        "platform": platform,
        "campaign": campaign,
        "content_type": content,
        "product_category": config.PRODUCT_CATEGORY,
        "likes": np.rint(likes).astype(int),
        "comments_count": np.rint(comments).astype(int),
        "shares": pd.array(shares, dtype="Int64"),
        "views": pd.array(views, dtype="Int64"),
        "days_since_post": days,
        "data_source": config.DATA_SOURCE,
        "collection_date": config.COLLECTION_DATE,
        "sim_anomaly": anomaly,
    })

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RAW_DIR / "posts.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Saved {len(df):,} posts to {out}")
    print(f"Columns: {len(df.columns)}")
    print("\nMissing values:")
    print(df[["views", "shares"]].isna().sum().to_string())
    print("\nPlatforms:")
    print(df["platform"].value_counts().to_string())
    print("\nPlanted anomalies:")
    print(df["sim_anomaly"].value_counts().to_string())
    print("\nMedian by content type:")
    print(df.groupby("content_type")[["views", "likes", "comments_count"]]
          .median().round(0).to_string())
    print("\nFirst 5 posts:")
    print(df.drop(columns=["product_category", "data_source", "collection_date"]).head().to_string(index=False))
    return df


if __name__ == "__main__":
    generate_posts()