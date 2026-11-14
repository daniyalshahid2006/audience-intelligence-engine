import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

import config


def load_posts():
    posts = pd.read_csv(config.RAW_DIR / "posts.csv")
    posts["shares_filled"] = posts["shares"].fillna(0)
    posts["interactions"] = posts["likes"] + posts["comments_count"] + posts["shares_filled"]
    # Engagement rate needs views, which some posts do not show publicly.
    posts["engagement_rate"] = posts["interactions"] / posts["views"]
    posts["comment_to_like"] = posts["comments_count"] / posts["likes"].clip(lower=1)
    return posts


def add_index(df):
    """Interactions relative to the platform's typical post (1.00 = platform median).
    Needs no view count, so it is fair to posts that hide views."""
    df = df.copy()
    df["interactions_index"] = df["interactions"] / df.groupby("platform")["interactions"].transform("median")
    return df


def summarise(df, by):
    out = df.groupby(by).agg(
        posts=("post_id", "count"),
        median_views=("views", "median"),
        median_likes=("likes", "median"),
        median_comments=("comments_count", "median"),
        attention_index=("interactions_index", "median"),
        median_comment_to_like=("comment_to_like", "median"),
        median_engagement_rate=("engagement_rate", "median"),
    ).round(3)
    return out.sort_values("attention_index", ascending=False)


def analyse_content():
    posts = load_posts()
    # Posts younger than a week are still collecting likes, so leave them out of comparisons.
    mature = add_index(posts[posts["days_since_post"] >= 7])

    by_content = summarise(mature, "content_type")
    by_platform = summarise(mature, "platform")
    by_campaign = summarise(mature, "campaign")
    by_pair = (mature.groupby(["platform", "content_type"])
               .agg(posts=("post_id", "count"),
                    attention_index=("interactions_index", "median"),
                    median_comments=("comments_count", "median"))
               .round(3).reset_index()
               .sort_values("attention_index", ascending=False))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_content.to_csv(config.OUTPUT_DIR / "q1_by_content_type.csv")
    by_platform.to_csv(config.OUTPUT_DIR / "q1_by_platform.csv")
    by_campaign.to_csv(config.OUTPUT_DIR / "q1_by_campaign.csv")
    by_pair.to_csv(config.OUTPUT_DIR / "q1_by_platform_and_content.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Q1: using {len(mature):,} posts that are at least 7 days old")
    print("\nBy content type (attention_index: 1.00 = typical post on the same platform):")
    print(by_content.to_string())
    print("\nBy platform:")
    print(by_platform.to_string())
    print("\nBy campaign:")
    print(by_campaign.to_string())
    print("\nTop 3 platform + content combinations:")
    print(by_pair.head(3).to_string(index=False))
    return by_content, by_platform, by_campaign, by_pair


if __name__ == "__main__":
    analyse_content()