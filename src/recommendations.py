import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

import config

ACTIONS = {
    "Price questions": "Show the price in the caption or a pinned comment so buyers do not have to ask.",
    "Discounts and sales": "Announce sale dates early and show the discount code inside the post.",
    "Availability and restock": "Add stock status to captions and announce restock dates.",
    "Delivery and shipping": "Put delivery time, charges and cash-on-delivery in a highlight or FAQ and the pinned comment.",
    "Sizing and fit": "Add a size-chart carousel and say clearly whether the suit is stitched or unstitched.",
    "Fabric and quality": "Post close-up fabric shots and wash-test content to answer quality worries.",
    "Ready to buy": "Add a direct order link to the post and bio: these commenters are ready to purchase.",
    "Complaints": "Reply to complaints within 24 hours and check delivery delays and colour-fastness.",
}


def plural(name):
    return name if name.endswith("s") else name + "s"


def build_recommendations():
    out = config.OUTPUT_DIR
    content = pd.read_csv(out / "q1_by_content_type.csv")
    pairs = pd.read_csv(out / "q1_by_platform_and_content.csv")
    clusters = pd.read_csv(out / "q2_audience_clusters.csv", encoding="utf-8-sig")
    unusual = pd.read_csv(out / "q3_unusual_posts.csv")
    comments = pd.read_csv(config.PROCESSED_DIR / "comments_clustered.csv", encoding="utf-8-sig")
    posts = pd.read_csv(config.RAW_DIR / "posts.csv")

    rows = []

    def add(question, finding, evidence, action):
        rows.append({"question": question, "finding": finding, "evidence": evidence, "recommended_action": action})

    # Q1: what attracts attention
    best, worst = content.iloc[0], content.iloc[-1]
    add("Q1 Content that attracts attention",
        f"{plural(best['content_type'])} earn the most attention",
        f"{best['attention_index']:.2f}x the typical post on the same platform (from {int(best['posts'])} posts)",
        f"Make more {best['content_type'].lower()}s; they are the strongest format.")
    top_pair = pairs.iloc[0]
    add("Q1 Content that attracts attention",
        f"Best platform and format: {plural(top_pair['content_type'])} on {top_pair['platform']}",
        f"{top_pair['attention_index']:.2f}x the platform's typical post",
        f"Prioritise {top_pair['content_type'].lower()}s on {top_pair['platform']}.")
    add("Q1 Content that attracts attention",
        f"{plural(worst['content_type'])} earn the least attention",
        f"{worst['attention_index']:.2f}x the typical post",
        f"Rework {worst['content_type'].lower()}s: pair them with video, a reel or an influencer.")

    # Q2/Q4: what the audience needs (merge clusters that share a need name)
    needs = (clusters.groupby("need")
             .agg(comments=("comments", "sum"), strong=("pct_strong_intent", "mean"))
             .reset_index())
    needs["share_pct"] = (100 * needs["comments"] / needs["comments"].sum()).round(1)
    needs = needs.sort_values("comments", ascending=False)
    actionable = needs[needs["need"].isin(ACTIONS)].head(4)
    for _, r in actionable.iterrows():
        add("Q2/Q4 What the audience wants",
            f"{r['need']} are {r['share_pct']:.1f}% of comments",
            f"{int(r['comments']):,} comments in this audience group",
            ACTIONS[r["need"]])

    # Which content type triggers price or discount questions most
    cm = comments.merge(clusters[["cluster", "need"]], on="cluster").merge(
        posts[["post_id", "content_type"]], on="post_id")
    pd_share = (cm.assign(is_price=cm["need"].isin(["Price questions", "Discounts and sales"]))
                .groupby("content_type")["is_price"].mean().sort_values(ascending=False))
    add("Q2/Q4 What the audience wants",
        f"{plural(pd_share.index[0])} draw the most price and discount questions",
        f"{100 * pd_share.iloc[0]:.0f}% of their comments (average {100 * pd_share.mean():.0f}%)",
        f"Put the price and offer inside every {pd_share.index[0].lower()}.")

    lang = comments["language_type"].value_counts(normalize=True) * 100
    add("Q2/Q4 What the audience wants",
        f"{lang.index[0]} is the main comment language ({lang.iloc[0]:.0f}%)",
        ", ".join(f"{k} {v:.0f}%" for k, v in lang.items()),
        "Reply to customers in the language they use, including Roman Urdu and Urdu script.")

    # Q3: unusual posts
    counts = unusual["type"].value_counts()
    for kind, action in [
        ("Outperformer", "Study these posts and repeat what they did (format, timing, creator)."),
        ("Underperformer", "Review these posts and stop repeating what they did."),
        ("Comment spike", "Check whether these are giveaways: they create comments but little buying intent."),
    ]:
        if kind in counts:
            ex = unusual[unusual["type"] == kind].iloc[0]
            add("Q3 Unusual patterns",
                f"{int(counts[kind])} {kind.lower()} posts flagged",
                f"for example {ex['post_id']} ({ex['platform']} {ex['content_type']}): {ex['why']}",
                action)

    result = pd.DataFrame(rows)
    result.insert(0, "priority", range(1, len(result) + 1))
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "q5_recommendations.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    print(f"{len(result)} recommendations saved to {out / 'q5_recommendations.csv'}\n")
    for _, r in result.iterrows():
        print(f"{r['priority']:>2}. [{r['question']}] {r['finding']}")
        print(f"    evidence: {r['evidence']}")
        print(f"    action:   {r['recommended_action']}")
    return result


if __name__ == "__main__":
    build_recommendations()