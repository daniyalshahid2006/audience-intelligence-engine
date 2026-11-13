import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config

posts = pd.read_csv(config.RAW_DIR / "posts.csv")
comments = pd.read_csv(config.RAW_DIR / "comments.csv", encoding="utf-8-sig")

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""))


check("3,000 unique post_ids", posts["post_id"].nunique() == len(posts) == config.N_POSTS)
check("No negative likes or comment counts",
      bool((posts[["likes", "comments_count"]] >= 0).all().all()))
check("Missing values only in views and shares",
      int(posts.drop(columns=["views", "shares"]).isna().sum().sum()) == 0,
      f"views missing {int(posts['views'].isna().sum())}, shares missing {int(posts['shares'].isna().sum())}")
check("Every comment links to a post", bool(comments["post_id"].isin(posts["post_id"]).all()))
check("No empty comment text",
      bool(comments["comment_text"].notna().all() and (comments["comment_text"].str.strip() != "").all()))
check("Valid language labels", set(comments["language_type"]) == {"English", "Roman Urdu", "Urdu"})

per_post = comments.groupby("post_id").size()
totals = posts.set_index("post_id")["comments_count"].reindex(per_post.index)
check("Sampled comments never exceed the post's comments_count", bool((per_post <= totals).all()))

has_urdu_script = comments["comment_text"].str.contains(r"[\u0600-\u06FF]", regex=True)
check("Urdu script saved and read back correctly",
      bool((has_urdu_script == (comments["language_type"] == "Urdu")).all()),
      f"{int(has_urdu_script.sum()):,} Urdu-script comments")

dup = comments["comment_text"].duplicated().mean() * 100
print(f"INFO  {dup:.0f}% of comments repeat an earlier text (normal for short comments)")
print(f"INFO  {len(posts):,} posts, {len(comments):,} comments, "
      f"{per_post.mean():.1f} comments per post on average")

size = sum(f.stat().st_size for f in config.RAW_DIR.glob("*.csv")) / 1e6
print(f"INFO  data folder size: {size:.1f} MB")
print()
print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")