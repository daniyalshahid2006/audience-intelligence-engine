from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
RAW = BASE / "data" / "raw"

st.set_page_config(page_title="Audience and Campaign Intelligence Engine", layout="wide")


@st.cache_data
def load(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def missing_files():
    needed = [RAW / "posts.csv", OUT / "q1_by_content_type.csv", OUT / "q1_by_platform_and_content.csv",
              OUT / "q1_by_campaign.csv", OUT / "q2_audience_clusters.csv", OUT / "q3_unusual_posts.csv",
              OUT / "q5_recommendations.csv"]
    return [p.name for p in needed if not p.exists()]


problems = missing_files()
if problems:
    st.error("Results are missing: " + ", ".join(problems) + ". Run `python src/run_pipeline.py` first.")
    st.stop()

posts = load(RAW / "posts.csv")
q1_content = load(OUT / "q1_by_content_type.csv")
q1_pairs = load(OUT / "q1_by_platform_and_content.csv")
q1_campaign = load(OUT / "q1_by_campaign.csv")
clusters = load(OUT / "q2_audience_clusters.csv")
unusual = load(OUT / "q3_unusual_posts.csv")
recs = load(OUT / "q5_recommendations.csv")

# Merge clusters that carry the same need name into one audience group.
needs = clusters.groupby("need").agg(
    comments=("comments", "sum"),
    english=("pct_english", "mean"),
    roman_urdu=("pct_roman_urdu", "mean"),
    urdu=("pct_urdu", "mean"),
    strong_intent=("pct_strong_intent", "mean"),
    possible_inquiry=("pct_possible_inquiry", "mean"),
    negative=("pct_negative", "mean"),
).reset_index()
needs["share_pct"] = (100 * needs["comments"] / needs["comments"].sum()).round(1)
needs = needs.sort_values("comments", ascending=False)

st.sidebar.title("Audience and Campaign Intelligence Engine")
page = st.sidebar.radio(
    "Choose a question",
    ["Overview",
     "1. Content that attracts attention",
     "2. What the audience wants",
     "3. Unusual patterns",
     "4. Audience needs to address",
     "5. What marketing should do next"],
)
st.sidebar.info("All data in this app is SIMULATED for a women's lawn-suit brand in Pakistan. "
                "It demonstrates the method and is not real customer data.")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Audience and Campaign Intelligence Engine")
    st.caption("Women's lawn suits | Instagram, TikTok and Facebook | Simulated data")
    total_comments = int(clusters["comments"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts analysed", f"{len(posts):,}")
    c2.metric("Comments analysed", f"{total_comments:,}")
    c3.metric("Audience groups found", f"{len(clusters)}")
    c4.metric("Unusual posts flagged", f"{len(unusual)}")

    st.subheader("What this app answers")
    st.markdown(
        "1. **Which content attracts attention?** Reels, photos, influencer posts and more, compared fairly.\n"
        "2. **What does the audience want?** Comments in English, Roman Urdu and Urdu are grouped by need.\n"
        "3. **Which patterns are unusual?** Posts that behave very differently from similar posts.\n"
        "4. **Which audience needs should the brand address?** Price, delivery, availability, sizing and more.\n"
        "5. **What should marketing do next?** Evidence-based actions."
    )
    st.subheader("How it works")
    st.markdown(
        "- **Text cleaning** merges spelling variants of Roman Urdu (for example *kitne / kitnay*, *bohat / bht*) "
        "and keeps Urdu script.\n"
        "- **Clustering** (TF-IDF with K-Means, number of groups chosen by silhouette score) groups comments by need.\n"
        "- **Anomaly detection** (Isolation Forest) compares each post with its expected engagement.\n"
        "- **Recommendations** are rules driven by the numbers above."
    )
    st.warning("Limitations: the data is simulated and template-based, so the results are cleaner than real social "
               "media comments would be, and a small English / Roman Urdu / Urdu word list links the three languages.")

# ---------------------------------------------------------------- Q1
elif page.startswith("1."):
    st.title("1. Which content attracts attention?")
    st.write("The **attention index** compares each post with the typical post on the same platform "
             "(1.00 = typical). It needs no view count, so it is fair to posts that hide views.")
    metric = st.radio("Measure", ["Attention index", "Comment-to-like ratio (conversation)"], horizontal=True)
    col = "attention_index" if metric == "Attention index" else "median_comment_to_like"
    data = q1_content.sort_values(col, ascending=False)
    fig = px.bar(data, x="content_type", y=col, text=col, labels={"content_type": "Content type", col: metric})
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")
    best = q1_content.sort_values("attention_index", ascending=False).iloc[0]
    st.success(f"**{best['content_type']}** attract the most attention ({best['attention_index']:.2f}x a typical post).")

    st.subheader("Platform and content type together")
    pivot = q1_pairs.pivot(index="content_type", columns="platform", values="attention_index")
    st.plotly_chart(px.imshow(pivot, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                              labels={"color": "Attention index"}), width="stretch")

    st.subheader("Campaigns")
    st.dataframe(q1_campaign[["campaign", "posts", "attention_index", "median_comments"]], hide_index=True)
    with st.expander("Full table by content type"):
        st.dataframe(q1_content, hide_index=True)

# ---------------------------------------------------------------- Q2
elif page.startswith("2."):
    st.title("2. What does the audience want?")
    st.write("Comments were cleaned and grouped automatically. Each group is named by its dominant wording.")
    fig = px.bar(needs.sort_values("share_pct"), x="share_pct", y="need", orientation="h", text="share_pct",
                 labels={"share_pct": "Share of all comments (%)", "need": ""})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Language mix by audience need")
    lang = needs.melt(id_vars="need", value_vars=["english", "roman_urdu", "urdu"],
                      var_name="language", value_name="percent")
    lang["language"] = lang["language"].map({"english": "English", "roman_urdu": "Roman Urdu", "urdu": "Urdu"})
    st.plotly_chart(px.bar(lang, x="need", y="percent", color="language", barmode="stack",
                           labels={"need": "", "percent": "% of comments"}), width="stretch")

    st.subheader("Look inside a group")
    choice = st.selectbox("Audience group", clusters.sort_values("comments", ascending=False)["need"].unique())
    for _, r in clusters[clusters["need"] == choice].iterrows():
        st.markdown(f"**Cluster {int(r['cluster'])}**: {int(r['comments']):,} comments ({r['share_pct']}%)")
        st.write(f"Top words: {r['top_terms']}")
        st.write(f"Example comments: {r['examples']}")
        st.write(f"Strong buying intent: {r['pct_strong_intent']}% | Possible inquiry: {r['pct_possible_inquiry']}% "
                 f"| Negative: {r['pct_negative']}%")

# ---------------------------------------------------------------- Q3
elif page.startswith("3."):
    st.title("3. Which patterns are unusual?")
    st.write("Each post is compared with the engagement we would expect given its platform, content type, brand "
             "and age. The 3% most unusual posts are flagged.")
    plot = posts.copy()
    plot["flagged"] = plot["post_id"].isin(unusual["post_id"])
    plot = plot.merge(unusual[["post_id", "type"]], on="post_id", how="left")
    plot["status"] = plot["type"].fillna("Normal")
    fig = px.scatter(plot, x="likes", y="comments_count", color="status", log_x=True, log_y=True,
                     hover_data=["post_id", "platform", "content_type"], opacity=0.7,
                     labels={"likes": "Likes", "comments_count": "Comments", "status": "Post status"})
    st.plotly_chart(fig, width="stretch")

    counts = unusual["type"].value_counts()
    cols = st.columns(len(counts))
    for col, (kind, n) in zip(cols, counts.items()):
        col.metric(kind, int(n))

    kind = st.selectbox("Show flagged posts of type", ["All"] + list(counts.index))
    shown = unusual if kind == "All" else unusual[unusual["type"] == kind]
    st.dataframe(shown[["post_id", "platform", "content_type", "campaign", "likes", "comments_count",
                        "type", "why"]], hide_index=True)

# ---------------------------------------------------------------- Q4
elif page.startswith("4."):
    st.title("4. Which audience needs should the brand address?")
    st.write("Needs are ranked by how many comments they represent, with how ready those commenters are to buy.")
    table = needs[["need", "comments", "share_pct", "strong_intent", "possible_inquiry", "negative"]].copy()
    table.columns = ["Audience need", "Comments", "Share (%)", "Strong intent (%)", "Possible inquiry (%)",
                     "Negative (%)"]
    st.dataframe(table.round(1), hide_index=True)
    actionable = needs[~needs["need"].isin(["Praise", "Tagging and giveaways"])].head(5)
    st.subheader("Biggest needs the brand can act on")
    for _, r in actionable.iterrows():
        st.markdown(f"- **{r['need']}**: {r['share_pct']}% of comments ({int(r['comments']):,}), "
                    f"{r['possible_inquiry']:.0f}% are buying questions")
    st.info("Praise and giveaway entries are healthy for reach, but they carry little purchase intent.")

# ---------------------------------------------------------------- Q5
else:
    st.title("5. What should marketing do next?")
    st.write("Every action below comes with the evidence behind it.")
    for question in recs["question"].unique():
        st.subheader(question)
        for _, r in recs[recs["question"] == question].iterrows():
            with st.container(border=True):
                st.markdown(f"**{int(r['priority'])}. {r['finding']}**")
                st.caption(f"Evidence: {r['evidence']}")
                st.write(f"Action: {r['recommended_action']}")
    st.download_button("Download recommendations (CSV)", recs.to_csv(index=False).encode("utf-8-sig"),
                       file_name="recommendations.csv", mime="text/csv")