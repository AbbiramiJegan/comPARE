# main.py

import streamlit as st
import pandas as pd
import re
import os
import json
import string
from fuzzywuzzy import fuzz, process
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
from fetchProfile import get_scholar_profile  # reuse API function

st.set_page_config(page_title="Scholar Report Card", layout="centered")

# ---------------- Embedding model ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# normalisation function to use on the venue string 
def normalize_text(s: str) -> str:
    s = s.lower() # lower-case
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove extra spaces
    s = re.sub(r"\s+", " ", s).strip()
    # remove common stop words
    stop_words = [
        "international", "on", "of", "the", "and"]
    tokens = [t for t in s.split() if t not in stop_words]
    return " ".join(tokens)

# ---------------- Load quality datasets ----------------
@st.cache_resource
def load_quality_data():
    def clean_columns(df):
        # Strip whitespace, remove duplicates, lower-case optional
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    # Load CSVs
    journals = pd.read_csv("qualityJournal.csv")
    conf_main = pd.read_csv("qualityConferences.csv")
    conf_journal = pd.read_csv("qualityConferences-journal.csv")

    # Clean columns
    journals = clean_columns(journals)
    conf_main = clean_columns(conf_main)
    conf_journal = clean_columns(conf_journal)

    # Normalize conference column names to "Title"
    for df in [conf_main, conf_journal]:
        if "Title" not in df.columns:
            for col in ["Conference Name (DBLP)", "CORE Conference Name", "ERA Conference Name", "GS Name"]:
                if col in df.columns:
                    df.rename(columns={col: "Title"}, inplace=True)

    # Ensure all columns are unique before concatenating
    conf_main = conf_main.loc[:, ~conf_main.columns.duplicated()]
    conf_journal = conf_journal.loc[:, ~conf_journal.columns.duplicated()]

    # Concatenate safely
    conferences = pd.concat([conf_main, conf_journal], ignore_index=True)

    # ---- Add normalized titles for matching ----
    journals["Title"] = journals["Title"].astype(str)
    conferences["Title"] = conferences["Title"].astype(str)

    journals["Title_norm"] = journals["Title"].apply(normalize_text)
    conferences["Title_norm"] = conferences["Title"].apply(normalize_text)

    return journals, conferences

# Actually load the datasets
journals, conferences = load_quality_data()

# ---------------- Precompute embeddings ----------------
@st.cache_resource
def precompute_embeddings():
    journal_embs = model.encode(journals["Title"].astype(str).tolist(), convert_to_tensor=True)
    conf_embs = model.encode(conferences["Title"].astype(str).tolist(), convert_to_tensor=True)
    return journal_embs, conf_embs

journal_embs, conf_embs = precompute_embeddings()


# ---------------- Helper functions ----------------
def extract_author_id(url):
    match = re.search(r"user=([\w-]+)", url)
    return match.group(1) if match else None

def load_or_fetch_author(author_id):
    """Check dataset folder, if missing call API and save JSON"""
    os.makedirs("dataset", exist_ok=True)
    # Check if any file exists with author_id prefix
    files = [f for f in os.listdir("dataset") if f.startswith(author_id)]
    if files:
        # Load the latest file
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join("dataset", x)))
        with open(os.path.join("dataset", latest_file), "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Fetch via SerpAPI
        data = get_scholar_profile(author_id)
    return data

# Semantic match that returns top_k candidates (index, score)
def semantic_match(venue, choices, choice_embs, top_k=10):
    if not isinstance(venue, str) or venue.strip() == "":
        return []
    venue_emb = model.encode(venue, convert_to_tensor=True)
    cos_scores = util.cos_sim(venue_emb, choice_embs)[0]
    k = min(top_k, len(choices))
    if k == 0:
        return []
    top_results = torch.topk(cos_scores, k=k)
    indices = top_results.indices.cpu().numpy()
    scores = top_results.values.cpu().numpy()
    return list(zip(indices, scores))

# return a venue guess based on keywords 
def guesser(venue: str) -> str:
    v = venue.lower()

    # suggested words from chat GPT - can look into refining these
    conf_keywords = [
        "conference", "conf.", "workshop", "symposium", "proceedings",
        "meeting", "colloquium", "seminar", "summit"
    ]
    journ_keywords = [
        "journal", "transactions", "trans.", "letters", "magazine",
        "revue", "revista", "annals", "archives", "bulletin"
    ]
    strong_conf_acronyms = [
        "icml", "neurips", "nips", "aaai", "ijcai", "cvpr", "iccv", "eccv",
        "kdd", "sdm", "aistats", "emnlp", "acl", "naacl", "eacl", "iclr",
        "icra", "iros", "uai", "ecml", "pkdd"
    ]

    if any(k in v for k in conf_keywords) or any(acronym in v for acronym in strong_conf_acronyms):
        return "conference"
    if any(k in v for k in journ_keywords):
        return "journal"
    return "unknown"

def match_quality(venue, is_cs_ai=False):
    # handling missing values
    if pd.isna(venue) or not isinstance(venue, str) or venue.strip() == "":
        return None, None, None, 0

    venue_norm = normalize_text(venue)
    venue_tokens = set(venue_norm.split())

    hint = guesser(venue) # get the guess

    # default ordering
    if is_cs_ai:
        first_df, first_kind = conferences, "Conference"
        second_df, second_kind = journals, "Journal"
    else:
        first_df, first_kind = journals, "Journal"
        second_df, second_kind = conferences, "Conference"

    # override ordering based on guess
    if hint == "conference":
        first_df, first_kind = conferences, "Conference"
        second_df, second_kind = journals, "Journal"
    elif hint == "journal":
        first_df, first_kind = journals, "Journal"
        second_df, second_kind = conferences, "Conference"

    # looking for exact or substring matches
    # exact match is when the normalised tokens are the same
    # substring matches require at least 50% token overlap (Ex: "machine learning" ⊂ "icml machine learning")
    def exact_or_substring_match(df, kind):
        if "Title_norm" not in df.columns:
            return None

        for idx, row in df[["Title", "Title_norm"]].dropna().iterrows():
            t_norm = row["Title_norm"]
            if not t_norm:
                continue
            title_tokens = set(t_norm.split())
            if not title_tokens:
                continue

            # 50% token overlap check
            overlap = len(venue_tokens & title_tokens) / max(len(venue_tokens), len(title_tokens))
            if overlap < 0.5:
                continue

            if venue_norm == t_norm:
                overlap_score = 100.0
            elif venue_norm in t_norm or t_norm in venue_norm:
                overlap_score = 95.0
            else:
                continue

            if kind == "Journal":
                rank = row.get("SCIMAGO Best Quartile 2023", "-")
            else:
                rank = row.get("CORE 2021 Rank", row.get("Rank", "-"))

            return kind, row["Title"], rank, overlap_score

        return None

    # get the exact or substring match guess and score
    res = exact_or_substring_match(first_df, first_kind)
    if res:
        return res
    res = exact_or_substring_match(second_df, second_kind)
    if res:
        return res

    def pick_from_semantic(df, embs, kind):
        """
        1. Get top-k semantic matches from embeddings
        2. Evaluate both semantic score AND fuzzy score
        3. Accept if:
           - semantic ≥ 0.75
           OR
           - semantic ≥ 0.65 AND fuzzy ≥ 80
        """
        candidates = semantic_match(venue, df["Title"], embs, top_k=10)
        best = None
        best_score = -1.0  # raw cosine

        for idx, score in candidates:
            row = df.iloc[idx]
            title = str(row["Title"])
            title_norm = row.get("Title_norm", normalize_text(title))

            fuzzy_score = fuzz.ratio(venue_norm, title_norm)

            if score >= 0.75 or (score >= 0.65 and fuzzy_score >= 80):
                sim_percent = round(float(score) * 100, 1)
                if kind == "Journal":
                    rank = row.get("SCIMAGO Best Quartile 2023", "-")
                else:
                    rank = row.get("CORE 2021 Rank", row.get("Rank", "-"))

                if score > best_score:
                    best_score = float(score)
                    best = (kind, title, rank, sim_percent)

        return best

    first_embs = conf_embs if first_kind == "Conference" else journal_embs
    second_embs = conf_embs if second_kind == "Conference" else journal_embs

    res = pick_from_semantic(first_df, first_embs, first_kind)
    if res:
        return res
    res = pick_from_semantic(second_df, second_embs, second_kind)
    if res:
        return res

    def best_fuzzy(df, kind):
        """
        Pure fuzzy matching, but still labeled as Journal/Conference.
        Only used if semantic stages fail.
        """
        if df.empty:
            return None
        titles = df["Title"].astype(str).tolist()
        result = process.extractOne(venue, titles)
        if not result:
            return None
        match, score = result[0], result[1]
        row = df[df["Title"] == match].iloc[0]
        if kind == "Journal":
            rank = row.get("SCIMAGO Best Quartile 2023", "-")
        else:
            rank = row.get("CORE 2021 Rank", row.get("Rank", "-"))
        return kind, match, rank, score

    best_j = best_fuzzy(journals, "Journal")
    best_c = best_fuzzy(conferences, "Conference")

    best = None
    # pick the fuzzy match that is stronger
    if best_j and best_c:
        best = best_j if best_j[3] >= best_c[3] else best_c
    else:
        best = best_j or best_c
    # only trust the fuzzy match if its above threshhold
    if best and best[3] >= 80:
        return best

    all_titles = pd.concat([journals["Title"], conferences["Title"]]).astype(str).tolist()
    result = process.extractOne(venue, all_titles)
    if result:
        match, fuzzy_score = result[0], result[1]
        return "Fuzzy", match, "-", fuzzy_score
    else:
        return "Fuzzy", None, "-", 0



def evaluate_author_data(author_data, cs_ai=False):
    articles = pd.DataFrame(author_data["articles"])

    # Ensure 'year' and 'citations' are numeric
    articles["year"] = pd.to_numeric(articles["year"], errors="coerce")       # invalid parsing -> NaN
    articles["citations"] = pd.to_numeric(articles["citations"], errors="coerce").fillna(0).astype(int)

    total_papers = len(articles)
    total_citations = articles["citations"].sum()
    cpp = total_citations / total_papers if total_papers else 0

    # Filter recent articles safely
    recent_articles = articles[articles["year"] >= 2020]
    recent_papers = len(recent_articles)
    recent_citations = recent_articles["citations"].sum()

    top5 = articles.sort_values("citations", ascending=False).head(5)
    top5_avg = top5["citations"].mean() if not top5.empty else 0

    # Match venues
    articles[["match_type","matched_title","rank","match_score"]] = articles.apply(
        lambda row: pd.Series(match_quality(row["venue"], is_cs_ai=cs_ai)),
        axis=1
    )

    results = {
        "author_id": author_data["author_id"],
        "name": author_data["name"],
        "total_papers": total_papers,
        "total_citations": total_citations,
        "cpp": round(cpp, 2),
        "recent_papers": recent_papers,
        "recent_citations": recent_citations,
        "top5_avg_citations": round(top5_avg, 2)
    }

    return results, articles

# ---------------- Streamlit UI ----------------
st.title("🎓 Research Performance Dashboard")
st.caption("Comprehensive evaluation of researcher publication quality and impact")

url = st.text_input("🔗 Enter Google Scholar Profile Link:")
is_cs_ai = st.toggle("💻 CS/AI researcher (prioritize conferences)", value=False)

if url:
    author_id = extract_author_id(url)
    if not author_id:
        st.error("❌ Could not extract author ID from URL")
    else:
        author_data = load_or_fetch_author(author_id)
        if not author_data or not author_data.get("articles"):
            st.error("❌ No articles found for this author")
        else:
            results, papers = evaluate_author_data(author_data, cs_ai=is_cs_ai)

            st.markdown(f"## 👤 {results['name']} — *Research Summary*")

            # --- KPIs ---
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("📄 Total Papers", results['total_papers'])
            kpi2.metric("📚 Total Citations", results['total_citations'])
            kpi3.metric("⭐ Citations per Paper", results['cpp'])

            kpi4, kpi5, kpi6 = st.columns(3)
            kpi4.metric("🕒 Recent Papers (≥2020)", results['recent_papers'])
            kpi5.metric("🔍 Recent Citations", results['recent_citations'])
            kpi6.metric("🏆 Top 5 Avg Citations", results['top5_avg_citations'])

            st.markdown("---")

            # --- Publication Type Distribution ---
            st.subheader("🏛️ Publication Type Distribution")
            venue_counts = papers["match_type"].value_counts()
            st.dataframe(venue_counts.rename_axis("Type").reset_index(), hide_index=True, use_container_width=True)

            import plotly.express as px
            fig_venue = px.pie(
                values=venue_counts.values,
                names=venue_counts.index,
                title="Publication Type Share",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_venue.update_traces(textinfo="percent+label", pull=[0.05]*len(venue_counts))
            st.plotly_chart(fig_venue, use_container_width=True)

            # --- Rank / Quartile Distribution ---
            st.subheader("📊 Venue Quality (Rank / Quartile)")
            def rank_category(row):
                valid_ranks = {"Q1","Q2","Q3","Q4","A*","A","B","C"}
                rank = str(row["rank"]).strip() if pd.notna(row["rank"]) else None
                return rank if rank in valid_ranks else "Unranked"

            rank_counts = papers.apply(rank_category, axis=1).value_counts()
            desired_order = ["Q1","Q2","Q3","Q4","A*","A","B","C","Unranked"]
            rank_counts = rank_counts.reindex(desired_order, fill_value=0)

            fig_rank = px.bar(
                x=rank_counts.values,
                y=rank_counts.index,
                orientation="h",
                text=rank_counts.values,
                title="Rank / Quartile Distribution",
                color=rank_counts.index,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig_rank.update_layout(yaxis_title="", xaxis_title="Paper Count")
            st.plotly_chart(fig_rank, use_container_width=True)

            # --- Citation Distribution ---
            st.subheader("📉 Citation Distribution")
            import numpy as np
            bins = [0, 10, 50, 100, 500, 1000]
            labels = ["0–10", "11–50", "51–100", "101–500", "500+"]
            papers["citation_band"] = pd.cut(papers["citations"], bins=bins, labels=labels, include_lowest=True)
            cit_dist = papers["citation_band"].value_counts().sort_index()

            fig_citations = px.bar(
                x=labels,
                y=cit_dist.values,
                title="Citation Range Distribution",
                color=cit_dist.values,
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_citations, use_container_width=True)

            # --- Full Paper List ---
            st.markdown("### 📑 Full Paper List with Matched Quality Info")
            def highlight_rank(val):
                if val in ["Q1","A*"]:
                    return "background-color:#E3F2FD; font-weight:bold; color:#1E3A8A"
                elif val in ["Q2","A"]:
                    return "background-color:#D1FAE5; color:#065F46"
                elif val in ["Q3","B"]:
                    return "background-color:#FEF9C3; color:#92400E"
                else:
                    return "background-color:#F3F4F6; color:#374151"

            styled = papers[["title","venue","citations","match_type","matched_title","rank","match_score"]]\
                .sort_values(by="citations", ascending=False)\
                .style.applymap(highlight_rank, subset=["rank"])
            
            st.dataframe(styled, use_container_width=True)

            st.markdown("---")
            st.success(f"✅ Evaluation complete for **{results['name']}**. Total of {results['total_papers']} papers analyzed.")
            st.caption("📘 Powered by semantic embeddings, fuzzy matching, and citation analytics")
