# Import streamlit library -> for UI
import streamlit as st
# Import pandas (dataframes) -> for reading and manipulating CSV data
import pandas as pd
# Regular expressions module -> Extract author id
import re
# fuzz -> String similarity helper & process -> fallback when embeddings don't give high confidence
from fuzzywuzzy import fuzz, process
# ST -> To load an embedding model & util -> helper functions
from sentence_transformers import SentenceTransformer, util
# PyTorch
import torch

# page_title sets the browser tab title
# layout makes the page wider 
st.set_page_config(page_title="Scholar Report Card", layout="wide")

# ---------------- Load datasets ----------------
authors = pd.read_csv("Authors.csv")
articles = pd.read_csv("Articles.csv")

# Journals
journals = pd.read_csv("qualityJournal.csv")
# Conferences (main)
conf_main = pd.read_csv("qualityConferences.csv")
# Conferences (main + journal versions)
conf_journal = pd.read_csv("qualityConferences-journal.csv")

# Normalize column names for conf_main
if "Title" not in conf_main.columns:
    if "Conference Name (DBLP)" in conf_main.columns:
        conf_main.rename(columns={"Conference Name (DBLP)": "Title"}, inplace=True)
    elif "CORE Conference Name" in conf_main.columns:
        conf_main.rename(columns={"CORE Conference Name": "Title"}, inplace=True)
    elif "ERA Conference Name" in conf_main.columns:
        conf_main.rename(columns={"ERA Conference Name": "Title"}, inplace=True)
    elif "GS Name" in conf_main.columns:
        conf_main.rename(columns={"GS Name": "Title"}, inplace=True)

# Normalize column names for conf_journal
if "Title" not in conf_journal.columns:
    if "Conference Name (DBLP)" in conf_journal.columns:
        conf_journal.rename(columns={"Conference Name (DBLP)": "Title"}, inplace=True)
    elif "ERA Conference Name" in conf_journal.columns:
        conf_journal.rename(columns={"ERA Conference Name": "Title"}, inplace=True)
    elif "GS Name" in conf_journal.columns:
        conf_journal.rename(columns={"GS Name": "Title"}, inplace=True)

# Merge the two conference sources
conferences = pd.concat([conf_main, conf_journal], ignore_index=True)

# ---------------- Embedding model ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Precompute embeddings for journals & conferences
@st.cache_resource
def precompute_embeddings():
    journal_embs = model.encode(journals["Title"].astype(str).tolist(), convert_to_tensor=True)
    conf_embs = model.encode(conferences["Title"].astype(str).tolist(), convert_to_tensor=True)
    return journal_embs, conf_embs

journal_embs, conf_embs = precompute_embeddings()

# ---------------- Helper functions ----------------
def extract_author_id(url):
    """Extract author_id from Google Scholar URL."""
    match = re.search(r"user=([\w-]+)", url)
    return match.group(1) if match else None

def semantic_match(venue, choices, choice_embs, top_k=5):
    """Semantic search with embeddings + cosine similarity."""
    if not isinstance(venue, str) or venue.strip() == "":
        return None, 0, None
    venue_emb = model.encode(venue, convert_to_tensor=True)
    cos_scores = util.cos_sim(venue_emb, choice_embs)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    best_idx = top_results.indices[0].item()
    best_score = top_results.values[0].item()
    return choices.iloc[best_idx], best_score, best_idx

def match_quality(venue, is_cs_ai=False):
    """Match venue to Journals/Conferences using embeddings + fuzzy fallback."""
    if pd.isna(venue):
        return None, None, None, 0

    # --- CS/AI first: Conferences ---
    if is_cs_ai:
        choice, score, idx = semantic_match(venue, conferences["Title"], conf_embs)
        if choice and score >= 0.7:  # semantic threshold
            row = conferences.iloc[idx]
            return "Conference", choice, row.get("CORE 2021 Rank", row.get("Rank", "-")), round(score*100, 1)
        # fallback to journals
        choice, score, idx = semantic_match(venue, journals["Title"], journal_embs)
        if choice and score >= 0.7:
            row = journals.iloc[idx]
            return "Journal", choice, row.get("SCIMAGO Best Quartile 2023", "-"), round(score*100, 1)

    # --- Non-CS/AI: Journals first ---
    choice, score, idx = semantic_match(venue, journals["Title"], journal_embs)
    if choice and score >= 0.7:
        row = journals.iloc[idx]
        return "Journal", choice, row.get("SCIMAGO Best Quartile 2023", "-"), round(score*100, 1)

    # fallback to conferences
    choice, score, idx = semantic_match(venue, conferences["Title"], conf_embs)
    if choice and score >= 0.7:
        row = conferences.iloc[idx]
        return "Conference", choice, row.get("CORE 2021 Rank", row.get("Rank", "-")), round(score*100, 1)

    # Final fallback → fuzzy match
    result = process.extractOne(
        venue,
        pd.concat([journals["Title"], conferences["Title"]]).astype(str).tolist()
    )
    if result:
        match, fuzzy_score = result[0], result[1]
        return "Fuzzy", match, "-", fuzzy_score
    else:
        return "Fuzzy", None, "-", 0

def evaluate_author(author_id, authors, articles, cs_ai=False):
    if author_id not in authors["author_id"].values:
        return None, None

    author_name = authors.loc[authors["author_id"] == author_id, "name"].values[0]
    author_articles = articles[articles["author_id"] == author_id].copy()

    # Summary stats
    total_papers = len(author_articles)
    total_citations = author_articles["citations"].sum()
    cpp = total_citations / total_papers if total_papers else 0
    recent_articles = author_articles[author_articles["year"] >= 2020]
    recent_papers = len(recent_articles)
    recent_citations = recent_articles["citations"].sum()
    top5 = author_articles.sort_values("citations", ascending=False).head(5)
    top5_avg = top5["citations"].mean() if not top5.empty else 0

    # Match venues
    author_articles[["match_type", "matched_title", "rank", "match_score"]] = author_articles.apply(
        lambda row: pd.Series(match_quality(row["venue"], is_cs_ai=cs_ai)),
        axis=1
    )

    results = {
        "author_id": author_id,
        "name": author_name,
        "total_papers": total_papers,
        "total_citations": total_citations,
        "cpp": round(cpp, 2),
        "recent_papers": recent_papers,
        "recent_citations": recent_citations,
        "top5_avg_citations": round(top5_avg, 2)
    }

    return results, author_articles

# ---------------- Streamlit UI ----------------
st.title("📊 Researcher Quality Report Card")

url = st.text_input("Enter Google Scholar Profile Link:")
is_cs_ai = st.checkbox("💻 Is this a CS/AI researcher? (prefer conferences)")

if url:
    author_id = extract_author_id(url)
    if not author_id:
        st.error("❌ Could not extract author ID from URL")
    else:
        results, papers = evaluate_author(author_id, authors, articles, cs_ai=is_cs_ai)
        if results is None:
            st.error("❌ Author ID not found in Authors.csv")
        else:
            st.subheader(f"👤 Report Card for **{results['name']}**")

            # --- Stats cards ---
            st.markdown("### 📊 Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Papers", results['total_papers'])
            col2.metric("Total Citations", results['total_citations'])
            col3.metric("Citations per Paper", results['cpp'])

            col4, col5, col6 = st.columns(3)
            col4.metric("Recent Papers (>=2020)", results['recent_papers'])
            col5.metric("Recent Citations", results['recent_citations'])
            col6.metric("Top 5 Avg Citations", results['top5_avg_citations'])

            # Count by venue type
            st.markdown("### Paper Counts by Venue Type")
            venue_counts = papers["match_type"].value_counts()
            st.table(venue_counts)
            st.bar_chart(venue_counts)

            # Count by journal quartile / conference rank
            st.markdown("### Paper Counts by Rank / Quartile")
            def rank_category(row):
                if row["match_type"] == "Journal":
                    return row["rank"] if pd.notna(row["rank"]) else "Unranked"
                elif row["match_type"] == "Conference":
                    return row["rank"] if pd.notna(row["rank"]) else "Unranked"
                else:
                    return "Other"
            rank_counts = papers.apply(rank_category, axis=1).value_counts()
            st.table(rank_counts)
            st.bar_chart(rank_counts)

            # --- Papers table ---
            with st.expander("📑 Full Paper List with Quality Info", expanded=False):
                def highlight_rank(val):
                    if val in ["Q1", "A*"]:
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
