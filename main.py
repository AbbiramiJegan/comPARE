import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import os
import json
import string
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz, process
from datetime import datetime
import torch

# --- Import local modules ---
try:
    from fetchProfile import get_scholar_profile
    from scopus import search_scopus_venue, get_scopus_metrics_by_issn
except ImportError as e:
    st.error(f"Critical Error: Could not import helper files. {e}")
    st.stop()

st.set_page_config(page_title="ComPARE Analytics", layout="wide")

# --- 🎨 CSS Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; color: #0F172A; }
        .stTextInput > div > div > input { font-size: 1.1rem; padding: 10px; border: 2px solid #CBD5E1; border-radius: 6px; }
        
        /* Metric Cards */
        div[data-testid="stMetric"] { 
            background-color: #FFFFFF; 
            border: 1px solid #E2E8F0; 
            border-left: 4px solid #1E3A8A; 
            padding: 15px; 
            border-radius: 6px; 
            box-shadow: 0 1px 2px rgba(0,0,0,0.05); 
        }
        
        h1, h2, h3 { color: #1E293B; font-weight: 700; letter-spacing: -0.02em; }
        hr { margin: 2.5rem 0; border: 0; border-top: 1px solid #E2E8F0; }
        
        /* Tooltip/Help styling */
        div[data-testid="stTooltipHoverTarget"] > svg { color: #64748B; }
    </style>
""", unsafe_allow_html=True)

# --- Colors (Complete) ---
ROLE_COLORS = {
    "Solo Author": "#D97706",   # Gold
    "First Author": "#1E3A8A",  # Navy
    "Last Author": "#059669",   # Emerald
    "Middle Author": "#CBD5E1"  # Grey
}

CP = { 
    "primary": "#1E3A8A", 
    "accent": "#3B82F6", 
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#64748B",
    "teal": "#0D9488",
    "indexed": "#8B5CF6"
}

# ---------------- Embedding model ----------------
@st.cache_resource
def load_model(): 
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- Normalization function ----------------
def normalize_text(s: str) -> str:
    """Normalize text for better matching"""
    s = s.lower()
    # Remove years (1900-2099)
    s = re.sub(r'\b(19|20)\d{2}\b', '', s)
    # Remove common prefixes
    s = re.sub(r'\b(proceedings?|proc\.?|conference|conf\.?|workshop|symposium|journal)\s+(of|on)?\s*', '', s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Enhanced stop words
    stop_words = ["international", "on", "of", "the", "and", "annual", "ieee", "acm"]
    tokens = [t for t in s.split() if t not in stop_words]
    return " ".join(tokens)

# ---------------- Load Data ----------------
@st.cache_data
def load_quality_data():
    def clean_columns(df):
        df.columns = [c.strip() for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()]

    try:
        journals = clean_columns(pd.read_csv("qualityJournal.csv"))
        conf_main = clean_columns(pd.read_csv("qualityConferences.csv"))
        conf_journal = clean_columns(pd.read_csv("qualityConferences-journal.csv"))
    except FileNotFoundError:
        st.error("Database files missing.")
        st.stop()

    for df in [conf_main, conf_journal]:
        if "Title" not in df.columns:
            for col in ["Conference Name (DBLP)", "CORE Conference Name", "GS Name"]:
                if col in df.columns: 
                    df.rename(columns={col: "Title"}, inplace=True)

    conf_main = conf_main.loc[:, ~conf_main.columns.duplicated()]
    conf_journal = conf_journal.loc[:, ~conf_journal.columns.duplicated()]
    conferences = pd.concat([conf_main, conf_journal], ignore_index=True)
    
    # Add normalized titles
    journals["Title"] = journals["Title"].astype(str)
    conferences["Title"] = conferences["Title"].astype(str)
    journals["Title_norm"] = journals["Title"].apply(normalize_text)
    conferences["Title_norm"] = conferences["Title"].apply(normalize_text)
    
    # NEW: Add acronym column for conferences (extract text in parentheses)
    def extract_acronym(title):
        match = re.search(r'\(([A-Z]{3,})\)', title)
        return match.group(1).lower() if match else None
    
    conferences["Acronym"] = conferences["Title"].apply(extract_acronym)
    
    return journals, conferences

journals, conferences = load_quality_data()

@st.cache_data
def precompute_embeddings():
    journal_embs = model.encode(journals["Title"].astype(str).tolist(), convert_to_tensor=True)
    conf_embs = model.encode(conferences["Title"].astype(str).tolist(), convert_to_tensor=True)
    return journal_embs, conf_embs

journal_embs, conf_embs = precompute_embeddings()

# ---------------- Logic ----------------
def extract_author_id(url):
    if "scholar.google" not in url: return None
    match = re.search(r"user=([\w-]+)", url)
    return match.group(1) if match else None

def load_or_fetch_author(author_id):
    os.makedirs("dataset", exist_ok=True)
    files = [f for f in os.listdir("dataset") if f.startswith(author_id)]
    if files:
        latest = max(files, key=lambda x: os.path.getmtime(os.path.join("dataset", x)))
        with open(os.path.join("dataset", latest), "r", encoding="utf-8") as f: 
            return json.load(f)
    return get_scholar_profile(author_id)

def semantic_match(venue, choices, choice_embs, top_k=10):
    """Return top_k semantic matches as (index, score) tuples"""
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

def guesser(venue: str) -> str:
    """Guess if venue is conference or journal based on keywords"""
    v = venue.lower()
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
    """Enhanced venue matching with multi-stage approach"""
    if pd.isna(venue) or not isinstance(venue, str) or venue.strip() == "":
        return None, None, None, 0, "None"
    
    venue_norm = normalize_text(venue)
    venue_tokens = set(venue_norm.split())
    hint = guesser(venue)
    
    # NEW: Extract potential acronym from venue
    venue_acronym_match = re.search(r'\b([A-Z]{3,})\b', venue)
    venue_acronym = venue_acronym_match.group(1).lower() if venue_acronym_match else None
    
    # Determine search order
    if hint == "conference":
        first_df, first_kind = conferences, "Conference"
        second_df, second_kind = journals, "Journal"
    elif hint == "journal":
        first_df, first_kind = journals, "Journal"
        second_df, second_kind = conferences, "Conference"
    else:
        if is_cs_ai:
            first_df, first_kind = conferences, "Conference"
            second_df, second_kind = journals, "Journal"
        else:
            first_df, first_kind = journals, "Journal"
            second_df, second_kind = conferences, "Conference"
    
    # Stage 0: Acronym exact match (NEW - only for conferences with acronyms)
    if venue_acronym:
        # Check conferences first if available
        if "Acronym" in conferences.columns:
            acronym_matches = conferences[conferences["Acronym"] == venue_acronym]
            if not acronym_matches.empty:
                row = acronym_matches.iloc[0]
                rank = row.get("CORE 2021 Rank", "-")
                return "Conference", row["Title"], rank, 100.0, "CSV (Acronym)"
    
    # Stage 1: Exact or substring match
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
                rank = df.iloc[idx].get("SCIMAGO Best Quartile 2023", "-")
            else:
                rank = df.iloc[idx].get("CORE 2021 Rank", "-")
            
            return kind, row["Title"], rank, overlap_score, "CSV (Exact)"
        
        return None
    
    res = exact_or_substring_match(first_df, first_kind)
    if res:
        return res
    res = exact_or_substring_match(second_df, second_kind)
    if res:
        return res
    
    # Stage 2: Semantic + Fuzzy hybrid (LOWERED THRESHOLDS)
    def pick_from_semantic(df, embs, kind):
        candidates = semantic_match(venue, df["Title"], embs, top_k=10)
        best = None
        best_score = -1.0
        
        for idx, score in candidates:
            row = df.iloc[idx]
            title = str(row["Title"])
            title_norm = row.get("Title_norm", normalize_text(title))
            
            fuzzy_score = fuzz.ratio(venue_norm, title_norm)
            
            # IMPROVED: Lowered thresholds: semantic >= 0.70 OR (semantic >= 0.60 AND fuzzy >= 75)
            if score >= 0.70 or (score >= 0.60 and fuzzy_score >= 75):
                sim_percent = round(float(score) * 100, 1)
                if kind == "Journal":
                    rank = row.get("SCIMAGO Best Quartile 2023", "-")
                else:
                    rank = row.get("CORE 2021 Rank", "-")
                
                if score > best_score:
                    best_score = float(score)
                    best = (kind, title, rank, sim_percent, "CSV (AI+Fuzzy)")
        
        return best
    
    first_embs = conf_embs if first_kind == "Conference" else journal_embs
    second_embs = conf_embs if second_kind == "Conference" else journal_embs
    
    res = pick_from_semantic(first_df, first_embs, first_kind)
    if res:
        return res
    res = pick_from_semantic(second_df, second_embs, second_kind)
    if res:
        return res
    
    # Stage 3: Pure fuzzy fallback (LOWERED THRESHOLD)
    def best_fuzzy(df, kind):
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
            rank = row.get("CORE 2021 Rank", "-")
        return kind, match, rank, score, "CSV (Fuzzy)"
    
    best_j = best_fuzzy(journals, "Journal")
    best_c = best_fuzzy(conferences, "Conference")
    
    best = None
    if best_j and best_c:
        best = best_j if best_j[3] >= best_c[3] else best_c
    else:
        best = best_j or best_c
    
    # IMPROVED: Lowered threshold from 80 to 75
    if best and best[3] >= 75:
        return best
    
    # Stage 4: Scopus API fallback
    try:
        if len(venue) > 5:
            scopus_venue = search_scopus_venue(venue)
            if scopus_venue:
                issn = scopus_venue.get("prism:issn")
                if issn:
                    first_issn = issn.split(" ")[0] if isinstance(issn, str) else issn
                    metrics = get_scopus_metrics_by_issn(first_issn)
                    if metrics:
                        percentile_str = metrics.get("Highest Percentile", "0")
                        try:
                            p = float(percentile_str)
                            quartile = "Q1" if p >= 75 else "Q2" if p >= 50 else "Q3" if p >= 25 else "Q4"
                            return "Journal", metrics.get("Title"), quartile, 100, "Scopus API"
                        except ValueError:
                            if metrics.get("SJR"):
                                return "Journal", metrics.get("Title"), "Indexed", 100, "Scopus API"
    except Exception:
        pass
    
    return "Unranked", best[1] if best else None, "-", best[3] if best else 0, "Failed"

def evaluate_author_data(data, cs_ai):
    df = pd.DataFrame(data["articles"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0).astype(int)
    
    # Match Quality
    df[["match_type","matched_title","rank","match_score", "source"]] = df.apply(
        lambda r: pd.Series(match_quality(r["venue"], cs_ai)), axis=1
    )
    
    # Metrics
    citations = sorted(df["citations"].tolist(), reverse=True)
    h_index = sum(x >= i + 1 for i, x in enumerate(citations))
    i10_index = sum(x >= 10 for x in citations)
    g_index = 0
    for i in range(len(citations)):
        if sum(citations[:i+1]) >= (i+1)**2: 
            g_index = i + 1
    
    min_year = df["year"].min()
    academic_age = datetime.now().year - min_year if pd.notna(min_year) else 1
    if academic_age < 1: 
        academic_age = 1

    # Smart Author Cleaning
    target_name_parts = data["name"].lower().split()
    target_last = target_name_parts[-1]

    all_authors = []
    for auth_str in df["authors"].dropna():
        names = [n.strip() for n in str(auth_str).split(",")]
        clean_names = [n for n in names if "..." not in n and len(n) > 1]
        all_authors.extend(clean_names)
    
    unique_authors = set(all_authors)
    unique_authors = {a for a in unique_authors if target_last not in a.lower()}
    network_size = len(unique_authors)

    # Role Logic
    def get_role(authors_str):
        if not authors_str: 
            return "Unknown"
        parts = [p.strip().lower() for p in str(authors_str).split(",")]
        parts = [p for p in parts if "..." not in p]
        if not parts: 
            return "Unknown"
        
        if len(parts) == 1:
            if target_last in parts[0]: 
                return "Solo Author"
            return "Unknown"
        if target_last in parts[0]: 
            return "First Author"
        if target_last in parts[-1]: 
            return "Last Author"
        return "Middle Author"

    df["role"] = df["authors"].apply(get_role)
    lead_count = df[df["role"].isin(["Solo Author", "First Author"])].shape[0]
    leadership_score = round((lead_count / len(df)) * 100, 1) if len(df) > 0 else 0

    current_year = datetime.now().year
    recent_df = df[df["year"] >= (current_year - 5)]
    recent_p = len(recent_df)
    recent_c = recent_df["citations"].sum()

    total_c = df["citations"].sum()
    max_c = df["citations"].max() if len(df) > 0 else 0
    one_hit = round((max_c / total_c) * 100, 1) if total_c > 0 else 0
    
    res = {
        "id": data["author_id"], 
        "name": data["name"],
        "total_p": len(df), 
        "total_c": total_c,
        "h_index": h_index, 
        "i10_index": i10_index, 
        "g_index": g_index,
        "academic_age": int(academic_age),
        "cpp": round(df["citations"].mean(), 1) if len(df) > 0 else 0,
        "leadership_score": leadership_score,
        "network_size": network_size,
        "recent_p": recent_p, 
        "recent_c": recent_c,
        "one_hit": one_hit
    }
    return res, df

# ---------------- MAIN LAYOUT ----------------
st.title("ComPARE Intelligence")
st.markdown("### Researcher Integrity & Performance Toolkit")

c_input, c_toggle = st.columns([4, 1])
with c_input:
    url = st.text_input("Researcher Profile URL", placeholder="Paste Google Scholar link here...")
with c_toggle:
    st.write("") 
    st.write("") 
    is_cs_ai = st.toggle("CS/AI Mode", value=True, help="Optimizes ranking logic for Computer Science conferences")

if url:
    aid = extract_author_id(url)
    if not aid:
        st.error("Invalid Link.")
    else:
        data = load_or_fetch_author(aid)
        if not data or not data.get("articles"):
            st.error("No data found.")
        else:
            with st.spinner("Processing analytics..."):
                meta, df = evaluate_author_data(data, is_cs_ai)
                
            st.markdown("---")
            
            # --- HEADER INFO ---
            h1, h2 = st.columns([1, 3])
            with h1:
                st.metric("Researcher", meta["name"])
                st.caption(f"ID: {meta['id']} | Active for {int(meta['academic_age'])} Years")
            with h2:
                # ROW 1: CORE ACADEMIC METRICS
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Papers", meta["total_p"], help="Total count of publications found.")
                m2.metric("Total Citations", f"{meta['total_c']:,}", help="Sum of all citations received.", delta=f"+{meta['recent_c']} recent")
                m3.metric("H-Index", meta["h_index"], help="Productivity/Impact Balance: X papers have at least X citations.")
                m4.metric("i10-Index", meta["i10_index"], help="Consistency: Number of papers with at least 10 citations.")
                m5.metric("g-Index", meta["g_index"], help="High Impact Weight: Gives credit for highly-cited papers (better than H-index for superstars).")

            # ROW 2: INTEGRITY & DEPTH METRICS
            st.markdown("##### Integrity & Health Scorecard")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Recent Papers", f"{meta['recent_p']}", help="Volume of work published in the last 5 years.")
            d2.metric("One-Hit Wonder", f"{meta['one_hit']}%", help="Concentration risk: % of total citations coming from just the top paper. Lower is better.", delta_color="inverse")
            d3.metric("Network Size", f"{meta['network_size']}", help="Reach: Count of unique co-authors/collaborators.") 
            d4.metric("Leadership Score", f"{meta['leadership_score']}%", help="Independence: % of papers where they are the First or Solo Author.")

            st.markdown("---")

            # SECTION 1: INTEGRITY
            st.subheader("Integrity & Productivity Audit")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Contribution Role Analysis**")
                st.caption("Breakdown of author position. High volume of 'First Author' papers is a productivity indicator.")
                role_counts = df.groupby(["year", "role"]).size().reset_index(name="count")
                fig_prod = px.bar(role_counts, x="year", y="count", color="role", 
                                  color_discrete_map=ROLE_COLORS)
                fig_prod.add_hline(y=15, line_dash="dot", line_color=CP["danger"], annotation_text="High Volume Limit")
                fig_prod.update_layout(plot_bgcolor="white", barmode='stack', xaxis_title="Year", yaxis_title="Papers")
                st.plotly_chart(fig_prod, use_container_width=True)

            with c2:
                st.markdown("**Productivity vs. Impact**")
                st.caption("Comparison of output volume (Bars) versus citation impact (Line) over time.")
                yearly_stats = df.groupby("year").agg(papers=('title', 'count'), citations=('citations', 'sum')).reset_index()
                yearly_stats = yearly_stats[yearly_stats["year"] > 1980]
                fig_dual = go.Figure()
                fig_dual.add_trace(go.Bar(x=yearly_stats["year"], y=yearly_stats["papers"], name="Papers", marker_color="#CBD5E1", opacity=0.5, yaxis="y"))
                fig_dual.add_trace(go.Scatter(x=yearly_stats["year"], y=yearly_stats["citations"], name="Citations", mode='lines+markers', line=dict(color=CP["primary"], width=3), yaxis="y2"))
                fig_dual.update_layout(plot_bgcolor="white", yaxis=dict(title="Papers", side="left", showgrid=False), yaxis2=dict(title="Citations", side="right", overlaying="y", showgrid=True), xaxis=dict(title="Year"), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_dual, use_container_width=True)

            # SECTION 2: IMPACT & QUALITY
            st.subheader("Impact & Topics")
            c3, c4 = st.columns([2, 1])
            with c3:
                st.markdown("**Impact Reality Check (Scatter)**")
                st.caption("Distribution of papers by Year and Citations (Log Scale). Dots colored by Venue Type.")
                scatter_df = df[df["citations"] > 0]
                fig_scatter = px.scatter(
                    scatter_df, x="year", y="citations", color="match_type", size="citations",
                    hover_data=["title", "venue"], 
                    color_discrete_sequence=[CP["primary"], CP["teal"], CP["neutral"]]
                )
                fig_scatter.update_layout(yaxis_type="log", plot_bgcolor="white", xaxis_title="Year", yaxis_title="Citations (Log)")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with c4:
                st.markdown("**Top Research Keywords**")
                st.caption("Most frequent terms in paper titles (excluding common academic stopwords).")
                all_titles = " ".join(df["title"].dropna()).lower()
                stopwords = set(['the','and','of','a','in','for','on','with','to','an','at','using','based','method','analysis','system','approach','study','review','new','framework','application','via','towards','survey','design','development','comparative','performance','evaluation','implementation','algorithm','model','data','user','proposed'])
                words = [w for w in re.findall(r'\w+', all_titles) if w not in stopwords and len(w) > 3]
                if words:
                    kw_df = pd.DataFrame(Counter(words).most_common(10), columns=["Keyword", "Frequency"])
                    fig_kw = px.bar(kw_df, x="Frequency", y="Keyword", orientation='h', color="Frequency", color_continuous_scale="Teal")
                    fig_kw.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white")
                    st.plotly_chart(fig_kw, use_container_width=True)

            # SECTION 3: VENUE QUALITY (ENHANCED VISUALS)
            st.subheader("Venue Quality Analysis")

            # Publication Type Pie Chart
            st.markdown("### 📊 Publication Type Distribution")
            st.caption("Breakdown of publication venues (Journal • Conference • Unranked)")

            venue_counts = df["match_type"].value_counts()

            fig_pie = px.pie(
                values=venue_counts.values,
                names=venue_counts.index,
                hole=0.45,
                color_discrete_sequence=[
                    "#5A6FF0",   # Vibrant indigo
                    "#FF7C7C",   # Coral red
                    "#4FD3C4",   # Aqua green
                    "#F8C537",   # Golden yellow
                ],
            )

            fig_pie.update_traces(
                textinfo="percent",
                pull=[0.08] * len(venue_counts),
                rotation=140,
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}",
                marker=dict(
                    line=dict(color="white", width=2.5),
                )
            )

            fig_pie.update_layout(
                showlegend=True,
                legend_title_text="Venue Type",
                margin=dict(t=50, b=20),
                annotations=[
                    dict(
                        text="Venue Mix",
                        x=0.5, y=0.5,
                        font=dict(size=16, color="#333", family="Arial Black"),
                        showarrow=False,
                    )
                ],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
     
            q1, q2 = st.columns(2)
            with q1:
                st.markdown("**Journal Quartiles**")
                st.caption("Distribution of Journals by SCImago Quartile Rank.")
                j_df = df[df["match_type"]=="Journal"]
                q_counts = j_df["rank"].value_counts()
                # Add any missing expected ranks
                for rank in ["Q1","Q2","Q3","Q4","Indexed"]:
                    if rank not in q_counts:
                        q_counts[rank] = 0
                # Reindex to include both expected and unexpected ranks
                all_ranks = ["Q1","Q2","Q3","Q4","Indexed"] + [r for r in q_counts.index if r not in ["Q1","Q2","Q3","Q4","Indexed"]]
                q_counts = q_counts.reindex(all_ranks, fill_value=0)
                
                color_map = {"Q1": CP["primary"], "Q2": CP["accent"], "Q3": CP["warning"], "Q4": CP["danger"], "Indexed": CP["indexed"]}
                fig_q = px.bar(x=q_counts.index, y=q_counts.values, color=q_counts.index, color_discrete_map=color_map)
                fig_q.update_layout(plot_bgcolor="white", showlegend=False, yaxis_title="Count")
                st.plotly_chart(fig_q, use_container_width=True)

            with q2:
                st.markdown("**Conference Ranks**")
                st.caption("Distribution of Conferences by CORE Rank.")
                c_df = df[df["match_type"]=="Conference"]
                c_counts = c_df["rank"].value_counts()
                # Add any missing expected ranks
                for rank in ["A*","A","B","C","Indexed"]:
                    if rank not in c_counts:
                        c_counts[rank] = 0
                # Reindex to include both expected and unexpected ranks
                all_ranks = ["A*","A","B","C","Indexed"] + [r for r in c_counts.index if r not in ["A*","A","B","C","Indexed"]]
                c_counts = c_counts.reindex(all_ranks, fill_value=0)
                
                conf_colors = {"A*": CP["primary"], "A": CP["accent"], "B": CP["warning"], "C": CP["danger"], "Indexed": CP["indexed"]}
                fig_c = px.bar(x=c_counts.index, y=c_counts.values, color=c_counts.index, color_discrete_map=conf_colors)
                fig_c.update_layout(plot_bgcolor="white", showlegend=False, yaxis_title="Count")
                st.plotly_chart(fig_c, use_container_width=True)

            # SECTION 4: DATA TABLE
            st.markdown("---")
            with st.expander("📑 View Full Paper List & Data Source"):
                def color_rank(val):
                    if val in ["Q1", "A*"]: return "background-color: #DBEAFE; color: #1E3A8A; font-weight: bold"
                    if val in ["Q2", "A"]: return "background-color: #DCFCE7; color: #166534; font-weight: bold"
                    if val in ["Q3", "B"]: return "background-color: #FEF3C7; color: #92400E"
                    if val in ["Q4", "C"]: return "background-color: #FEE2E2; color: #991B1B"
                    if val == "Indexed": return "background-color: #F3E8FF; color: #6B21A8"
                    return ""

                clean_df = df.copy()
                clean_df["year"] = clean_df["year"].fillna(0).astype(int).astype(str)
                clean_df["year"] = clean_df["year"].replace("0", "N/A")

                table_df = clean_df[["title", "venue", "year", "citations", "rank", "match_type", "source"]]
                
                st.dataframe(
                    table_df.style.applymap(color_rank, subset=["rank"]), 
                    use_container_width=True, 
                    height=600,
                    column_config={
                        "year": st.column_config.TextColumn("Year"),
                        "citations": st.column_config.NumberColumn("Citations", format="%d")
                    }
                )
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV Report", csv, f"report_{meta['name']}.csv", "text/csv")