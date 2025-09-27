import pandas as pd 

authors = pd.read_csv("Authors.csv") 
articles = pd.read_csv("Articles.csv")

def evaluate_author(author_id, authors, articles, pretty_print=True):
    """Evaluate research performance of a single author."""

    # Get author name
    author_name = authors.loc[authors["author_id"] == author_id, "name"].values[0]

    # Filter articles belonging to this author
    author_articles = articles[articles["author_id"] == author_id]

    # Total publications
    total_papers = int(len(author_articles))

    # Total citations
    total_citations = int(author_articles["citations"].sum())

    # Citations per paper 
    cpp = round(total_citations / total_papers, 2) if total_papers > 0 else 0.00

    # Recent performance (since 2020)
    recent_articles = author_articles[author_articles["year"] >= 2020]
    recent_papers = int(len(recent_articles))
    recent_citations = int(recent_articles["citations"].sum())

    # Top papers impact 
    if total_papers > 0:
        top5 = author_articles.sort_values(by="citations", ascending=False).head(5)
        top5_avg_citations = round(top5["citations"].mean(), 2)
    else:
        top5_avg_citations = 0.00

    # Co-author network 
    coauthors_map = {}
    for a in author_articles["authors"]:
        for person in a.split(","):
            person_clean = person.strip()
            if person_clean and person_clean.lower() != author_name.lower():
                coauthors_map.setdefault(person_clean.lower(), person_clean)
    coauthors = set(coauthors_map.values())
    coauthor_count = int(len(coauthors))

    # Store results
    results = {
        "author_id": author_id,
        "name": author_name,
        "total_papers": total_papers,
        "total_citations": total_citations,
        "cpp": cpp,
        "recent_papers": recent_papers,
        "recent_citations": recent_citations,
        "top5_avg_citations": top5_avg_citations,
        "coauthor_count": coauthor_count,
        "coauthors": sorted(coauthors),
    }

    # Print results
    if pretty_print:
        print("=" * 50)
        print(" Author Performance Report ")
        print("=" * 50)
        print(f"Name: {author_name}")
        print(f"Author ID: {author_id}")
        print("-" * 50)
        print(f"Total Papers: {total_papers}")
        print(f"Total Citations: {total_citations}")
        print(f"Citations per Paper: {cpp:.2f}")
        print(f"Recent Papers (>=2020): {recent_papers}")
        print(f"Recent Citations: {recent_citations}")
        print(f"Top 5 Avg Citations: {top5_avg_citations:.2f}")
        print(f"Co-Author Network Size: {coauthor_count}")
        print("Co-Authors:")
        for c in sorted(coauthors):
            print(f"   - {c}")
        print("=" * 50)

    return results

# Main
evaluate_author("oMjrLWcAAAAJ", authors, articles)
