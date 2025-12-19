# fetchProfile.py

import os
import json
import serpapi
from dotenv import load_dotenv
from datetime import datetime
from name_variations import name_variations

load_dotenv()
api_key = os.getenv("SERPAPI_KEY")  # Create a .env file with your api key and run this file
client = serpapi.Client(api_key=api_key)


def get_author_pos(authors_str: str, variations) -> str:
    top = 1
    if not authors_str:
        return top

    # SerpAPI returns a comma-separated author string
    authors = [a.strip().lower() for a in authors_str.split(",") if a.strip()]

    pos = None
    for idx, author in enumerate(authors):
        if idx > top: 
            top = idx
        if author in variations:
            pos = idx + 1
            break

    if pos is None:
        # Not found in first 5 → treat as 5+
        return f"{top}+"

    else: 
        return str(pos)


def get_scholar_profile(author_id: str, max_pages: int = 50):
    profile = {
        "author_id": author_id,
        "name": None,
        "variations": None,
        "affiliations": None,
        "metrics": None,
        "co_authors": [],
        "articles": [],
    }

    start = 0
    while True:
        # Call SerpAPI with pagination
        result = client.search(
            engine="google_scholar_author",
            author_id=author_id,
            hl="en",
            start=start,
        )

        # Fill author details only once
        if start == 0:
            profile["name"] = result["author"].get("name")
            # Precompute name variations for matching
            profile["variations"] = name_variations(profile["name"] or "")
            profile["affiliations"] = result["author"].get("affiliations")
            profile["metrics"] = result["cited_by"]["table"]
            profile["co_authors"] = [co["name"] for co in result.get("co_authors", [])]

        articles = result.get("articles", [])
        if not articles:  # stop if no more articles
            break

        for article in articles:
            authors_str = article.get("authors", "") or ""
            author_pos_bucket = get_author_pos(authors_str, profile["variations"])

            profile["articles"].append(
                {
                    "title": article.get("title"),
                    "authors": authors_str,
                    "venue": article.get("publication"),
                    "year": article.get("year"),
                    "author_position": author_pos_bucket,  
                    "citations": article.get("cited_by", {}).get("value", 0),
                }
            )

        # Prepare next page
        start += 20
        if start >= max_pages * 20:  # safety stop
            break

    # Save to JSON
    os.makedirs("dataset", exist_ok=True)
    filename = f"dataset/{author_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)

    print(f"Profile saved to {filename}")
    return profile


if __name__ == "__main__":
    scholar_data = get_scholar_profile("oMjrLWcAAAAJ")  # replace with any author_id
    print(f"Total articles collected: {len(scholar_data['articles'])}")
    # Quick sanity check: print first few positions
    for a in scholar_data["articles"][:5]:
        print(a.get("year"), "→", a.get("author_position"), a.get("title"))
