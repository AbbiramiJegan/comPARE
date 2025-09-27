import os
import json
import serpapi
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
api_key = os.getenv('SERPAPI_KEY')  # Create a .env file with your api key and run this file
client = serpapi.Client(api_key=api_key)

def get_scholar_profile(author_id: str, max_pages: int = 50):
    profile = {
        "author_id": author_id,
        "name": None,
        "affiliations": None,
        "metrics": None,
        "co_authors": [],
        "articles": []
    }

    start = 0
    while True:
        # Call SerpAPI with pagination
        result = client.search(
            engine="google_scholar_author",
            author_id=author_id,
            hl="en",
            start=start
        )

        # Fill author details only once 
        if start == 0:
            profile["name"] = result["author"].get("name")
            profile["affiliations"] = result["author"].get("affiliations")
            profile["metrics"] = result["cited_by"]["table"]
            profile["co_authors"] = [co["name"] for co in result.get("co_authors", [])]

        articles = result.get("articles", [])
        if not articles:  # stop if no more articles
            break

        for article in articles:
            profile["articles"].append({
                "title": article.get("title"),
                "authors": article.get("authors"),
                "venue": article.get("publication"),
                "year": article.get("year"),
                "citations": article.get("cited_by", {}).get("value", 0)
            })

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
    scholar_data = get_scholar_profile("1xWR0IgAAAAJ")  # replace with any author_id
    print(f"Total articles collected: {len(scholar_data['articles'])}")