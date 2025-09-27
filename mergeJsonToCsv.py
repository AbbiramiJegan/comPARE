import os
import json
import pandas as pd

# Path where JSONs are stored
DATASET_FOLDER = "dataset"

author_rows = []
article_rows = []

for filename in os.listdir(DATASET_FOLDER):
    if filename.endswith(".json"):
        filepath = os.path.join(DATASET_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # --- Author-level row ---
        author_rows.append({
            "author_id": data.get("author_id"),
            "name": data.get("name"),
            "affiliations": data.get("affiliations"),
            "citations_all": data["metrics"][0]["citations"]["all"] if "metrics" in data else None,
            "citations_since_2020": data["metrics"][0]["citations"]["since_2020"] if "metrics" in data else None,
            "h_index_all": data["metrics"][1]["h_index"]["all"] if "metrics" in data else None,
            "h_index_since_2020": data["metrics"][1]["h_index"]["since_2020"] if "metrics" in data else None,
            "i10_index_all": data["metrics"][2]["i10_index"]["all"] if "metrics" in data else None,
            "i10_index_since_2020": data["metrics"][2]["i10_index"]["since_2020"] if "metrics" in data else None,
            "co_authors": ", ".join(data.get("co_authors", []))
        })
        
        # --- Article-level rows ---
        for article in data.get("articles", []):
            article_rows.append({
                "author_id": data.get("author_id"),
                "author_name": data.get("name"),
                "title": article.get("title"),
                "authors": article.get("authors"),
                "venue": article.get("venue"),
                "year": article.get("year"),
                "citations": article.get("citations", 0)
            })

# Save authors.csv
df_authors = pd.DataFrame(author_rows)
df_authors.to_csv("authors.csv", index=False, encoding="utf-8-sig")

# Save articles.csv
df_articles = pd.DataFrame(article_rows)
df_articles.to_csv("articles.csv", index=False, encoding="utf-8-sig")

print("Merging complete: authors.csv & articles.csv generated!")
