import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')


def ensure_list(item):
    """Ensures any input becomes a list."""
    if isinstance(item, list):
        return item
    if isinstance(item, dict):
        return [item]
    return []


# Recursive function to search for rank/percentile keys in JSON
def search_for_keys(d, keys):
    found = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() in keys:
                found.append((k, v))
            found.extend(search_for_keys(v, keys))
    elif isinstance(d, list):
        for item in d:
            found.extend(search_for_keys(item, keys))
    return found


# -------------------------------------------------------------
# Search Journal by Title → get ISSN
# -------------------------------------------------------------
def search_scopus_venue(venue_title):
    url = "https://api.elsevier.com/content/search/scopus"
    params = {
        "query": f"SRCTITLE({venue_title})",
        "count": 5,
        "view": "STANDARD"
    }
    headers = {"X-ELS-APIKey": api_key}

    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        entries = resp.json().get("search-results", {}).get("entry", [])
        return entries[0] if entries else None
    except Exception as e:
        print(f"[Search Error] {e}")
        return None


# -------------------------------------------------------------
# Fetch Scopus Serial Metrics using ISSN
# -------------------------------------------------------------
def get_scopus_metrics_by_issn(issn):
    url = f"https://api.elsevier.com/content/serial/title/issn/{issn}"
    headers = {"X-ELS-APIKey": api_key}
    params = {"view": "STANDARD"}

    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        entry = data.get("serial-metadata-response", {}).get("entry", [{}])[0]

        # ---- Core Metrics ----
        cite_score = entry.get("citeScoreYearInfoList", {}).get("citeScoreCurrentMetric")
        sjr = entry.get("SJRList", {}).get("SJR", [{}])[0].get("$")
        snip = entry.get("SNIPList", {}).get("SNIP", [{}])[0].get("$")
        subjects = [s.get("$") for s in ensure_list(entry.get("subject-area"))]

        metrics = {
            "Title": entry.get("dc:title"),
            "Publisher": entry.get("dc:publisher"),
            "SJR": sjr,
            "SNIP": snip,
            "CiteScore": cite_score,
            "Subjects": subjects
        }

        # ---- Check for rank/percentile fields ----
        keys_to_check = ["rank", "ranktotal", "percentile", "quartile"]
        matches = search_for_keys(entry, keys_to_check)

        if matches:
            # If API had the data (rare)
            # For demonstration, just pick first match
            k, v = matches[0]
            metrics["Highest Percentile"] = str(v)
            metrics["Top Rank"] = "Found in API"
            metrics["Top Category"] = "Found in API"
        else:
            # No percentile/rank in API → set to N/A
            metrics["Highest Percentile"] = "N/A"
            metrics["Top Rank"] = "N/A"
            metrics["Top Category"] = "N/A"

        # Optional: print raw JSON for inspection
        print("----- RAW SCOPUS RESPONSE -----")
        print(json.dumps(data, indent=4))

        return metrics

    except Exception as e:
        print(f"[Metrics Error] {e}")
        return None


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    venue = input("Enter the paper venue title: ").strip()
    result = search_scopus_venue(venue)

    if not result:
        print("No venue found.")
        exit()

    issn = result.get("prism:issn")
    if not issn:
        print("ISSN not found.")
        exit()

    print(f"Found ISSN: {issn}")

    metrics = get_scopus_metrics_by_issn(issn)
    print("\nScopus Metrics:")
    print(json.dumps(metrics, indent=4))
