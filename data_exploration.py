from pathlib import Path
import json
import pandas as pd

def load_article_json(article_id, base_path="data"):
    """Find and load a single article JSON based on its ID."""
    matches = list(Path(base_path).rglob(f"{article_id}.json"))
    if not matches:
        return None
    with open(matches[0], "r", encoding="utf-8") as f:
        return json.load(f)

def build_text_pair_dataset(df, base_path="data", limit=None):
    """Build a DataFrame of article pairs including text, titles and similarity score."""
    data = []
    for idx, row in df.iterrows():
        if limit and idx >= limit:
            break
        try:
            id1, id2 = row["pair_id"].split("_")
            article1 = load_article_json(id1, base_path)
            article2 = load_article_json(id2, base_path)
            if article1 and article2:
                data.append({
                    "pair_id": row["pair_id"],
                    "title1": article1.get("title", ""),
                    "text1": article1.get("text", ""),
                    "title2": article2.get("title", ""),
                    "text2": article2.get("text", ""),
                    "label": float(row["Overall"])
                })
        except Exception as e:
            print(f"Error processing pair {row['pair_id']}: {e}")
            continue
    return pd.DataFrame(data)