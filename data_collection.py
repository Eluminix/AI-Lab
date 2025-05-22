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
                    "overall": float(row["Overall"]),
                    "overall_classification": int(round(float(row["Overall"]))),
                    "geography": float(row["Geography"]),
                    "entities": float(row["Entities"]),
                    "time": float(row["Time"]),
                    "narrative": float(row["Narrative"]),
                    "style": float(row["Style"]),
                    "tone": float(row["Tone"]),
                })
        except Exception as e:
            print(f"Error processing pair {row['pair_id']}: {e}")
            continue
    return pd.DataFrame(data)


if __name__ == "__main__":

    print("Reading CSV...")

    # Path to the CSV file
    csv_path = "data/semeval-2022_task8_train-data_batch.csv"

    # Load the data
    df = pd.read_csv(csv_path)

    print("Combining CSV with JSON Data...")

    # Process csv and json for all data
    dataset_df = build_text_pair_dataset(df, base_path="data")

    dataset_df.to_csv("data/full_dataset.csv", index=False)