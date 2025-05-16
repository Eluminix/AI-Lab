import pandas as pd

# Manuelle Definition der Paar-Daten
data = [{
    "pair_id": "a1_a2",
    "lang1": "en",
    "lang2": "en",
    "Geography": "Australia",
    "Entities": "",
    "Time": "2020",
    "Narrative": "climate",
    "Overall": "3",         # Skala je nach Aufgabe
    "Style": "neutral",
    "Tone": "serious"
}]

# Speichern als CSV
df = pd.DataFrame(data)
df.to_csv("train_links.csv", index=False)
