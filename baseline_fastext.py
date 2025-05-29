import os
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

# Download NLTK resources (only needs to be done once)
nltk.download('punkt')
nltk.download('punkt_tab')

# === Step 1: Basic preprocessing ===
def simple_preprocess(text):
    if not isinstance(text, str):
        text = ""
    return word_tokenize(text.lower())

# === Step 2: Compute average word embedding for a document ===
def average_embedding(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# === Step 3: Compute cosine similarity between two vectors ===
def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# === Step 4: Convert similarity score to class label (0 to 3) ===
def classify_similarity(sim):
    if sim >= 0.90:
        return 4
    elif sim >= 0.75:
        return 3
    elif sim >= 0.50:
        return 2
    else:
        return 1

# === Main comparison function ===
def compare_articles(text_a, text_b, model):
    tokens_a = simple_preprocess(text_a)
    tokens_b = simple_preprocess(text_b)

    vec_a = average_embedding(tokens_a, model)
    vec_b = average_embedding(tokens_b, model)

    sim = compute_similarity(vec_a, vec_b)
    label = classify_similarity(sim)

    return sim, label

def predict_classification_with_titles(X):
    predictions = []

    for _, row in X.iterrows():
        title_a = row['title1']
        title_b = row['title2']

        sim, label = compare_articles(title_a, title_b, model)
        predictions.append(label)

    return predictions

def predict_regression_with_titles(X):
    predictions = []

    for _, row in X.iterrows():
        title_a = row['title1']
        title_b = row['title2']

        sim, label = compare_articles(title_a, title_b, model)
        predictions.append(sim)

    return predictions

# === Example usage ===
if __name__ == "__main__":

    # Lade CSV mit allen Paaren
    print("Loading CSV...")
    df = pd.read_csv("data/full_dataset.csv")

    df["label_classification"] = df["overall"].round().astype(int)

    # # Load pre-trained Word2Vec model (Google News vectors)
    # print("Lade Word2Vec-Modell ...")
    # model = KeyedVectors.load_word2vec_format("pretrained_models/word2vec/GoogleNews-vectors-negative300.bin", binary=True)

    # Load FastText word vectors
    print("Loading FastText model...")
    if os.path.exists("pretrained_models\\fastext\\wiki-news-300d-1M-subword.kv"):
        model = KeyedVectors.load("pretrained_models\\fastext\\wiki-news-300d-1M-subword.kv")  # faster loading 
    else:
        model = KeyedVectors.load_word2vec_format("pretrained_models\\fastext\\wiki-news-300d-1M-subword.vec")
        model.save("pretrained_models\\fastext\\wiki-news-300d-1M-subword.kv")

    # Predict as Classification
    print("\nPredict article similarities as Classification using titles...")
    preds = predict_classification_with_titles(df)
    df['pred_class_title_only'] = preds

    acc = accuracy_score(df['label_classification'], df['pred_class_title_only'])
    print(f"Accuracy Classification (titles only): {acc:.3f}")

    # Predict as Regression
    print("\nPredict article similarities as Regression using titles...")
    preds = predict_regression_with_titles(df)
    df['pred_reg_title_only'] = preds
    # Scale from [0, 1] to [1, 4] range
    df['pred_reg_title_only'] = df['pred_reg_title_only'] * 3 + 1

    mse = mean_squared_error(df['overall'], df['pred_reg_title_only'])
    mae = mean_absolute_error(df['overall'], df['pred_reg_title_only'])
    r2 = r2_score(df['overall'], df['pred_reg_title_only'])

    print("Scores Regression (titles only):")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² : {r2:.4f}")

    print()
    print(df)
