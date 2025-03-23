import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ibm_watsonx_ai.foundation_models import Embeddings
import re
from tqdm import tqdm
from config import conf_dict

# Load evaluation file
df = pd.read_csv("healthcare_examples_with_all_models.csv")

# Helper to parse JSON safely
def parse_json_safe(s):
    if not isinstance(s, str) or s.strip().startswith("ERROR:"):
        return None
    try:
        # Attempt to decode any double-encoded JSON string
        while isinstance(s, str):
            s = json.loads(s)
        return s if isinstance(s, dict) else None
    except:
        return None


# Flatten nested JSON
def flatten(d, parent=''):
    items = {}
    for k, v in d.items():
        new_key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.update(flatten({f"{new_key}[{i}]": item}))
        else:
            items[new_key] = v
    return items

# Recursive Structural Similarity
def recursive_similarity(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    fa, fb = flatten(a), flatten(b)
    all_keys = set(fa.keys()) | set(fb.keys())
    matched = sum(1 for k in all_keys if fa.get(k) == fb.get(k))
    return round(matched / len(all_keys), 4) if all_keys else 0.0

# Flatten JSON to string for embedding
def flatten_json_to_string(d):
    if not isinstance(d, dict):
        return ""
    return " ".join([f"{k}={v}" for k, v in sorted(flatten(d).items())])

# Setup embedding model
embedding_model = Embeddings(
    model_id="ibm/granite-embedding-107m-multilingual",
    credentials={
        "url": conf_dict["url"],
        "apikey": conf_dict["apikey"],
    },
    project_id=conf_dict["project_id"],
)

# Cosine similarity using Watsonx embeddings
def compute_embedding_similarity(text1, text2):
    try:
        vec1 = np.array(embedding_model.embed_query(text=text1))
        vec2 = np.array(embedding_model.embed_query(text=text2))
        score = cosine_similarity([vec1], [vec2])[0][0]
        return round(float(score), 4)
    except Exception as e:
        print(f"Embedding error: {e}")
        return 0.0



# Prepare expected output and user input
df["expected_output"] = df["expected_output"].apply(parse_json_safe)
df["user_input"] = df["user_input"].astype(str)

# Identify model columns
model_columns = [col for col in df.columns if re.match(r".+?\d+$", col)]

# Evaluate each model
results = {"user_input": df["user_input"]}

for model_col in model_columns:
    rss_scores, ecs_scores = [], []

    for idx, row in tqdm(df.iterrows(), desc=model_col):
        pred = eval(row[model_col])
        expected = row["expected_output"]

        rss = recursive_similarity(pred, expected)
        rss_scores.append(rss)

        pred_text = flatten_json_to_string(pred)
        expected_text = flatten_json_to_string(expected)
        ecs = compute_embedding_similarity(pred_text, expected_text)
        ecs_scores.append(ecs)

    results[f"{model_col}-RSS"] = rss_scores
    results[f"{model_col}-ECS"] = ecs_scores

# Save to CSV
evaluation_df = pd.DataFrame(results)
evaluation_df.to_csv("evaluation_scores.csv", index=False)
