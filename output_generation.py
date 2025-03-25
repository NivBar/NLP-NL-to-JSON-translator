import pandas as pd
import json
import os
from tqdm import tqdm
from ibm_watsonx_ai.foundation_models import ModelInference
from jsonschema import Draft7Validator
from config import conf_dict

# -----------------------------
# Load Inputs
# -----------------------------

with open("Function_file.json") as f:
    schema_json = json.load(f)['function']['parameters']
validator = Draft7Validator(schema_json)

with open("context.txt") as f:
    fixed_context = f.read().replace(',\n}', '\n}')
    context = json.loads(fixed_context)

metrics_encoding = {}
with open("Metrics_encoding.txt") as f:
    for line in f:
        parts = line.strip().split(':')
        if len(parts) == 2:
            key, value = parts[0].strip().lower(), parts[1].strip().rstrip(',')
            try:
                metrics_encoding[key] = int(value)
            except ValueError:
                continue

examples_df = pd.read_csv("healthcare_examples.csv")

# -----------------------------
# Watsonx Setup
# -----------------------------

def get_credentials():
    return {
        "url": conf_dict["url"],
        "apikey": conf_dict["apikey"],
    }

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 500,
    "repetition_penalty": 1
}

project_id = conf_dict["project_id"]

model_names = {
    "granite328": "ibm/granite-3-2-8b-instruct",
    "llama318": "meta-llama/llama-3-1-8b-instruct",
    "mixtral8x7": "mistralai/mixtral-8x7b-instruct-v01",
    "llama3370": "meta-llama/llama-3-3-70b-instruct"
}

# -----------------------------
# Prompt Builder
# -----------------------------

def make_prompt(user_input, schema_json, context, metrics_encoding):
    return f"""
You are a JSON-generating assistant that converts healthcare-related natural language questions into structured JSON requests conforming to a predefined API schema.

* Your Goal:
Translate the user input into a valid JSON request.
- Always include required fields.
- Only include optional fields if they are mentioned in the input.
- Do not guess. If the user didn’t say something, don’t include it.

* Function Schema:
{json.dumps(schema_json, indent=2)}

* Default Context (used only if not overridden by user input):
- scopeIds: {context['scopeIds']}
- year: {context['year']}

* Metric Name to ID Mapping:
{json.dumps(metrics_encoding, indent=2)}

Convert this user input into a single valid JSON object:
\"{user_input}\"

JSON:
""".strip()

# -----------------------------
# JSON Cleaning
# -----------------------------

def extract_and_clean_json(response_text):
    start = response_text.find('{')
    end = response_text.find('}', start)
    while end != -1:
        try:
            snippet = response_text[start:end+1]
            parsed = json.loads(snippet)
            break
        except json.JSONDecodeError:
            end = response_text.find('}', end + 1)
    else:
        return "ERROR: No valid JSON found"

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()
                    if v not in [None, [], "", {}]}
        elif isinstance(obj, list):
            return [clean(v) for v in obj if v not in [None, [], "", {}]]
        return obj

    cleaned_json = clean(parsed)

    allowed_keys = set(schema_json["properties"].keys())
    for key in cleaned_json:
        if key not in allowed_keys:
            print(f"ERROR: Unexpected key '{key}' not defined in schema.")

    return cleaned_json


# Load existing result file if it exists
output_file = "healthcare_examples_with_all_models.csv"
if os.path.exists(output_file):
    print(f"Loading existing result file: {output_file}")
    examples_df = pd.read_csv(output_file)
else:
    examples_df = pd.read_csv("healthcare_examples.csv")

# Ensure all model columns exist
for model_col in model_names:
    if model_col not in examples_df.columns:
        examples_df[model_col] = None

# -----------------------------
# Run All Models + Validation
# -----------------------------

for short_name, full_model_name in model_names.items():
    print(f"\n[Model: {short_name}] Starting/resuming...")
    model = ModelInference(
        model_id=full_model_name,
        params=parameters,
        credentials=get_credentials(),
        project_id=project_id
    )

    for idx in tqdm(range(len(examples_df)), desc=f"{short_name}"):
        existing = examples_df.at[idx, short_name]
        if isinstance(existing, str) and not existing.startswith("ERROR:"):
            continue
        elif isinstance(existing, dict):
            continue  # already valid JSON

        # Retry until we get valid JSON
        user_input = examples_df.at[idx, "user_input"]
        while True:
            try:
                prompt = make_prompt(user_input, schema_json, context, metrics_encoding)
                raw_response = model.generate_text(prompt=prompt)
                cleaned = extract_and_clean_json(raw_response)

                if isinstance(cleaned, dict):
                    examples_df.at[idx, short_name] = json.dumps(cleaned)
                    break
                else:
                    print(f"Retrying ({short_name}, row {idx}) due to: {cleaned}")
            except Exception as e:
                print(f"Retrying ({short_name}, row {idx}) due to error: {e}")


# Save updated file
examples_df.to_csv(output_file, index=False)
print(f"Saved updated file: {output_file}")
