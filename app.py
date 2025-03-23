import streamlit as st
import json
from ibm_watsonx_ai.foundation_models import ModelInference
from PIL import Image
from config import conf_dict

# Set page config
st.set_page_config(page_title="Watsonx Healthcare JSON Translator", layout="centered")

# Constants
MODEL_ID = "ibm/granite-3-2-8b-instruct"
PROJECT_ID = conf_dict["project_id"]
APIKEY = conf_dict["apikey"]

IBM_BLUE = "#052F5F"
IBM_LIGHT_BLUE = "#A6C8FF"

# Load logo
logo = Image.open("logo.jpeg")

# Custom style
st.markdown(f"""
    <style>
        .stApp {{
            background-color: #f4f4f4;
            color: {IBM_BLUE};
        }}
        .json-output {{
            background-color: {IBM_LIGHT_BLUE};
            padding: 1em;
            border-radius: 10px;
            font-family: monospace;
            white-space: pre-wrap;
            color: black;
        }}
        .instruction {{
            font-size: 16px;
            color: #333333;
        }}
        .main-title {{
            font-size: 28px;
            font-weight: bold;
        }}
        .stButton > button {{
            background-color: {IBM_LIGHT_BLUE};
            color: black;
            border: 1px solid {IBM_BLUE};
            border-radius: 5px;
            padding: 0.5em 1em;
        }}
        .stTextArea textarea {{
            border: 2px solid {IBM_BLUE};
        }}
    </style>
""", unsafe_allow_html=True)

# Title and logo
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown('<div class="main-title">Watsonx-Powered Healthcare Query Translator</div>', unsafe_allow_html=True)
with col2:
    st.image(logo, width=80)

# Instructions
st.markdown(
    """
    <div class="instruction">
    This tool leverages IBM's Granite 3-2-8B Large Language Model (LLM) via Watsonx to convert natural language queries
    into structured JSON requests for a healthcare statistics API. These JSON outputs can be used to programmatically retrieve
    healthcare-related metrics, such as patient satisfaction scores, treatment success rates, and medication administration trends.
    </div>
    """,
    unsafe_allow_html=True
)

# Input
st.subheader("Enter a healthcare-related query")
user_input = st.text_area("Type your query below", height=150, label_visibility="collapsed")
submit = st.button("Generate JSON")

# Prompt builder
def make_prompt(user_input, schema, context, metrics_encoding):
    return f"""
You are a JSON-generating assistant that converts healthcare-related natural language questions into structured JSON requests conforming to a predefined API schema.

Translate the user input into a valid JSON request.
- Always include required fields.
- Only include optional fields if they are mentioned in the input.
- Do not guess. If the user didn’t say something, don’t include it.

Function Schema:
{json.dumps(schema, indent=2)}

Default Context (used only if not overridden by user input):
- scopeIds: {context['scopeIds']}
- year: {context['year']}

Metric Name to ID Mapping:
{json.dumps(metrics_encoding, indent=2)}

User input:
\"{user_input}\"

JSON:
""".strip()

# Process query
if submit and user_input.strip():
    with open("Function_file.json") as f:
        schema = json.load(f)['function']['parameters']
    with open("context.txt") as f:
        context = json.loads(f.read().replace(',\n}', '\n}'))

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

    model = ModelInference(
        model_id=MODEL_ID,
        params={"decoding_method": "greedy", "max_new_tokens": 300},
        credentials={"url": conf_dict["url"], "apikey": APIKEY},
        project_id=PROJECT_ID
    )

    prompt = make_prompt(user_input, schema, context, metrics_encoding)
    try:
        raw_response = model.generate_text(prompt=prompt)
        start = raw_response.find('{')
        end = raw_response.find('}', start)
        while end != -1:
            try:
                snippet = raw_response[start:end+1]
                parsed = json.loads(snippet)
                break
            except json.JSONDecodeError:
                end = raw_response.find('}', end + 1)
        else:
            st.error("Failed to extract valid JSON from model output.")
            parsed = None

        if parsed:
            st.subheader("Generated JSON Output")
            st.markdown(f"<div class='json-output'>{json.dumps(parsed, indent=2)}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating JSON: {e}")
