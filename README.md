
# Healthcare LLM JSON Translator

This project provides a complete pipeline for translating natural language healthcare queries into structured JSON requests using large language models (LLMs). It also includes a robust evaluation and visualization framework to assess the accuracy of the generated outputs.

## Overview

Healthcare data APIs often require complex, structured inputs, making them less accessible to non-technical stakeholders. This project addresses that challenge by leveraging foundation models to automatically convert user-friendly queries into compliant JSON objects, suitable for programmatic access to healthcare statistics.

The system includes the following components:

- **Prompt Engineering Interface**: A Streamlit-based app that generates schema-compliant JSON from natural language input.
- **Model Inference Pipeline**: Batch processing pipeline that runs multiple LLMs (e.g., IBM Granite, LLaMA, Mixtral) to generate JSON outputs for benchmarking.
- **Evaluation Module**: A dual-metric evaluation strategy combining Recursive Structural Similarity (RSS) and Embedding-based Cosine Similarity (ECS).
- **Visualization Module**: Clear plots and heatmaps that compare model performance across metrics.

## File Structure

- `app.py`: Streamlit app for real-time prompt testing.
- `output_generation.py`: Runs all models against a query dataset to generate outputs.
- `evaluation.py`: Scores the outputs based on structural and semantic similarity to ground truth.
- `visualization.py`: Plots performance comparisons across models and metrics.
- `Function_file.json`: JSON schema for the healthcare API.
- `context.txt`: Default values for optional fields (e.g., year, scopeId).
- `Metrics_encoding.txt`: Maps healthcare metric names to their corresponding IDs.
- `healthcare_examples.csv`: Ground truth dataset of user inputs and their expected outputs.
- `healthcare_examples_with_all_models.csv`: Output of all LLMs for each example.
- `evaluation_scores.csv`: Final evaluation scores per model and metric.

## Requirements

- Python >= 3.8
- IBM Watsonx AI SDK (`ibm-watsonx-ai`)
- pandas, numpy, scikit-learn, matplotlib, seaborn, tqdm, jsonschema, streamlit


## Notes

- Models are accessed via IBM Watsonx, using credentials stored in a `config.py` file. Ensure this is set up properly before running.
- Output JSON is validated against the API schema.
- Errors during model inference or formatting are handled and logged for transparency.

## UI interface screenshot
![Image](https://github.com/user-attachments/assets/6a60e468-bb02-4daf-9a67-4b236c6e506d)

