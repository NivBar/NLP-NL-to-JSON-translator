import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load evaluation results
df = pd.read_csv("evaluation_scores.csv")

# Rename model nicknames to readable names
model_name_map = {
    "granite328": "granite-3-2-8b",
    "llama318": "llama-3-1-8b",
    "mixtral8x7": "mixtral-8x7b",
    "llama3370": "llama-3-3-70b"
}

# Extract model-metric columns
model_metrics = [col for col in df.columns if '-' in col and col != 'user_input']
models = sorted(set(col.split('-')[0] for col in model_metrics))
metrics = sorted(set(col.split('-')[1] for col in model_metrics))

# Build long-format DataFrame
plot_data = []
for model in models:
    for metric in metrics:
        col = f"{model}-{metric}"
        if col in df.columns:
            scores = df[col].dropna().astype(float)
            avg = scores.mean()
            readable_name = model_name_map.get(model, model)
            plot_data.append({'Model': readable_name, 'Metric': metric, 'Average Score': avg})

summary_df = pd.DataFrame(plot_data)

# Barplot with annotations
plt.figure(figsize=(10, 6))
palette = {'ECS': 'steelblue', 'RSS': 'darkorange'}
barplot = sns.barplot(data=summary_df, x='Model', y='Average Score', hue='Metric', palette=palette)
for container in barplot.containers:
    barplot.bar_label(container, fmt="%.2f", padding=3)
plt.title("Average Similarity Scores by Model and Metric")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("model_metric_barplot_annotated.png")
plt.close()

# Heatmap: normalize each column individually (per metric)
heatmap_df = summary_df.pivot(index='Model', columns='Metric', values='Average Score')
normed_df = heatmap_df.copy()
for metric in metrics:
    col = normed_df[metric]
    normed_df[metric] = (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0.5

# Plot heatmap with color per metric (split into two heatmaps)
for metric in metrics:
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(
        normed_df[[metric]],
        annot=heatmap_df[[metric]].round(2),
        fmt=".2f",
        cmap='Blues' if metric == 'ECS' else 'Oranges',
        cbar=False,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f"Model Scores Heatmap: {metric}")
    plt.tight_layout()
    plt.savefig(f"model_metric_heatmap_{metric}.png")
    plt.close()
