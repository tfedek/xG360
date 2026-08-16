from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from football_xg.config import OUTPUT_DIR

RESULTS_PATH = OUTPUT_DIR / "model_training" / "cv_results_all.csv"
OUT_DIR = OUTPUT_DIR / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_PATH)
overall = df[df["fold"].astype(str) == "overall"].copy()
overall.to_csv(OUT_DIR / "overall_results.csv", index=False)

print("\\n=== OVERALL RESULTS ===")
print(overall[[
    "feature_set", "model", "validation", "threshold",
    "roc_auc", "pr_auc", "precision", "recall", "f1", "brier"
]].to_string(index=False))

metrics = ["roc_auc", "pr_auc", "f1", "brier"]
for metric in metrics:
    plt.figure(figsize=(12, 5))
    labels = overall["feature_set"] + "\\n" + overall["model"] + "\\n" + overall["validation"]
    plt.bar(range(len(overall)), overall[metric])
    plt.xticks(range(len(overall)), labels, rotation=60, ha="right")
    plt.title(metric.upper())
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"comparison_{metric}.png", dpi=160)
    plt.close()

print(f"\\nSaved evaluation outputs: {OUT_DIR}")
