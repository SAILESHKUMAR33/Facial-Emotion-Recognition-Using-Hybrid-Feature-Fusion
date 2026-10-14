"""STEP 6 - Compare the three individual approaches (no training)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import *

res = json.load(open(RESULTS_DIR / "results.json"))
names = {"custom_cnn": "Custom CNN", "transfer_learning": "Transfer Learning",
         "landmark_mlp": "Landmarks MLP"}
missing = [k for k in names if k not in res]
if missing:
    raise SystemExit(f"Run the training steps first, missing: {missing}")

df = pd.DataFrame([{
    "Model": names[k], "Accuracy": res[k]["test_accuracy"], "Precision": res[k]["precision_macro"],
    "Recall": res[k]["recall_macro"], "F1": res[k]["f1_macro"],
    "Train time (s)": res[k]["train_time_s"], "Inference (ms/img)": res[k]["inference_ms_per_image"],
    "Params": res[k]["params"]} for k in names])
print("\n=== Individual models (test set) ===")
print(df.round(4).to_string(index=False))
df.to_csv(RESULTS_DIR / "individual_comparison.csv", index=False)

pc = pd.DataFrame({names[k]: res[k]["per_class_f1"] for k in names})
print("\n=== Per-class F1 ===")
print(pc.round(3).to_string())
print("\n=== Strengths / weaknesses ===")
for m in pc.columns:
    s = pc[m].sort_values()
    print(f"{m:<20} best: {s.index[-1]} ({s.iloc[-1]:.2f}) | worst: {s.index[0]} ({s.iloc[0]:.2f})")
print(f"\nBest accuracy : {df.loc[df['Accuracy'].idxmax(), 'Model']}")
print(f"Fastest       : {df.loc[df['Inference (ms/img)'].idxmin(), 'Model']}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(ax=ax[0], rot=15, ylim=(0, 1))
ax[0].set_title("Test metrics")
pc.plot.bar(ax=ax[1], rot=30, ylim=(0, 1)); ax[1].set_title("Per-class F1")
fig.tight_layout(); fig.savefig(RESULTS_DIR / "individual_comparison.png", dpi=130)
print(f"\nSaved: {RESULTS_DIR / 'individual_comparison.png'}")
