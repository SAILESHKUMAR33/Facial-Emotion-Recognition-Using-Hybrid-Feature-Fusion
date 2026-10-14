"""STEP 9 - Final comparison table, ablation table, significance test, confusion analysis."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import get_classes, split_files
from config import *

res = json.load(open(RESULTS_DIR / "results.json"))
ORDER = [
    ("custom_cnn", "Custom CNN"),
    ("transfer_learning", "Transfer Learning"),
    ("landmark_mlp", "Landmarks only"),
    ("fusion_cnn_tl", "CNN + TL"),
    ("fusion_cnn_lm", "CNN + Landmarks"),
    ("fusion_tl_lm", "TL + Landmarks"),
    ("fusion_all_frozen", "CNN + TL + LM (frozen)"),
    ("fusion_all_finetuned", "CNN + TL + LM (fine-tuned)"),
]
ORDER = [(k, n) for k, n in ORDER if k in res]
df = pd.DataFrame([{
    "Model": n, "key": k, "Val acc": res[k]["val_accuracy"], "Test acc": res[k]["test_accuracy"],
    "Precision": res[k]["precision_macro"], "Recall": res[k]["recall_macro"], "F1": res[k]["f1_macro"],
    "Train (s)": res[k]["train_time_s"], "Infer (ms/img)": res[k]["inference_ms_per_image"],
    "Params": res[k]["params"]} for k, n in ORDER])
print("\n=========== FINAL COMPARISON / ABLATION (test set) ===========")
print(df.drop(columns="key").round(4).to_string(index=False))
df.drop(columns="key").to_csv(RESULTS_DIR / "final_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(df)); w = 0.38
ax.bar(x - w / 2, df["Test acc"], w, label="Accuracy")
ax.bar(x + w / 2, df["F1"], w, label="Macro F1")
ax.set_xticks(x); ax.set_xticklabels(df["Model"], rotation=25, ha="right"); ax.set_ylim(0, 1)
ax.set_title("Individual models vs fusion (ablation)"); ax.legend()
fig.tight_layout(); fig.savefig(RESULTS_DIR / "final_comparison.png", dpi=130)

# ---- is the best fusion really better than the best single model? ---------
singles = df[df.key.isin(["custom_cnn", "transfer_learning", "landmark_mlp"])]
fusions = df[df.key.str.startswith("fusion")]
if len(singles) and len(fusions):
    bs = singles.loc[singles["Val acc"].idxmax()]      # chosen on VALIDATION accuracy
    bf = fusions.loc[fusions["Val acc"].idxmax()]
    _, y_true = split_files("test")
    ps = np.load(RESULTS_DIR / f"preds_{bs.key}.npy")
    pf = np.load(RESULTS_DIR / f"preds_{bf.key}.npy")
    b = int(((ps == y_true) & (pf != y_true)).sum())   # single right, fusion wrong
    c = int(((ps != y_true) & (pf == y_true)).sum())   # single wrong, fusion right
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    print(f"\nBest single (by val): {bs.Model}  | best fusion (by val): {bf.Model}")
    print(f"Test acc {bs['Test acc']:.4f} vs {bf['Test acc']:.4f}  "
          f"(difference {bf['Test acc'] - bs['Test acc']:+.4f})")
    print(f"McNemar exact test: single-only-correct={b}, fusion-only-correct={c}, p={p:.4f}")
    verdict = ("statistically significant (p<0.05)" if p < 0.05 else
               "NOT statistically significant - could be chance on this test set")
    print(f"Verdict: the difference is {verdict}.")

    cm = np.zeros((len(get_classes()),) * 2, dtype=int)
    for t, q in zip(y_true, pf):
        cm[t, q] += 1
    classes = get_classes()
    pairs = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
             for j in range(len(classes)) if i != j and cm[i, j] > 0]
    print(f"\nMost frequent confusions of {bf.Model} (true -> predicted):")
    for n, t, q in sorted(pairs, reverse=True)[:6]:
        print(f"  {t:>10} -> {q:<10} {n}x")

print(f"\nSaved: final_comparison.csv/png and confusion_*.png in {RESULTS_DIR}")
