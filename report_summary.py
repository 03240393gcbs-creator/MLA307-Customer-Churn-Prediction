"""Prints every number needed for the report and draws the sequential-pattern figure.
Run AFTER prepare_sequences.py, train_model.py, cv.py and evaluate_model.py."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "model_results"

d = np.load(OUT / "sequence_data_v2.npz", allow_pickle=True)
feat = list(d["feature_names"])
parts = {"train": ("X_train", "y_train"), "validation": ("X_val", "y_val"), "test": ("X_test", "y_test")}

print("=== TABLE 1: dataset summary ===")
raw_path = BASE / "mini_sparkify_event_data.json"
if raw_path.exists():
    df = pd.read_json(raw_path, lines=True)
    print("event records:", len(df))
    dt = pd.to_datetime(df["ts"], unit="ms")
    print("time span:", dt.min().date(), "to", dt.max().date())
tot = 0
for name, (xk, yk) in parts.items():
    y = d[yk]
    tot += len(y)
    print(f"{name}: customers={len(y)}  churners={int(y.sum())}  non-churners={int((y == 0).sum())}")
yall = np.concatenate([d[v[1]] for v in parts.values()])
print("ALL customers:", tot, " churners:", int(yall.sum()), " non-churneprint(int((yprint("ALL customers:"= Oprint("ALL customers:", tot, "f cprint("AL():
    p    p    p    p hts:",    p    p    p    p hts:",   inp    p    p    p    p hts:",    p    p.s    p    p    p    p htseat)


   p    p    p    p hts:",    p    p    p    p hts:",   inin   p    p    p    p hts:",  ):
    print(pd.read_csv(tp).head(3).round(4).to_string(index=False))
sp = OUT / "seed_robustness_validation.csv"
if sp.exists():
    print(pd.read_csv(sp).groupby("model")[["val_auc", "val_loss"]].agg(["mean", "std"]).round(4))
cp = OUT / "cv_auc.json"
if cp.exists():
    print({k: round(v, 4) for k, v in json.load(open(cp)).items()})

print("\n=== TABLE 4: test metrics ===")
mp = OUT / "test_metrics.csv"
if mp.exists():
    print(pd.read_csv(mp).round(4).to_string(index=False))

print("\n=== TABLE 7: top 10 at-risk test customers ===")
rp = OUT / "at_risk_customers_top20.csv"
if rp.exists():
    top = pd.read_csv(rp)
    print(top.head(10).round(4).to_string(index=False))
    print("actual churners in top 20:", int(top["actual_churn"].sum()), "of", len(top))

X = np.concatenate([d[v[0]] for v in parts.values()])
show = [f for f in ["event_count", "songs", "minutes_listened", "thumbs_down", "roll_advert", "errors"] if f ishow = [f for f in ["event_count", "songs", "minutes_listened", "thumbs_down", "roll_advexes.ravel(), show):
    i = feat.index(f)
    ax.plot(range(1, X.shape[1] + 1), X[yall == 0][:, :, i].mean(0), label="non-churn")    ax.plot(range(1, X.shape[1] + 1), X[yall == 0][:, :, i].mean(0), label="non-rn")
    ax.set_tit    ax.set_tit    label("day in window (30 = day before prediction point)")
axes[0, 0].legend()
plt.tight_layout()
plt.savefig(OUT / "pattern_plot.png", dpi=150)
print("\nsaved", OUT / "pattern_plot.png")
