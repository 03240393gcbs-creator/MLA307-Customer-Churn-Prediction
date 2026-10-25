import os, json, random, itertools
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
set_seed()

IN = "/Users/jigme/Downloads/member1_sparkify_outputs/member1_outputs"
OUT = "/Users/jigme/Downloads/member1_sparkify_outputs/member2_outputs"
os.makedirs(OUT, exist_ok=True)

d = np.load(f"{OUT}/sequence_data_v2.npz", allow_pickle=True)
Xtr, ytr, Xva, yva, Xte, yte = [d[k] for k in ["X_train","y_train","X_val","y_val","X_test","y_test"]]
test_ids = d["test_user_ids"]; val_ids = d["val_user_ids"]; train_ids = d["train_user_ids"]
cw = {int(k): v for k, v in json.load(open(f"{OUT}/class_weights_v2.json")).items()}
T, F = Xtr.shape[1], Xtr.shape[2]
print("shapes", Xtr.shape, Xva.shape, Xte.shape, "class weights", cw)

def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p))

lr = LogisticRegression(class_weight=cw, max_iter=2000, C=0.1, random_state=SEED)
lr.fit(Xtr.reshape(len(Xtr), -1), ytr)
p_lr_va = lr.predict_proba(Xva.reshape(len(Xva), -1))[:, 1]
p_lr_te = lr.predict_proba(Xte.reshape(len(Xte), -1))[:, 1]

def build(kind, units=32, dropout=0.3, lr_=1e-3, layers_n=1):
    m = keras.Sequential([layers.Input(shape=(T, F))])
    for i in range(layers_n):
        last = i == layers_n - 1
        cell = {"GRU": layers.GRU, "LSTM": layers.LSTM, "RNN": layers.SimpleRNN}[kind]
        m.add(cell(units, return_sequences=not last))
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(16, activation="relu"))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(lr_), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

def fit(kind, seed=SEED, epochs=60, **kw):
    set_seed(seed)
    m = build(kind, **kw)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=16,
              class_weight=cw, callbacks=[es], verbose=0)
    return m, h

rnn, h_rnn = fit("RNN", units=32, dropout=0.3, lr_=1e-3)

grid = list(itertools.product(["GRU", "LSTM"], [16, 32], [0.2, 0.4], [1e-3, 5e-3]))
rows = []
for kind, u, dr, l in grid:
    m, h = fit(kind, units=u, dropout=dr, lr_=l)
    pv = m.predict(Xva, verbose=0).ravel()
    rows.append(dict(model=kind, units=u, dropout=dr, learning_rate=l,
                     params=m.count_params(), epochs=len(h.history["loss"]),
                     best_val_loss=min(h.history["val_loss"]), val_auc=roc_auc_score(yva, pv)))
    print(rows[-1], flush=True)
    keras.backend.clear_session()
tune = pd.DataFrame(rows).sort_values(["best_val_loss"]).reset_index(drop=True)
tune.to_csv(f"{OUT}/tuning_results.csv", index=False)

best = {k: tune[tune.model == k].iloc[0] for k in ["GRU", "LSTM"]}
print(best)

seed_rows, models = [], {}
for kind in ["GRU", "LSTM"]:
    b = best[kind]
    for s in range(3):
        m, h = fit(kind, seed=SEED + s, units=int(b.units), dropout=float(b.dropout), lr_=float(b.learning_rate))
        pv = m.predict(Xva, verbose=0).ravel()
        seed_rows.append(dict(model=kind, seed=SEED + s, val_auc=roc_auc_score(yva, pv),
                              val_loss=min(h.history["val_loss"])))
        models[(kind, SEED + s)] = (m, h)
seeds = pd.DataFrame(seed_rows)
seeds.to_csv(f"{OUT}/seed_robustness_validation.csv", index=False)
print(seeds.groupby("model")[["val_auc", "val_loss"]].agg(["mean", "std"]))

final_kind = seeds.groupby("model")["val_loss"].mean().idxmin()
final_model, final_hist = models[(final_kind, SEED)]
fb = best[final_kind]
print("FINAL:", final_kind, dict(fb))

p_te = final_model.predict(Xte, verbose=0).ravel()
p_rnn_te = rnn.predict(Xte, verbose=0).ravel()
res = {"Logistic regression (baseline) @ 0.5": metrics(yte, p_lr_te, 0.5),
       "Simple RNN @ 0.5": metrics(yte, p_rnn_te, 0.5),
       f"{final_kind} (final) @ 0.5": metrics(yte, p_te, 0.5)}
print(pd.DataFrame(res).T.round(3))

final_model.save(f"{OUT}/final_model.keras")

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(final_hist.history["loss"], label="train"); ax[0].plot(final_hist.history["val_loss"], label="validation")
ax[0].set_title(f"{final_kind}: loss"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("binary cross-entropy"); ax[0].legend()
ax[1].plot(final_hist.history["auc"], label="train"); ax[1].plot(final_hist.history["val_auc"], label="validation")
ax[1].set_title(f"{final_kind}: AUC"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/training_curves.png"); plt.close()

fig, ax = plt.subplots(figsize=(9, 1.9)); ax.axis("off")
steps = [f"Input\n({T} days x {F} features)", f"{final_kind}\n({int(fb.units)} units)", f"Dropout\n({fb.dropout})",
         "Dense 16\n(ReLU)", "Dense 1\n(sigmoid)", "Churn\nprobability"]
for i, s in enumerate(steps):
    ax.text(i * 1.6 + 0.5, 0.5, s, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="#dbe9f6", ec="#2b6cb0"))
    if i < len(steps) - 1: ax.annotate("", xy=(i * 1.6 + 1.5, 0.5), xytext=(i * 1.6 + 1.0, 0.5), arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-0.2, 9.9); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/model_diagram.png"); plt.close()
print("DONE - saved model and figures to", OUT)
