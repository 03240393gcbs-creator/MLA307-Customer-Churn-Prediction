import os, json, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pathlib import Path

OUT = str(Path(__file__).resolve().parent / "model_results")
d = np.load(f"{OUT}/sequence_data_v2.npz", allow_pickle=True)
X = np.concatenate([d["X_train"], d["X_val"]]); y = np.concatenate([d["y_train"], d["y_val"]])
cw = {int(k): v for k, v in json.load(open(f"{OUT}/class_weights_v2.json")).items()}
T, F = X.shape[1:]
def seed(s): random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def build(kind, units, dropout, lr):
    cell = {"GRU": layers.GRU, "LSTM": layers.LSTM}[kind]
    m = keras.Sequential([layers.Input(shape=(T, F)), cell(units), layers.Dropout(dropout),
                          layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return m

configs = {"GRU": ("GRU", 32, 0.2, 5e-3), "LSTM": ("LSTM", 32, 0.4, 5e-3), "GRU_small": ("GRU", 16, 0.4, 5e-3)}
oof = {k: np.zeros(len(y)) for k in list(configs) + ["LogReg"]}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for f, (tr, te) in enumerate(skf.split(X, y)):
    lr = LogisticRegression(class_weight=cw, max_iter=2000, C=0.1).fit(X[tr].reshape(len(tr), -1), y[tr])
    oof["LogReg"][te] = lr.predict_proba(X[te].reshape(len(te), -1))[:, 1]
    itr, iva = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=f)
    for name, (kind, u, dr, l) in configs.items():
        seed(42 + f); m = build(kind, u, dr, l)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        m.fit(X[itr], y[itr], validation_data=(X[iva], y[iva]), epochs=50, batch_size=16,
              class_weight=cw, callbacks=[es], verbose=0)
        oof[name][te] = m.predict(X[te], verbose=0).ravel()
        keras.backend.clear_session()
    print("fold", f, {k: round(roc_auc_score(y[te], v[te]), 3) for k, v in oof.items()}, flush=True)

res = {k: float(roc_auc_score(y, v)) for k, v in oof.items()}
print("OOF AUC", res)
pd.DataFrame({"y": y, **oof}).to_csv(f"{OUT}/cv_oof_predictions.csv", index=False)
json.dump(res, open(f"{OUT}/cv_auc.json", "w"), indent=2)
