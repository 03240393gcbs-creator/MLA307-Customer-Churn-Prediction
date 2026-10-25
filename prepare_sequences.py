import json, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Project folder
BASE = Path(__file__).resolve().parent
OUT = BASE / "model_results"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW, GAP = 30, 1
SEED = 42

FEATURES = ["event_count", "sessions", "songs", "minutes_listened", "thumbs_up",
            "thumbs_down", "add_to_playlist", "add_friend", "roll_advert",
            "downgrade", "upgrade", "home", "errors"]

# ---------------------------------------------------------------
# 1. Load raw events

# ---------------------------------------------------------------
df = pd.read_json(BASE / "mini_sparkify_event_data.json", lines=True)
df = df[df["userId"].astype(str).str.strip() != ""].copy()
df["userId"] = df["userId"].astype(str)
df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
df["date"] = df["datetime"].dt.normalize()

# ---------------------------------------------------------------
# 2. Per-customer prediction point and label
#    churner  -> first Cancellation Confirmation (label 1)
#    others   -> last recorded event             (label 0)
# ---------------------------------------------------------------
churn_ts = df[df.page == "Cancellation Confirmation"].groupby("userId")["ts"].min()
last_ts = df.groupby("userId")["ts"].max()

# ---------------------------------------------------------------
# 3. Stratified customer-level split built from ALL customers (60/20/20)
# ---------------------------------------------------------------
all_ids = sorted(df["userId"].unique())
y_all = np.array([int(u in churn_ts.index) for u in all_ids])

tr_u, tmp_u, _, y_tmp = train_test_split(
    all_ids, y_all, test_size=0.4, stratify=y_all, random_state=SEED)
va_u, te_u = train_test_split(
    tmp_u, test_size=0.5, stratify=y_tmp, random_state=SEED)
split_ids = {"train": list(tr_u), "validation": list(va_u), "test": list(te_u)}

print("customers:", len(all_ids), " churners:", int(y_all.sum()),
      " non-churners:", int((y_all == 0).sum()))

# ---------------------------------------------------------------
# 4. Remove cancellation events and anything at/after cancellation
# ---------------------------------------------------------------
ev = df[df.page != "Cancellation Confirmation"].copy()
m = ev.userId.isin(churn_ts.index)
ev = ev[(~m) | (ev.ts < ev.userId.map(churn_ts))].copy()

ev["is_song"] = (ev.page == "NextSong").astype(int)
ev["minutes_row"] = np.where(
    ev.page == "NextSong", pd.to_numeric(ev.length, errors="coerce").fillna(0) / 60, 0.0)

p2f = {"Thumbs Up": "thumbs_up", "Thumbs Down": "thumbs_down", "Add to Playlist": "add_to_playlist",
       "Add Friend": "add_friend", "Roll Advert": "roll_advert", "Downgrade": "downgrade",
       "Upgrade": "upgrade", "Home": "home", "Error": "errors"}
for p, f in p2f.items():
    ev[f] = (ev.page == p).astype(int)

# ---------------------------------------------------------------
# 5. Daily aggregates per customer
# ---------------------------------------------------------------
daily = ev.groupby(["userId", "date"]).agg(
    event_count=("page", "size"), sessions=("sessionId", "nunique"), songs=("is_song", "sum"),
    minutes_listened=("minutes_row", "sum"), thumbs_up=("thumbs_up", "sum"),
    thumbs_down=("thumbs_down", "sum"), add_to_playlist=("add_to_playlist", "sum"),
    add_friend=("add_friend", "sum"), roll_advert=("roll_advert", "sum"),
    downgrade=("downgrade", "sum"), upgrade=("upgrade", "sum"),
    home=("home", "sum"), errors=("errors", "sum")
).reset_index()

lookup = {(r.userId, r.date): np.array([getattr(r, c) for c in FEATURES], dtype=np.float32)
          for r in daily.itertuples(index=False)}

# ---------------------------------------------------------------
# 6. Build one 30-day sequence per customer, ending GAP day(s) before the prediction point
# ---------------------------------------------------------------
def build(uid):
    y = int(uid in churn_ts.index)
    cutoff_ts = int(churn_ts[uid]) if y else int(last_ts[uid])
    cutoff_date = pd.to_datetime(cutoff_ts, unit="ms").normalize()
    end = cutoff_date - pd.Timedelta(days=GAP)
    dates = pd.date_range(end=end, periods=WINDOW, freq="D")
    seq = np.zeros((WINDOW, len(FEATURES)), dtype=np.float32)
    for i, dt in enumerate(dates):
        if (uid, dt) in lookup:
            seq[i] = lookup[(uid, dt)]
    return seq, y

raw, labels = {}, {}
for sp, ids in split_ids.items():
    for u in ids:
        raw[u], labels[u] = build(str(u))

def stack(sp):
    ids = [str(u) for u in split_ids[sp]]
    return ids, np.stack([raw[u] for u in split_ids[sp]]), np.array([labels[u] for u in split_ids[sp]])

tr_ids, Xtr_raw, ytr = stack("train")
va_ids, Xva_raw, yva = stack("validation")
te_ids, Xte_raw, yte = stack("test")

# ---------------------------------------------------------------
# 7. Scale: fit on ACTIVE-day rows of training customers only; zero rows stay zero
# ---------------------------------------------------------------
a = Xtr_raw.reshape(-1, len(FEATURES))
a = a[a[:, 0] > 0]
scaler = StandardScaler().fit(a)

def scale(X):
    out = np.zeros_like(X)
    for i in range(len(X)):
        act = X[i, :, 0] > 0
        if act.any():
            out[i, act] = scaler.transform(X[i, act])
    return np.nan_to_num(out).astype(np.float32)

Xtr, Xva, Xte = scale(Xtr_raw), scale(Xva_raw), scale(Xte_raw)

# ---------------------------------------------------------------
# 8. Class weights from training labels
# ---------------------------------------------------------------
cnt = np.bincount(ytr, minlength=2)
cw = {0: len(ytr) / (2 * cnt[0]), 1: len(ytr) / (2 * cnt[1])}

# ---------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------
np.savez(OUT / "sequence_data_v2.npz",
         X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte, y_test=yte,
         train_user_ids=np.array(tr_ids), val_user_ids=np.array(va_ids), test_user_ids=np.array(te_ids),
         feature_names=np.array(FEATURES))
json.dump({str(k): float(v) for k, v in cw.items()}, open(OUT / "class_weights_v2.json", "w"), indent=2)
joblib.dump(scaler, OUT / "feature_scaler_v2.joblib")

# ---------------------------------------------------------------
# 10. Summary / sanity checks
# ---------------------------------------------------------------
for name, yy in [("train", ytr), ("validation", yva), ("test", yte)]:
    print(f"{name}: customers={len(yy)}  churners={int(yy.sum())}  non-churners={int((yy == 0).sum())}")
print("class weights:", {k: round(float(v), 3) for k, v in cw.items()})

Xall = np.concatenate([Xtr_raw, Xva_raw, Xte_raw]); yall = np.concatenate([ytr, yva, yte])
act = Xall[:, :, 0] > 0
print("last-day active, churn:", act[yall == 1][:, -1].mean(), " non-churn:", act[yall == 0][:, -1].mean())
print("saved to", OUT)