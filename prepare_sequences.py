
import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "mini_sparkify_event_data.json"
OUT = BASE / "model_results"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 30
CUTOFF_DATE = pd.Timestamp("2018-11-03")
FEATURES = [
    "event_count",
    "sessions",
    "songs",
    "minutes_listened",
    "thumbs_up",
    "thumbs_down",
    "add_to_playlist",
    "add_friend",
    "roll_advert",
    "downgrade",
    "upgrade",
    "home",
    "errors",
]

SEED = 42



df = pd.read_json(DATA_FILE, lines=True)

df["userId"] = df["userId"].astype("string").str.strip()
df = df[
    df["userId"].notna()
    & (df["userId"] != "")
].copy()

df["userId"] = df["userId"].astype(str)
df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
df = df.dropna(subset=["ts", "page"]).copy()
df["ts"] = df["ts"].astype("int64")

df["datetime"] = pd.to_datetime(
    df["ts"], unit="ms"
)
df["date"] = df["datetime"].dt.normalize()

cutoff_ts = int(
    CUTOFF_DATE.normalize().timestamp() * 1000
) + (24 * 60 * 60 * 1000 - 1)

dataset_end_ts = int(df["ts"].max())

if dataset_end_ts <= cutoff_ts:
    raise ValueError(
        "The dataset does not contain data after the chosen cutoff."
    )


cancel_events = df[
    df["page"] == "Cancellation Confirmation"
].copy()

first_cancel_ts = (
    cancel_events.groupby("userId")["ts"].min()
)

# Customers who have already cancelled by the cutoff
cancelled_before_or_at_cutoff = set(
    first_cancel_ts[
        first_cancel_ts <= cutoff_ts
    ].index
)

# Customers with at least one recorded event by cutoff
observed_customers = set(
    df.loc[df["ts"] <= cutoff_ts, "userId"].unique()
)

# Only customers who were still eligible for prediction
eligible_customers = sorted(
    observed_customers - cancelled_before_or_at_cutoff
)

if not eligible_customers:
    raise ValueError(
        "No eligible customers remain. Check the cutoff date."
    )

future_cancel_ids = set(
    first_cancel_ts[
        (first_cancel_ts > cutoff_ts)
        & (first_cancel_ts <= dataset_end_ts)
    ].index
)

labels = {
    uid: int(uid in future_cancel_ids)
    for uid in eligible_customers
}

y_all = np.array(
    [labels[uid] for uid in eligible_customers],
    dtype=np.int64,
)

class_counts = np.bincount(y_all, minlength=2)

if class_counts.min() < 2:
    raise ValueError(
        "There are too few examples in one class for a "
        "stratified train/validation/test split. "
        "Choose a suitable cutoff and prediction period."
    )

print("Prediction cut-off:", CUTOFF_DATE.date())
print("Dataset ends:", df["datetime"].max())
print("Prediction period: after cutoff through dataset end")
print("Eligible customers:", len(eligible_customers))
print("Future churners:", int(y_all.sum()))
print("Non-churners in observed prediction period:",
      int((y_all == 0).sum()))



# Input features must come from on or before the cutoff.
ev = df[
    (df["ts"] <= cutoff_ts)
    & (df["page"] != "Cancellation Confirmation")
].copy()

ev["is_song"] = (
    ev["page"] == "NextSong"
).astype(int)

ev["minutes_row"] = np.where(
    ev["page"] == "NextSong",
    pd.to_numeric(
        ev["length"], errors="coerce"
    ).fillna(0) / 60,
    0.0,
)

page_to_feature = {
    "Thumbs Up": "thumbs_up",
    "Thumbs Down": "thumbs_down",
    "Add to Playlist": "add_to_playlist",
    "Add Friend": "add_friend",
    "Roll Advert": "roll_advert",
    "Downgrade": "downgrade",
    "Upgrade": "upgrade",
    "Home": "home",
    "Error": "errors",
}

for page, feature in page_to_feature.items():
    ev[feature] = (ev["page"] == page).astype(int)

daily = (
    ev[ev["userId"].isin(eligible_customers)]
    .groupby(["userId", "date"])
    .agg(
        event_count=("page", "size"),
        sessions=("sessionId", "nunique"),
        songs=("is_song", "sum"),
        minutes_listened=("minutes_row", "sum"),
        thumbs_up=("thumbs_up", "sum"),
        thumbs_down=("thumbs_down", "sum"),
        add_to_playlist=("add_to_playlist", "sum"),
        add_friend=("add_friend", "sum"),
        roll_advert=("roll_advert", "sum"),
        downgrade=("downgrade", "sum"),
        upgrade=("upgrade", "sum"),
        home=("home", "sum"),
        errors=("errors", "sum"),
    )
    .reset_index()
)

daily_lookup = {
    (row.userId, row.date): np.array(
        [getattr(row, col) for col in FEATURES],
        dtype=np.float32,
    )
    for row in daily.itertuples(index=False)
}

sequence_dates = pd.date_range(
    end=CUTOFF_DATE,
    periods=WINDOW,
    freq="D",
)

X_raw = np.zeros(
    (len(eligible_customers), WINDOW, len(FEATURES)),
    dtype=np.float32,
)

for i, uid in enumerate(eligible_customers):
    for j, day in enumerate(sequence_dates):
        record = daily_lookup.get((uid, day))

        if record is not None:
            X_raw[i, j, :] = record



indices = np.arange(len(eligible_customers))

train_idx, temporary_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y_all,
)

val_idx, test_idx = train_test_split(
    temporary_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=y_all[temporary_idx],
)

# Fit scaler using active days from training customers only.
X_train_raw = X_raw[train_idx]
train_rows = X_train_raw.reshape(-1, len(FEATURES))
active_train_rows = train_rows[train_rows[:, 0] > 0]

if len(active_train_rows) == 0:
    raise ValueError("No active training days found.")

scaler = StandardScaler()
scaler.fit(active_train_rows)

def scale_sequences(X):
    scaled = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        active = X[i, :, 0] > 0

        if active.any():
            scaled[i, active, :] = scaler.transform(
                X[i, active, :]
            )

    return np.nan_to_num(
        scaled, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

X_train = scale_sequences(X_raw[train_idx])
X_val = scale_sequences(X_raw[val_idx])
X_test = scale_sequences(X_raw[test_idx])

y_train = y_all[train_idx]
y_val = y_all[val_idx]
y_test = y_all[test_idx]

train_ids = np.array(
    [eligible_customers[i] for i in train_idx]
)
val_ids = np.array(
    [eligible_customers[i] for i in val_idx]
)
test_ids = np.array(
    [eligible_customers[i] for i in test_idx]
)



counts = np.bincount(y_train, minlength=2)

if counts.min() == 0:
    raise ValueError(
        "Training split is missing a class; adjust the split."
    )

class_weights = {
    0: float(len(y_train) / (2 * counts[0])),
    1: float(len(y_train) / (2 * counts[1])),
}



np.savez(
    OUT / "sequence_data_v2.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    train_user_ids=train_ids,
    val_user_ids=val_ids,
    test_user_ids=test_ids,
    feature_names=np.array(FEATURES),
)

joblib.dump(
    scaler,
    OUT / "feature_scaler_v2.joblib",
)

with open(
    OUT / "class_weights_v2.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {str(k): v for k, v in class_weights.items()},
        f,
        indent=2,
    )

metadata = pd.DataFrame({
    "userId": eligible_customers,
    "churn_in_prediction_period": y_all,
    "observation_cutoff": CUTOFF_DATE.strftime("%Y-%m-%d"),
    "observation_days": WINDOW,
    "prediction_period_end": df["datetime"].max().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
})

split_map = {}
for uid in train_ids:
    split_map[uid] = "train"
for uid in val_ids:
    split_map[uid] = "validation"
for uid in test_ids:
    split_map[uid] = "test"

metadata["split"] = metadata["userId"].map(split_map)

metadata.to_csv(
    OUT / "customer_metadata_and_splits_v2.csv",
    index=False,
)

summary = {
    "observation_cutoff": str(CUTOFF_DATE.date()),
    "observation_days": WINDOW,
    "dataset_end": str(df["datetime"].max()),
    "eligible_customers": len(eligible_customers),
    "future_churners": int(y_all.sum()),
    "non_churners_in_observed_period": int((y_all == 0).sum()),
    "train_customers": len(train_idx),
    "validation_customers": len(val_idx),
    "test_customers": len(test_idx),
    "train_churners": int(y_train.sum()),
    "validation_churners": int(y_val.sum()),
    "test_churners": int(y_test.sum()),
    "features": FEATURES,
    "class_weights": class_weights,
}

with open(
    OUT / "member1_prediction_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

print("\nSaved corrected sequence data to:", OUT)
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
print("Training class weights:", class_weights)
print("Summary saved to member1_prediction_summary.json")
