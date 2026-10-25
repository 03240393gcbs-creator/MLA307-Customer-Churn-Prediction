"""
Member 1 – Data Lead
MLA307 Sequence Models Mini Project: Customer Churn Prediction
Run from VS Code with mini_sparkify_event_data.json in the same folder.

Outputs are saved in ./member1_outputs/
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SEED = 42
WINDOW = 30
DATA_PATH = Path("mini_sparkify_event_data.json")
OUT_DIR = Path("member1_outputs")

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


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Load and inspect the raw Sparkify event data.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH.resolve()}. Put the JSON file beside this script "
            "or change DATA_PATH near the top of this file."
        )

    print("Loading dataset (this may take a little time)...")
    df = pd.read_json(DATA_PATH, lines=True)
    required = {"ts", "userId", "sessionId", "page", "length"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    raw_rows = len(df)
    raw_columns = list(df.columns)
    exact_duplicates = int(df.duplicated().sum())
    missing_by_column = df.isna().sum().sort_values(ascending=False)
    df["userId"] = df["userId"].astype("string").str.strip()
    # Empty user IDs cannot be assigned to a customer sequence.
    empty_user_rows = int((df["userId"].isna() | (df["userId"] == "")).sum())
    df = df[df["userId"].notna() & (df["userId"] != "")].copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "userId", "page"]).copy()
    df["ts"] = df["ts"].astype("int64")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df["date"] = df["datetime"].dt.normalize()

    # 2. Define churn: a customer is a churner if their history contains
    #    "Cancellation Confirmation". The confirmation event itself will
    #    not be used as a model input.
    churn_events = df[df["page"].eq("Cancellation Confirmation")].copy()
    churn_ts = churn_events.groupby("userId")["ts"].min()
    all_users = sorted(df["userId"].unique().tolist())
    labels = pd.Series(0, index=all_users, dtype="int64", name="churn")
    labels.loc[labels.index.isin(churn_ts.index)] = 1

    # Keep an event-level file for the report's descriptive statistics.
    report = {
        "dataset_file": DATA_PATH.name,
        "raw_records": int(raw_rows),
        "records_after_removing_empty_user_ids": int(len(df)),
        "raw_columns": raw_columns,
        "number_of_columns": int(len(raw_columns)),
        "unique_customers_with_user_id": int(df["userId"].nunique()),
        "empty_user_id_rows_excluded": empty_user_rows,
        "exact_duplicate_rows_in_raw_data": exact_duplicates,
        "dataset_start": str(df["datetime"].min()),
        "dataset_end": str(df["datetime"].max()),
        "cancellation_confirmation_events": int(len(churn_events)),
        "unique_churned_customers": int(labels.sum()),
        "non_churned_customers": int((labels == 0).sum()),
        "churn_definition": 'Customer has a "Cancellation Confirmation" event.',
        "sequence_length_days": WINDOW,
        "features": FEATURES,
        "note": "The cancellation confirmation event is excluded from input features.",
    }
    with open(OUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    missing_by_column.rename_axis("column").reset_index(name="missing_count").to_csv(
        OUT_DIR / "missing_values.csv", index=False
    )
    pd.DataFrame({
        "page": df["page"].value_counts().index,
        "event_count": df["page"].value_counts().values,
    }).to_csv(OUT_DIR / "page_event_counts.csv", index=False)

    print("\n========== DATASET SUMMARY ==========")
    print(f"Raw records: {raw_rows:,}")
    print(f"Columns: {len(raw_columns)}")
    print(f"Customers with a user ID: {df['userId'].nunique():,}")
    print(f"Rows with empty user ID excluded: {empty_user_rows:,}")
    print(f"Exact duplicate rows in raw data: {exact_duplicates:,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Churned customers: {int(labels.sum())}")
    print(f"Non-churned customers: {int((labels == 0).sum())}")
    print("Missing values by column:")
    print(missing_by_column.to_string())

    # 3. Remove the cancellation event before constructing input features.
    #    For churners, only events strictly before their first cancellation
    #    timestamp are retained. For non-churners, use all available events.
    feature_events = df[df["page"].ne("Cancellation Confirmation")].copy()
    churner_mask = feature_events["userId"].isin(churn_ts.index)
    feature_events = feature_events[
        (~churner_mask) | (
            feature_events["ts"] < feature_events["userId"].map(churn_ts)
        )
    ].copy()

    # Build daily features from event logs.
    feature_events["is_song"] = feature_events["page"].eq("NextSong").astype(int)
    feature_events["minutes_listened_row"] = np.where(
        feature_events["page"].eq("NextSong"),
        pd.to_numeric(feature_events["length"], errors="coerce").fillna(0) / 60.0,
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
    for page_name, feature_name in page_to_feature.items():
        feature_events[feature_name] = feature_events["page"].eq(page_name).astype(int)

    daily = feature_events.groupby(["userId", "date"], sort=False).agg(
        event_count=("page", "size"),
        sessions=("sessionId", "nunique"),
        songs=("is_song", "sum"),
        minutes_listened=("minutes_listened_row", "sum"),
        thumbs_up=("thumbs_up", "sum"),
        thumbs_down=("thumbs_down", "sum"),
        add_to_playlist=("add_to_playlist", "sum"),
        add_friend=("add_friend", "sum"),
        roll_advert=("roll_advert", "sum"),
        downgrade=("downgrade", "sum"),
        upgrade=("upgrade", "sum"),
        home=("home", "sum"),
        errors=("errors", "sum"),
    ).reset_index()

    daily_lookup = {
        (row.userId, row.date): row._asdict()
        for row in daily.itertuples(index=False)
    }
    global_last_date = df["date"].max()

    # Create one 30-day sequence per customer. Day 1 is the oldest day and
    # day 30 is the cutoff day. Missing days are represented by all-zero rows.
    user_ids = []
    sequence_list = []
    label_list = []
    metadata_rows = []
    daily_long_rows = []
    for user_id in all_users:
        y = int(labels.loc[user_id])
        if y == 1:
            cutoff_ts = int(churn_ts.loc[user_id])
            cutoff_date = pd.to_datetime(cutoff_ts, unit="ms").normalize()
        else:
            cutoff_ts = int(df.loc[df["userId"].eq(user_id), "ts"].max())
            cutoff_date = global_last_date

        dates = pd.date_range(end=cutoff_date, periods=WINDOW, freq="D")
        seq = np.zeros((WINDOW, len(FEATURES)), dtype=np.float32)
        for day_index, day in enumerate(dates):
            record = daily_lookup.get((user_id, day))
            if record is not None:
                seq[day_index] = np.array([record.get(col, 0) for col in FEATURES], dtype=np.float32)
            daily_long_rows.append({
                "userId": user_id,
                "churn": y,
                "day_index": day_index + 1,
                "date": day.strftime("%Y-%m-%d"),
                **{FEATURES[j]: float(seq[day_index, j]) for j in range(len(FEATURES))}
            })

        user_ids.append(user_id)
        sequence_list.append(seq)
        label_list.append(y)
        metadata_rows.append({
            "userId": user_id,
            "churn": y,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "cutoff_timestamp": cutoff_ts,
            "sequence_days": WINDOW,
        })

    X_raw = np.stack(sequence_list).astype(np.float32)
    y_all = np.array(label_list, dtype=np.int64)
    user_ids = np.array(user_ids, dtype=str)
    metadata = pd.DataFrame(metadata_rows)

    # 4. Split by customer (never by individual event/day) into 70/15/15.
    indices = np.arange(len(user_ids))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=y_all
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y_all[temp_idx]
    )

    # 5. Fit the scaler using training customers only. Empty/no-activity days
    #    remain all-zero padding after scaling.
    scaler = StandardScaler()
    train_rows = X_raw[train_idx].reshape(-1, len(FEATURES))
    active_train_rows = train_rows[train_rows[:, 0] > 0]
    if len(active_train_rows) == 0:
        raise ValueError("No active training days found; cannot fit the scaler.")
    scaler.fit(active_train_rows)

    X_scaled = np.zeros_like(X_raw, dtype=np.float32)
    for i in range(len(user_ids)):
        active_mask = X_raw[i, :, 0] > 0
        if np.any(active_mask):
            X_scaled[i, active_mask, :] = scaler.transform(X_raw[i, active_mask, :]).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Class weights help account for fewer churners. Member 2 can use this
    # dictionary directly as Keras model.fit(class_weight=CLASS_WEIGHTS).
    train_counts = np.bincount(y_all[train_idx], minlength=2)
    total_train = int(len(train_idx))
    class_weights = {
        int(c): float(total_train / (2 * max(int(train_counts[c]), 1)))
        for c in (0, 1)
    }

    # Save the arrays for Member 2.
    np.savez_compressed(
        OUT_DIR / "sequence_data.npz",
        X_train=X_scaled[train_idx],
        y_train=y_all[train_idx],
        X_val=X_scaled[val_idx],
        y_val=y_all[val_idx],
        X_test=X_scaled[test_idx],
        y_test=y_all[test_idx],
        train_user_ids=user_ids[train_idx],
        val_user_ids=user_ids[val_idx],
        test_user_ids=user_ids[test_idx],
        feature_names=np.array(FEATURES),
    )
    joblib.dump(scaler, OUT_DIR / "feature_scaler.joblib")
    with open(OUT_DIR / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    split_map = {}
    for i in train_idx:
        split_map[user_ids[i]] = "train"
    for i in val_idx:
        split_map[user_ids[i]] = "validation"
    for i in test_idx:
        split_map[user_ids[i]] = "test"
    metadata["split"] = metadata["userId"].map(split_map)
    metadata.to_csv(OUT_DIR / "customer_metadata_and_splits.csv", index=False)
    pd.DataFrame(daily_long_rows).to_csv(OUT_DIR / "daily_sequences.csv", index=False)

    # 6. Chart: average songs and listening minutes across the 30-day sequence.
    #    This describes patterns before churn/cutoff; it does not prove causation.
    daily_long = pd.DataFrame(daily_long_rows)
    daily_long["relative_day"] = daily_long["day_index"] - WINDOW
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, name in [(1, "Churners"), (0, "Non-churners")]:
        subset = daily_long[daily_long["churn"] == label]
        mean_songs = subset.groupby("day_index")["songs"].mean()
        mean_minutes = subset.groupby("day_index")["minutes_listened"].mean()
        axes[0].plot(range(1, WINDOW + 1), mean_songs.reindex(range(1, WINDOW + 1), fill_value=0), label=name)
        axes[1].plot(range(1, WINDOW + 1), mean_minutes.reindex(range(1, WINDOW + 1), fill_value=0), label=name)
    axes[0].set_title("Average Songs per Day Across the 30-Day Sequence")
    axes[0].set_ylabel("Mean songs")
    axes[1].set_title("Average Listening Minutes per Day Across the 30-Day Sequence")
    axes[1].set_ylabel("Mean minutes")
    axes[1].set_xlabel("Sequence day (30 = cutoff day)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "churn_vs_nonchurn_activity.png", dpi=160)
    plt.close(fig)

    # 7. A report-ready text file summarising preparation choices.
    summary_lines = [
        "MEMBER 1 – DATASET AND PREPARATION SUMMARY",
        "",
        f"Dataset file: {DATA_PATH.name}",
        "Dataset source: Udacity Sparkify event-log dataset (the supplied mini Sparkify sample).",
        f"Raw records: {raw_rows:,}",
        f"Records after excluding rows without a user ID: {len(df):,}",
        f"Number of original columns: {len(raw_columns)}",
        f"Number of customers with a user ID: {df['userId'].nunique():,}",
        f"Date range: {df['datetime'].min()} to {df['datetime'].max()}",
        f"Exact duplicate rows: {exact_duplicates:,}",
        f"Rows with empty user ID excluded: {empty_user_rows:,}",
        f"Churned customers: {int(labels.sum()):,}",
        f"Non-churned customers: {int((labels == 0).sum()):,}",
        'Target variable: churn (1 = a "Cancellation Confirmation" event; 0 = no such event in the supplied data).',
        f"Sequence: {WINDOW} daily time steps per customer, oldest day first and cutoff day last.",
        "Features: " + ", ".join(FEATURES),
        "Preparation: removed rows without user IDs; converted millisecond timestamps to dates; aggregated events by customer/day; created fixed-length sequences; represented days with no recorded activity using zeros.",
        "Leakage prevention: the cancellation confirmation event is not an input feature; for churners, events at or after the first cancellation timestamp are excluded.",
        "Scaling: StandardScaler fitted only on active days from the training customers; zero-activity/padding days remain zero.",
        "Split: stratified customer-level split, 70% train / 15% validation / 15% test, random seed 42.",
        f"Training class weights: {json.dumps(class_weights)}",
        "Important limitation: a customer without a cancellation event is labelled non-churned only within the observed dataset window; they may churn later.",
        "Business use: Member 2 can train a sequence model on sequence_data.npz; after inference, merge predicted churn probabilities with customer_metadata_and_splits.csv to prepare the at-risk list.",
        "",
        "Outputs are in the member1_outputs folder.",
    ]
    (OUT_DIR / "member1_report_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n========== SEQUENCE DATA READY ==========")
    print("Features:", FEATURES)
    print("X_train:", X_scaled[train_idx].shape, "y_train:", y_all[train_idx].shape)
    print("X_val:  ", X_scaled[val_idx].shape, "y_val:  ", y_all[val_idx].shape)
    print("X_test: ", X_scaled[test_idx].shape, "y_test: ", y_all[test_idx].shape)
    print("Class weights:", class_weights)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")
    print("Key files: sequence_data.npz, member1_report_summary.txt, dataset_summary.json,")
    print("           customer_metadata_and_splits.csv, daily_sequences.csv,")
    print("           churn_vs_nonchurn_activity.png, class_weights.json")


if __name__ == "__main__":
    main()
