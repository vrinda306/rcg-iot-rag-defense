"""Random Forest baseline: multi-class attack-type classification on a CICIoT2023 sample.

Same procedure as the Edge-IIoTset baseline (seed 42, shuffle, stratified 80/20 on the
attack-type label, numeric features only), with one documented difference:
- Exact duplicate rows are dropped BEFORE the split (about 1.2% of this sample).
  The Edge-IIoTset sample keeps its 25 duplicates because they are negligible.

Usage (run from the repo's main folder):
    python src/classifier/random_forest_ciciot.py --n-estimators 20 --tag quick   # quick test
    python src/classifier/random_forest_ciciot.py                                  # full baseline
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

SEED = 42
DATA = Path("data/processed/sample_ciciot_v1.parquet")
RESULTS = Path("results")
MODELS = Path("models")
LABEL = "label"
NORMAL = "BenignTraffic"

# Constant in the SAMPLE (not proven constant in the full dataset). Harmless to a Random Forest.
CONSTANT_IN_SAMPLE = {"Telnet", "IRC", "DHCP"}
EXCLUDE = {LABEL} | CONSTANT_IN_SAMPLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1", help="suffix for output files")
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"Sample not found: {DATA.resolve()}")
    df = pd.read_parquet(DATA)

    # ---- Drop exact duplicates BEFORE the split ----
    rows_before = len(df)
    dup_mask = df.duplicated()
    dup_per_class = {k: int(v) for k, v in df.loc[dup_mask, LABEL].value_counts().items()}
    df = df.loc[~dup_mask].reset_index(drop=True)
    print(f"Rows before: {rows_before}, duplicates removed: {int(dup_mask.sum())}, rows after: {len(df)}")

    features = [c for c in df.columns if c not in EXCLUDE]
    assert len(features) == 43, f"Expected 43 features, got {len(features)}"
    assert not (set(features) & EXCLUDE), "Label or excluded column leaked into features"
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in features), "Non-numeric feature found"

    X = df[features]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True, stratify=y)
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows, Features: {len(features)}")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=SEED, n_jobs=-1,
        class_weight=None if args.class_weight == "none" else args.class_weight)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0
    print(f"Training time: {train_seconds:.1f} s")

    pred = model.predict(X_test)
    labels = sorted(y.unique())

    # ---- Multi-class metrics ----
    multi = {
        "accuracy": accuracy_score(y_test, pred),
        "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }

    # ---- Binary view derived by collapsing predictions ----
    yt_bin = (y_test != NORMAL).astype(int)
    yp_bin = (pd.Series(pred, index=y_test.index) != NORMAL).astype(int)
    binary = {
        "accuracy": accuracy_score(yt_bin, yp_bin),
        "precision": precision_score(yt_bin, yp_bin, zero_division=0),
        "recall": recall_score(yt_bin, yp_bin, zero_division=0),
        "f1": f1_score(yt_bin, yp_bin, zero_division=0),
        "confusion_matrix_[[TN,FP],[FN,TP]]": confusion_matrix(yt_bin, yp_bin).tolist(),
    }

    # ---- Leakage diagnostic: test rows whose exact feature vector also appears in training ----
    h_train = pd.util.hash_pandas_object(X_train, index=False).to_numpy()
    h_test = pd.util.hash_pandas_object(X_test, index=False).to_numpy()
    seen = np.isin(h_test, h_train)
    correct = (pred == y_test.to_numpy())
    overlap = {
        "test_rows_with_feature_vector_seen_in_train": int(seen.sum()),
        "fraction_of_test": float(seen.mean()),
        "accuracy_on_seen_rows": float(correct[seen].mean()) if seen.any() else None,
        "accuracy_on_unseen_rows": float(correct[~seen].mean()) if (~seen).any() else None,
    }

    # ---- Most confused class pairs ----
    cm = confusion_matrix(y_test, pred, labels=labels)
    off = [(labels[i], labels[j], int(cm[i, j]))
           for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    top_confusions = sorted(off, key=lambda t: t[2], reverse=True)[:10]

    # ---- Save outputs ----
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    model_path = MODELS / f"rf_ciciot_multiclass_{args.tag}.joblib"
    joblib.dump(model, model_path)

    report = classification_report(y_test, pred, labels=labels, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(RESULTS / f"rf_ciciot_multiclass_{args.tag}_per_class.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        RESULTS / f"rf_ciciot_multiclass_{args.tag}_confusion.csv")

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv(RESULTS / f"rf_ciciot_multiclass_{args.tag}_feature_importance.csv",
                       header=["importance"])

    summary = {
        "data": str(DATA),
        "seed": SEED,
        "tag": args.tag,
        "class_weight": args.class_weight,
        "rows_before_dedupe": rows_before,
        "duplicates_removed_before_split": int(dup_mask.sum()),
        "duplicates_removed_per_class": dup_per_class,
        "rows_after_dedupe": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(features),
        "features": features,
        "excluded_constant_in_sample": sorted(CONSTANT_IN_SAMPLE),
        "normal_label": NORMAL,
        "labels": labels,
        "model_params": model.get_params(),
        "train_seconds": round(train_seconds, 2),
        "multiclass": multi,
        "binary_derived": binary,
        "train_test_overlap": overlap,
        "top_confusions": top_confusions,
        "notes": "Trained on a class-capped sample of the CICIoT2023 train file, not the full dataset. "
                 "Numeric features only. Exact duplicates removed before a random stratified split. "
                 "Near-duplicate feature vectors may still appear in both train and test. "
                 "Results do not represent full-dataset performance.",
    }
    (RESULTS / f"rf_ciciot_multiclass_{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")

    # ---- Console report ----
    print("\nMulti-class metrics:")
    for k, v in multi.items():
        print(f"  {k}: {v:.4f}")
    print("\nBinary (derived) metrics:")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k}: {binary[k]:.4f}")
    print("  confusion [[TN,FP],[FN,TP]]:", binary["confusion_matrix_[[TN,FP],[FN,TP]]"])
    print("\nTrain/test overlap check:")
    for k, v in overlap.items():
        print(f"  {k}: {v}")
    print("\nPer-class report:")
    print(classification_report(y_test, pred, labels=labels, zero_division=0))
    print("Top confused pairs (true -> predicted, count):")
    for t, p, n in top_confusions:
        print(f"  {t} -> {p}: {n}")
    print("\nTop 10 features:")
    print(importances.head(10).to_string())
    print("\nSaved model to", model_path, "and results to", RESULTS.resolve())


if __name__ == "__main__":
    main()