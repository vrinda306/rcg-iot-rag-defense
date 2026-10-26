"""Random Forest baseline: multi-class attack-type classification on an Edge-IIoTset sample.

Design notes (for the viva):
- Target is Attack_type (15 classes) because the RAG knowledge base is looked up by attack class.
- Only numeric features are used. Labels, src_row_group and 4 sample-constant columns are excluded.
- Random Forest needs no scaling, so there is no preprocessing step that could leak test data.
- Binary (Normal vs Attack) metrics are derived by collapsing the multi-class predictions.
- Leakage check: we measure how many test rows have a feature vector that also occurs in training.
  This is a diagnostic only. It detects exact matches, not near-duplicates, and it does not prevent them.
  A grouped split can be run later as a separate, labelled sensitivity experiment.

Usage:
    python src/classifier/random_forest.py                                           # plain baseline
    python src/classifier/random_forest.py --class-weight balanced --tag balanced    # separate experiment
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
DATA = Path("data/processed/sample_v1.parquet")
RESULTS = Path("results")
MODELS = Path("models")

# Constant in the SAMPLE (not proven constant in the full dataset). Harmless to a Random Forest.
CONSTANT_IN_SAMPLE = {"icmp.checksum", "icmp.unused", "mqtt.conflags", "mqtt.hdrflags"}
EXCLUDE = {"Attack_label", "Attack_type", "src_row_group"} | CONSTANT_IN_SAMPLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1", help="suffix for output files")
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"Sample not found: {DATA.resolve()}")
    df = pd.read_parquet(DATA)

    features = [c for c in df.columns if c not in EXCLUDE]
    assert len(features) == 33, f"Expected 33 features, got {len(features)}"
    assert not (set(features) & EXCLUDE), "Label or diagnostic column leaked into features"
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in features), "Non-numeric feature found"

    X = df[features]
    y = df["Attack_type"]

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
    yt_bin = (y_test != "Normal").astype(int)
    yp_bin = (pd.Series(pred, index=y_test.index) != "Normal").astype(int)
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
    model_path = MODELS / f"rf_multiclass_{args.tag}.joblib"
    joblib.dump(model, model_path)

    report = classification_report(y_test, pred, labels=labels, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(RESULTS / f"rf_multiclass_{args.tag}_per_class.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        RESULTS / f"rf_multiclass_{args.tag}_confusion.csv")
    proba = model.predict_proba(X_test)
    pd.DataFrame({
        "row_in_sample": y_test.index,
        "true_label": y_test.to_numpy(),
        "predicted_label": pred,
        "confidence": np.round(proba.max(axis=1), 2),
    }).to_csv(RESULTS / f"rf_multiclass_{args.tag}_test_predictions.csv", index=False)
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv(RESULTS / f"rf_multiclass_{args.tag}_feature_importance.csv",
                       header=["importance"])

    summary = {
        "data": str(DATA),
        "seed": SEED,
        "tag": args.tag,
        "class_weight": args.class_weight,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(features),
        "features": features,
        "excluded_constant_in_sample": sorted(CONSTANT_IN_SAMPLE),
        "labels": labels,
        "model_params": model.get_params(),
        "train_seconds": round(train_seconds, 2),
        "multiclass": multi,
        "binary_derived": binary,
        "train_test_overlap": overlap,
        "top_confusions": top_confusions,
        "notes": "Trained on a class-capped sample, not the full dataset. "
                 "Numeric features only. Random stratified split, so identical or conflicting "
                 "feature vectors may appear in both train and test. "
                 "Results do not represent full-dataset performance.",
    }
    (RESULTS / f"rf_multiclass_{args.tag}_summary.json").write_text(
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