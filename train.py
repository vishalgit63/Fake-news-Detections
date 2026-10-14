import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import preprocess_text


RANDOM_STATE = 42


def load_dataset(csv_path: str, fake_label="1"):
    df = pd.read_csv(csv_path, low_memory=False)

    if "text" not in df.columns and "title" not in df.columns:
        raise ValueError("CSV must contain a 'text' column, a 'title' column, or both.")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    title = df["title"].fillna("").astype(str) if "title" in df.columns else ""
    body = df["text"].fillna("").astype(str) if "text" in df.columns else ""

    if isinstance(title, str):
        combined = body
    elif isinstance(body, str):
        combined = title
    else:
        combined = (title + " " + body).str.strip()

    raw_label = df["label"].astype(str).str.strip().str.lower()
    fake_label_norm = str(fake_label).strip().lower()

    # Supports labels such as FAKE/REAL or 1/0.
    named_label_mask = raw_label.isin({"fake", "real"})
    if named_label_mask.any():
        df = df.loc[named_label_mask].copy()
        raw_label = df["label"].astype(str).str.strip().str.lower()
        y = raw_label.map({"real": 0, "fake": 1})
    else:
        y = (raw_label == fake_label_norm).astype(int)

    data = pd.DataFrame({"text": combined, "label": y})
    data = data.dropna().drop_duplicates(subset=["text"])
    data = data[data["text"].str.len() > 10]

    if data["label"].nunique() != 2:
        raise ValueError(
            "The processed label column does not contain two classes. "
            "Check --fake-label and your dataset labels."
        )
    return data


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["REAL", "FAKE"], zero_division=0
        ),
    }, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train a fake-news classifier.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset.")
    parser.add_argument(
        "--preprocess",
        choices=["stemming", "lemmatization", "none"],
        default="lemmatization",
        help="Text normalization method.",
    )
    parser.add_argument(
        "--fake-label",
        default="1",
        help="Value that represents FAKE for numeric/custom labels. Default: 1",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args.data, args.fake_label)
    print(f"Rows after cleaning: {len(df):,}")
    print(df["label"].value_counts().rename(index={0: "REAL", 1: "FAKE"}))

    print(f"\nPreprocessing using: {args.preprocess}")
    X = df["text"].apply(lambda x: preprocess_text(x, args.preprocess))
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
    )

    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "clf__C": [0.5, 1.0, 2.0, 5.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "Linear SVM": (
            LinearSVC(random_state=RANDOM_STATE),
            {"clf__C": [0.25, 0.5, 1.0, 2.0]},
        ),
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
        "Passive Aggressive": (
            PassiveAggressiveClassifier(
                max_iter=2000, random_state=RANDOM_STATE, tol=1e-3
            ),
            {
                "clf__C": [0.1, 0.5, 1.0],
                "clf__loss": ["hinge", "squared_hinge"],
            },
        ),
    }

    results = {}
    fitted_models = {}

    for name, (classifier, param_grid) in candidates.items():
        print(f"\nTuning {name}...")
        pipe = Pipeline(
            [
                ("tfidf", clone(tfidf)),
                ("clf", classifier),
            ]
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        metrics, y_pred = evaluate(grid.best_estimator_, X_test, y_test)
        metrics["best_params"] = grid.best_params_
        metrics["cv_best_f1"] = float(grid.best_score_)
        results[name] = metrics
        fitted_models[name] = grid.best_estimator_

        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )

    best_name = max(results, key=lambda n: results[n]["cv_best_f1"])
    best_model = fitted_models[best_name]

    bundle = {
        "model": best_model,
        "preprocess_method": args.preprocess,
        "best_model_name": best_name,
        "labels": {0: "REAL", 1: "FAKE"},
    }
    joblib.dump(bundle, out_dir / "fake_news_model.joblib")

    summary = {
        "best_model": best_name,
        "preprocess_method": args.preprocess,
        "dataset_rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "results": results,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save confusion matrix for the best model.
    best_pred = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_pred,
        display_labels=["REAL", "FAKE"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": info["accuracy"],
                "precision": info["precision"],
                "recall": info["recall"],
                "f1": info["f1"],
                "cv_best_f1": info["cv_best_f1"],
                "best_params": str(info["best_params"]),
            }
            for name, info in results.items()
        ]
    ).sort_values("f1", ascending=False)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    print("\nBest model:", best_name)
    print(comparison[["model", "accuracy", "precision", "recall", "f1"]].to_string(index=False))
    print("\nSaved:")
    print("  artifacts/fake_news_model.joblib")
    print("  artifacts/metrics.json")
    print("  artifacts/model_comparison.csv")
    print("  artifacts/confusion_matrix.png")


if __name__ == "__main__":
    main()
