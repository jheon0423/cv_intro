import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from dataset import CLASS_NAMES, load_cifar10, select_subset
from knn import KNNClassifier


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(10), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def write_csv(path, records, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def cross_validate(x, y, k_values, metrics, folds, batch_size, seed):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    records = []
    total_steps = folds * len(metrics)
    step = 0

    for fold, (train_indices, valid_indices) in enumerate(splitter.split(x, y), start=1):
        train_x, train_y = x[train_indices], y[train_indices]
        valid_x, valid_y = x[valid_indices], y[valid_indices]
        for metric in metrics:
            step += 1
            model = KNNClassifier(metric=metric).fit(train_x, train_y)
            predictions = model.predict_many(valid_x, k_values, batch_size=batch_size)
            for k in k_values:
                records.append({
                    "fold": fold,
                    "metric": metric,
                    "k": k,
                    **calculate_metrics(valid_y, predictions[k]),
                })
            print(f"Cross-validation {step}/{total_steps}: fold={fold}, metric={metric}", flush=True)
    return records


def aggregate_cv(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["metric"], record["k"])].append(record)

    summary = []
    for (metric, k), rows in grouped.items():
        result = {"metric": metric, "k": k}
        for score in ("accuracy", "precision", "recall", "f1"):
            values = np.asarray([row[score] for row in rows], dtype=float)
            result[f"{score}_mean"] = float(np.mean(values))
            result[f"{score}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    return sorted(summary, key=lambda row: (row["metric"], row["k"]))


def format_table(summary):
    lines = [
        "| Distance | k | Accuracy | Precision | Recall | F1-score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: (-item["accuracy_mean"], item["k"])):
        cells = [
            row["metric"], str(row["k"]),
            *[f"{row[f'{score}_mean']:.4f} ± {row[f'{score}_std']:.4f}" for score in
              ("accuracy", "precision", "recall", "f1")],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_report(summary, selected, test_result, train_count, test_count, folds):
    metric_winners = {}
    for row in summary:
        metric = row["metric"]
        if metric not in metric_winners or row["accuracy_mean"] > metric_winners[metric]["accuracy_mean"]:
            metric_winners[metric] = row
    metric_comparison = ", ".join(
        f"{metric}: k={row['k']} (Accuracy {row['accuracy_mean']:.4f})"
        for metric, row in sorted(metric_winners.items())
    )
    k_comparison = []
    for metric in sorted(metric_winners):
        rows = sorted([row for row in summary if row["metric"] == metric], key=lambda item: item["k"])
        first, last = rows[0], rows[-1]
        delta = last["accuracy_mean"] - first["accuracy_mean"]
        k_comparison.append(
            f"- {metric}: k={first['k']} → k={last['k']}에서 평균 Accuracy 변화 {delta:+.4f}. "
            f"해당 거리 척도의 최적 k는 {metric_winners[metric]['k']}."
        )

    return "\n".join([
        "# CIFAR-10 KNN Experiment Report",
        "",
        "## Experimental setup",
        "",
        f"- Training samples: {train_count}",
        f"- Test samples: {test_count}",
        "- Features: 32×32×3 pixels, flattened to 3072 dimensions; pixel intensities divided by 255",
        f"- Validation: {folds}-fold stratified cross-validation on training data only",
        "- Metrics: Accuracy, macro Precision, macro Recall, macro F1-score",
        "- Tie handling: smallest class label is selected when votes are tied",
        "- Model selection: highest mean cross-validation Accuracy, then macro F1-score",
        "",
        "## Cross-validation results (mean ± sample standard deviation)",
        "",
        format_table(summary),
        "",
        "## Selected configuration and held-out test results",
        "",
        f"- Distance metric: **{selected['metric']}**",
        f"- k: **{selected['k']}**",
        f"- Test Accuracy: **{test_result['accuracy']:.4f}**",
        f"- Test macro Precision: **{test_result['precision']:.4f}**",
        f"- Test macro Recall: **{test_result['recall']:.4f}**",
        f"- Test macro F1-score: **{test_result['f1']:.4f}**",
        "",
        "## Brief comparison and explanation",
        "",
        f"- Best k for each distance metric: {metric_comparison}.",
        *k_comparison,
        "- Euclidean distance measures straight-line pixel differences, Manhattan distance sums absolute pixel differences, and cosine distance compares the direction of pixel vectors.",
        "- A small k can be sensitive to individual images or noise; a large k smooths predictions but can include neighbors from other classes.",
        "- Raw-pixel KNN ignores spatial structure such as edges and object positions, which can limit image-classification performance.",
        "- The test set was not used to tune k or the distance metric.",
        "",
        "## Output files",
        "",
        "- cv_folds.csv: metrics for each configuration and fold",
        "- cv_summary.csv: cross-validation means and standard deviations",
        "- test_metrics.csv: held-out test metrics for the selected configuration",
        "- test_per_class.csv: per-class Precision, Recall and F1-score",
        "- test_confusion_matrix.csv: true-class rows and predicted-class columns",
        "",
    ])


def parse_arguments():
    parser = argparse.ArgumentParser(description="CIFAR-10 KNN with distance comparisons and 5-fold CV")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--metrics", nargs="+", choices=KNNClassifier.supported_metrics, default=list(KNNClassifier.supported_metrics))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.train_size < 0 or args.test_size < 0:
        raise ValueError("Sample sizes must be nonnegative; 0 means full dataset")
    if args.folds < 2:
        raise ValueError("At least two folds are required")
    k_values = sorted(set(args.k_values))
    if not k_values or k_values[0] < 1:
        raise ValueError("All k values must be positive")
    metrics = list(dict.fromkeys(args.metrics))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_x, raw_train_y, raw_test_x, raw_test_y = load_cifar10(args.data)
    train_x, train_y = select_subset(raw_train_x, raw_train_y, args.train_size, args.seed)
    test_x, test_y = select_subset(raw_test_x, raw_test_y, args.test_size, args.seed + 1)
    del raw_train_x, raw_test_x, raw_train_y, raw_test_y

    label_counts = np.bincount(train_y, minlength=10)
    if np.min(label_counts) < args.folds:
        raise ValueError("Each class must contain at least as many samples as the number of folds")
    minimum_train_fold_size = len(train_y) - int(np.ceil(len(train_y) / args.folds))
    if k_values[-1] > minimum_train_fold_size:
        raise ValueError("Largest k cannot exceed the number of training samples in a CV fold")

    print(f"Train: {len(train_y)} | Test: {len(test_y)} | Features: {train_x.shape[1]}", flush=True)
    fold_results = cross_validate(train_x, train_y, k_values, metrics, args.folds, args.batch_size, args.seed)
    summary = aggregate_cv(fold_results)
    selected = max(summary, key=lambda row: (row["accuracy_mean"], row["f1_mean"], -row["k"]))

    fold_columns = ["fold", "metric", "k", "accuracy", "precision", "recall", "f1"]
    summary_columns = ["metric", "k"] + [
        f"{score}_{stat}" for score in ("accuracy", "precision", "recall", "f1") for stat in ("mean", "std")
    ]
    write_csv(output_dir / "cv_folds.csv", fold_results, fold_columns)
    write_csv(output_dir / "cv_summary.csv", summary, summary_columns)

    final_model = KNNClassifier(metric=selected["metric"]).fit(train_x, train_y)
    test_prediction = final_model.predict(test_x, k=selected["k"], batch_size=args.batch_size)
    test_result = calculate_metrics(test_y, test_prediction)
    write_csv(
        output_dir / "test_metrics.csv",
        [{"metric": selected["metric"], "k": selected["k"], **test_result}],
        ["metric", "k", "accuracy", "precision", "recall", "f1"],
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        test_y, test_prediction, labels=np.arange(10), zero_division=0
    )
    class_results = [
        {"class_id": i, "class_name": name, "precision": float(precision[i]),
         "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i, name in enumerate(CLASS_NAMES)
    ]
    write_csv(
        output_dir / "test_per_class.csv", class_results,
        ["class_id", "class_name", "precision", "recall", "f1", "support"],
    )

    matrix = confusion_matrix(test_y, test_prediction, labels=np.arange(10))
    matrix_rows = [
        {"true_class": name, **{CLASS_NAMES[j]: int(matrix[i, j]) for j in range(10)}}
        for i, name in enumerate(CLASS_NAMES)
    ]
    write_csv(output_dir / "test_confusion_matrix.csv", matrix_rows, ["true_class", *CLASS_NAMES])

    report = make_report(summary, selected, test_result, len(train_y), len(test_y), args.folds)
    (output_dir / "experiment_report.md").write_text(report, encoding="utf-8")
    print(f"Best: metric={selected['metric']}, k={selected['k']}")
    print("Test " + " | ".join(f"{key}={value:.4f}" for key, value in test_result.items()))
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
