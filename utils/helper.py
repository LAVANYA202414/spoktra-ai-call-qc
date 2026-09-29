import json
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
    f1_score
)


class EvaluationUtils:
    def __init__(self, labels):
        self.labels = labels

    # LABEL UTILITIES
    def issues_to_labels(self, issues):
        """
        Convert issue list to multi-hot vector
        """
        return [1 if label in issues else 0 for label in self.labels]

    # CLASSIFICATION METRICS
    def compute_classification_metrics(self, y_true, y_pred):
        """
        Compute precision, recall, f1 per label
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0
        )

        metrics = {}
        for i, label in enumerate(self.labels):
            metrics[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }

        return metrics

    # REGRESSION METRICS
    def compute_regression_metrics(self, y_true_scores, y_pred_scores):
        """
        Compute MAE and RMSE for quality score
        """
        mae = mean_absolute_error(y_true_scores, y_pred_scores)
        rmse = mean_squared_error(y_true_scores, y_pred_scores) ** 0.5

        return {
            "mae": float(mae),
            "rmse": float(rmse)
        }

    # THRESHOLD CALIBRATION
    def tune_thresholds(self, y_true, y_pred_probs, threshold_range=None):
        """
        Find best threshold per label using F1 score
        """
        if threshold_range is None:
            threshold_range = [i / 100 for i in range(30, 91, 5)]

        optimal_thresholds = {}

        for label_idx, label in enumerate(self.labels):
            best_f1 = 0
            best_threshold = 0.5

            for t in threshold_range:
                preds = (y_pred_probs[:, label_idx] > t).astype(int)

                f1 = f1_score(
                    y_true[:, label_idx],
                    preds,
                    zero_division=0
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

            optimal_thresholds[label] = best_threshold

        return optimal_thresholds

    def save_thresholds(self, thresholds, path="thresholds.json"):
        """
        Save thresholds to json
        """
        with open(path, "w") as f:
            json.dump(thresholds, f, indent=2)

    def load_thresholds(self, path="thresholds.json", default=0.5):
        """
        Load thresholds safely
        """
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {label: default for label in self.labels}

    # ERROR ANALYSIS
    def collect_error_records(self, y_true, y_pred, y_pred_probs):
        """
        Collect false positives and false negatives
        """
        error_records = []

        for i in range(len(y_true)):
            for j, label in enumerate(self.labels):
                if y_true[i][j] != y_pred[i][j]:
                    error_records.append({
                        "issue": label,
                        "true": int(y_true[i][j]),
                        "pred": int(y_pred[i][j]),
                        "probability": float(y_pred_probs[i][j])
                    })

        return error_records



LABELS = [
    "repetition",
    "unclear_intro",
    "missing_confirmation",
    "long_agent_monologue",
    "compliance_risk"
]

evaluator = EvaluationUtils(LABELS)

metrics = evaluator.compute_classification_metrics(y_true, y_pred)
thresholds = evaluator.tune_thresholds(y_true, y_pred_probs)
evaluator.save_thresholds(thresholds)

errors = evaluator.collect_error_records(y_true, y_pred, y_pred_probs)
