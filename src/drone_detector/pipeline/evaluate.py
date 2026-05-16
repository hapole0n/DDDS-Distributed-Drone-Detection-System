"""Evaluation: confusion matrix, classification report, ROC curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # headless safe for CI / Streamlit
import matplotlib.pyplot as plt  # noqa: E402

from drone_detector.models.baseline import ModelBundle


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(labels), 4 + 0.4 * len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="viridis", colorbar=False)
    ax.set_title("Confusion matrix (held-out test split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_roc(
    y_true: np.ndarray, y_score: np.ndarray, label_name: str, out_path: Path
) -> tuple[Path, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — positive class: {label_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path, float(auc)


def write_report(
    bundle: ModelBundle,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
    positive_class: str = "shahed",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = bundle.predict(X_test)
    proba = bundle.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(bundle.labels))))
    cm_png = plot_confusion(cm, bundle.labels, out_dir / "confusion_matrix.png")

    report_txt = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(bundle.labels))),
        target_names=bundle.labels,
        digits=3,
        zero_division=0,
    )

    pos_id = bundle.positive_class_id(positive_class)
    roc_lines: list[str] = []
    if pos_id is not None and len(np.unique(y_test)) > 1:
        y_true_bin = (y_test == pos_id).astype(int)
        y_score = proba[:, pos_id]
        roc_png, auc = plot_roc(
            y_true_bin, y_score, positive_class, out_dir / "roc.png"
        )
        roc_lines = [
            "",
            f"## ROC (positive = `{positive_class}`)",
            f"AUC = **{auc:.3f}**",
            "",
            f"![ROC]({roc_png.name})",
        ]

    md_path = out_dir / "evaluation.md"
    md_path.write_text(
        "\n".join(
            [
                "# Evaluation report",
                "",
                "## Dataset",
                f"- classes: `{bundle.labels}`",
                f"- feature_dim: `{bundle.feature_dim}`",
                f"- meta: `{bundle.meta}`",
                "",
                "## Confusion matrix",
                f"![Confusion matrix]({cm_png.name})",
                "",
                "## Per-class metrics",
                "```",
                report_txt,
                "```",
                *roc_lines,
            ]
        ),
        encoding="utf-8",
    )
    return md_path
