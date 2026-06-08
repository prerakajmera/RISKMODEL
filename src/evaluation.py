

import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

logger = logging.getLogger(__name__)

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")

PALETTE = ["#0d6efd", "#198754", "#dc3545", "#ffc107", "#6f42c1", "#20c997"]


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "logloss":   round(log_loss(y_test, y_proba), 4),
        }
        logger.info("Metrics for %s: %s", type(model).__name__, metrics)
        return metrics

    except Exception as e:
        logger.error("Error evaluating model: %s", e, exc_info=True)
        raise


def evaluate_all_models(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    try:
        rows = {}
        for name, model in models_dict.items():
            rows[name] = evaluate_model(model, X_test, y_test)
        results = pd.DataFrame(rows).T
        results.index.name = "model"
        logger.info("Model comparison:\n%s", results.to_string())
        return results

    except Exception as e:
        logger.error("Error in evaluate_all_models: %s", e, exc_info=True)
        raise


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> plt.Figure:
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s", save_path)
        return fig

    except Exception as e:
        logger.error("Error plotting confusion matrix: %s", e, exc_info=True)
        raise


def plot_roc_curves(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, (name, model) in enumerate(models_dict.items()):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            ax.plot(
                fpr, tpr,
                label=f"{name} (AUC = {auc_val:.3f})",
                color=PALETTE[idx % len(PALETTE)],
                linewidth=2,
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("ROC curves saved to %s", save_path)
        return fig

    except Exception as e:
        logger.error("Error plotting ROC curves: %s", e, exc_info=True)
        raise


def plot_precision_recall_curves(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, (name, model) in enumerate(models_dict.items()):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            ax.plot(
                recall, precision,
                label=f"{name} (AP = {ap:.3f})",
                color=PALETTE[idx % len(PALETTE)],
                linewidth=2,
            )

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("PR curves saved to %s", save_path)
        return fig

    except Exception as e:
        logger.error("Error plotting PR curves: %s", e, exc_info=True)
        raise


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    save_path: str = None,
) -> plt.Figure:
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model has no feature_importances_ or coef_.")
            return None

        series = pd.Series(importances, index=feature_names).nlargest(top_n)

        fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
        series.sort_values().plot.barh(ax=ax, color="#0d6efd", edgecolor="white")
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Feature importance plot saved to %s", save_path)
        return fig

    except Exception as e:
        logger.error("Error plotting feature importance: %s", e, exc_info=True)
        raise


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Correlation heatmap saved to %s", save_path)
        return fig

    except Exception as e:
        logger.error("Error plotting correlation heatmap: %s", e, exc_info=True)
        raise


def compare_models_chart(
    results_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    try:
        plot_df = results_df.drop(columns=["logloss"], errors="ignore")

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_df.plot.bar(ax=ax, colormap="viridis", edgecolor="white", width=0.75)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison — Evaluation Metrics", fontsize=14, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Model comparison chart saved to %s", save_path)
        return fig

    except Exception as e:
        logger.error("Error plotting model comparison: %s", e, exc_info=True)
        raise
