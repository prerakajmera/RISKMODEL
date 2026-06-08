
import logging
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)


def get_models() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, random_state=42, verbose=-1,
        ),
    }
    logger.info("Initialised %d model templates.", len(models))
    return models


def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    try:
        model_name = type(model).__name__
        logger.info("Training %s …", model_name)
        model.fit(X_train, y_train)
        logger.info("Training complete for %s.", model_name)
        return model
    except Exception as e:
        logger.error("Error training %s: %s", type(model).__name__, e, exc_info=True)
        raise


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    try:
        models = get_models()
        fitted = {}
        for name, model in models.items():
            fitted[name] = train_model(model, X_train, y_train)
        logger.info("All %d base models trained successfully.", len(fitted))
        return fitted
    except Exception as e:
        logger.error("Error in train_all_models: %s", e, exc_info=True)
        raise


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
):
    try:
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info("Starting Optuna LightGBM tuning (%d trials) …", n_trials)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
                "verbose": -1,
            }
            model = LGBMClassifier(**params)
            scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["verbose"] = -1
        logger.info("Best Optuna ROC-AUC: %.4f", study.best_value)
        logger.info("Best params: %s", best_params)

        best_model = LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train)
        logger.info("Tuned LightGBM re-trained on full training set.")

        return best_model

    except ImportError:
        logger.error("Optuna is not installed. Install via: pip install optuna")
        raise
    except Exception as e:
        logger.error("Error tuning LightGBM: %s", e, exc_info=True)
        raise


def calculate_woe(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> tuple:
    try:
        eps = 1e-6
        total_good = (df[target] == 0).sum()
        total_bad = (df[target] == 1).sum()

        grouped = df.groupby(feature)[target].agg(["count", "sum"])
        grouped.columns = ["total", "bad"]
        grouped["good"] = grouped["total"] - grouped["bad"]

        grouped["dist_good"] = grouped["good"] / (total_good + eps)
        grouped["dist_bad"] = grouped["bad"] / (total_bad + eps)

        grouped["woe"] = np.log((grouped["dist_good"] + eps) / (grouped["dist_bad"] + eps))
        grouped["iv_component"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

        woe_mapping = grouped["woe"].to_dict()
        iv_value = grouped["iv_component"].sum()

        logger.debug("WoE for '%s': IV=%.4f", feature, iv_value)
        return woe_mapping, iv_value

    except Exception as e:
        logger.error("Error calculating WoE for '%s': %s", feature, e, exc_info=True)
        raise


def build_scorecard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list = None,
) -> tuple:
    try:
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_train["target"] = np.array(y_train).ravel()

        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test["target"] = np.array(y_test).ravel()

        woe_dict = {}
        iv_values = {}

        for col in feature_names:
            n_unique = df_train[col].nunique()
            if n_unique > 10:
                bin_col = col + "_bin"
                df_train[bin_col] = pd.qcut(
                    df_train[col], q=4, duplicates="drop",
                ).astype(str)
                df_test[bin_col] = pd.qcut(
                    df_test[col], q=4, duplicates="drop",
                ).astype(str)
                woe_map, iv = calculate_woe(df_train, bin_col, "target")
                woe_dict[col] = {"binned": True, "woe_map": woe_map, "bin_col": bin_col}
            else:
                str_col = col + "_str"
                df_train[str_col] = df_train[col].astype(str)
                df_test[str_col] = df_test[col].astype(str)
                woe_map, iv = calculate_woe(df_train, str_col, "target")
                woe_dict[col] = {"binned": False, "woe_map": woe_map, "str_col": str_col}
            iv_values[col] = iv

        logger.info("Top-5 IV features: %s",
                     sorted(iv_values.items(), key=lambda x: x[1], reverse=True)[:5])

        def woe_transform(df_src, woe_info):
            transformed = pd.DataFrame()
            for col, info in woe_info.items():
                lookup_col = info.get("bin_col") or info.get("str_col")
                if lookup_col and lookup_col in df_src.columns:
                    transformed[col] = df_src[lookup_col].map(info["woe_map"]).fillna(0)
                else:
                    transformed[col] = 0
            return transformed

        X_train_woe = woe_transform(df_train, woe_dict)
        X_test_woe = woe_transform(df_test, woe_dict)

        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_woe, y_train)
        logger.info("Scorecard LogisticRegression fitted on WoE features.")

        scorecard_points = {
            "base_score": 580,
            "pdo": 40,
            "min_score": 300,
            "max_score": 850,
            "formula": "score = 850 - P(default) * 550, clamped [300, 850]",
            "coefficients": dict(zip(feature_names, lr_model.coef_[0].tolist())),
            "intercept": float(lr_model.intercept_[0]),
            "iv_values": iv_values,
        }

        logger.info("Scorecard built successfully (score range 300-850).")
        return lr_model, woe_dict, scorecard_points

    except Exception as e:
        logger.error("Error building scorecard: %s", e, exc_info=True)
        raise


def save_model(model, filepath: str) -> None:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logger.info("Model saved to %s", filepath)
    except Exception as e:
        logger.error("Error saving model to %s: %s", filepath, e, exc_info=True)
        raise


def load_model(filepath: str):
    try:
        model = joblib.load(filepath)
        logger.info("Model loaded from %s", filepath)
        return model
    except Exception as e:
        logger.error("Error loading model from %s: %s", filepath, e, exc_info=True)
        raise
