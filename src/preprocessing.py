
import os
import logging
import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

COLUMN_NAMES = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents",
    "telephone", "foreign_worker", "target",
]

CATEGORICAL_COLUMNS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
]

NUMERICAL_COLUMNS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

DEFAULT_CACHE_PATH = os.path.join("data", "german_credit.csv")


def load_german_credit_data(cache_path: str = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    try:
        if os.path.exists(cache_path):
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_csv(cache_path)
            logger.info("Loaded %d rows, %d columns from cache.", *df.shape)
            return df

        logger.info("Downloading German Credit Dataset from UCI …")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        local_tmp = cache_path + ".tmp"
        urllib.request.urlretrieve(DATA_URL, local_tmp)
        logger.info("Download complete.")

        df = pd.read_csv(local_tmp, sep=r"\s+", header=None, names=COLUMN_NAMES)

        df["target"] = df["target"].map({1: 0, 2: 1})
        logger.info("Target remapped: 1→0 (Good), 2→1 (Bad).")

        df.to_csv(cache_path, index=False)
        logger.info("Data cached to %s (%d rows, %d cols).", cache_path, *df.shape)

        if os.path.exists(local_tmp):
            os.remove(local_tmp)

        return df

    except Exception as e:
        logger.error("Failed to load German Credit data: %s", e, exc_info=True)
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        initial_rows = len(df)

        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            logger.warning("Found %d missing values – filling.", missing_total)
            for col in NUMERICAL_COLUMNS:
                if col in df.columns and df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info("Filled NaN in '%s' with median=%s.", col, median_val)
            for col in CATEGORICAL_COLUMNS:
                if col in df.columns and df[col].isnull().any():
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    logger.info("Filled NaN in '%s' with mode=%s.", col, mode_val)
        else:
            logger.info("No missing values detected.")

        for col in NUMERICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str)
        logger.info("Data types verified/fixed.")

        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df.drop_duplicates(inplace=True)
            logger.warning(
                "Removed %d duplicate rows (%d → %d).",
                dup_count, initial_rows, len(df),
            )
        else:
            logger.info("No duplicate rows found.")

        return df

    except Exception as e:
        logger.error("Error during data cleaning: %s", e, exc_info=True)
        raise


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict = None,
) -> tuple:
    try:
        df = df.copy()
        if encoders is None:
            encoders = {}

        cats_in_df = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

        if fit:
            for col in cats_in_df:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                logger.debug("Fitted encoder for '%s' (%d classes).", col, len(le.classes_))
            logger.info("Fit-encoded %d categorical columns.", len(cats_in_df))
        else:
            if not encoders:
                raise ValueError("encoders dict must be provided when fit=False.")
            for col in cats_in_df:
                if col not in encoders:
                    logger.warning("No encoder for '%s'; skipping.", col)
                    continue
                le = encoders[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x, _k=known, _le=le: (
                        _le.transform([x])[0] if x in _k else -1
                    )
                )
            logger.info("Transform-encoded %d categorical columns.", len(cats_in_df))

        return df, encoders

    except Exception as e:
        logger.error("Error encoding categoricals: %s", e, exc_info=True)
        raise


def scale_features(
    X: np.ndarray,
    fit: bool = True,
    scaler: StandardScaler = None,
) -> tuple:
    try:
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info("Fitted StandardScaler on %d features.", X.shape[1] if hasattr(X, 'shape') else 'unknown')
        else:
            if scaler is None:
                raise ValueError("scaler must be provided when fit=False.")
            X_scaled = scaler.transform(X)
            logger.info("Transformed features using pre-fitted scaler.")

        return X_scaled, scaler

    except Exception as e:
        logger.error("Error scaling features: %s", e, exc_info=True)
        raise


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
        logger.info(
            "Data split: train=%d, test=%d (test_size=%.2f, stratified).",
            len(X_train), len(X_test), test_size,
        )
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error("Error splitting data: %s", e, exc_info=True)
        raise


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple:
    try:
        logger.info(
            "Class distribution before SMOTE: %s",
            dict(zip(*np.unique(y_train, return_counts=True))),
        )
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            "Class distribution after SMOTE:  %s",
            dict(zip(*np.unique(y_res, return_counts=True))),
        )
        return X_res, y_res

    except Exception as e:
        logger.error("Error applying SMOTE: %s", e, exc_info=True)
        raise
