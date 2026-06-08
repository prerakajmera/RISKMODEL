
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        required_cols = {"credit_amount", "duration", "age", "installment_rate"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns for feature engineering: {missing}")

        df["credit_per_month"] = df["credit_amount"] / df["duration"].replace(0, np.nan)
        df["credit_per_month"].fillna(0, inplace=True)
        logger.debug("Created feature: credit_per_month")

        df["age_credit_ratio"] = df["age"] / (df["credit_amount"] + 1)
        logger.debug("Created feature: age_credit_ratio")

        df["credit_age_years"] = df["age"] - 18
        logger.debug("Created feature: credit_age_years")

        credit_median = df["credit_amount"].median()
        df["high_credit_flag"] = (df["credit_amount"] > credit_median).astype(int)
        logger.debug("Created feature: high_credit_flag (median=%s)", credit_median)

        df["installment_credit_ratio"] = (
            df["installment_rate"] / (df["credit_amount"] + 1) * 10000
        )
        logger.debug("Created feature: installment_credit_ratio")

        df["young_borrower"] = (df["age"] < 30).astype(int)
        logger.debug("Created feature: young_borrower")

        df["senior_borrower"] = (df["age"] > 55).astype(int)
        logger.debug("Created feature: senior_borrower")

        df["long_duration"] = (df["duration"] > 24).astype(int)
        logger.debug("Created feature: long_duration")

        df["credit_duration_interaction"] = df["credit_amount"] * df["duration"]
        logger.debug("Created feature: credit_duration_interaction")

        logger.info(
            "Engineered 9 features. DataFrame shape: %s → %s columns total.",
            df.shape, df.shape[1],
        )
        return df

    except Exception as e:
        logger.error("Error during feature engineering: %s", e, exc_info=True)
        raise
