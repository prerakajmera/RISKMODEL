
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def predict_default_probability(model, input_data) -> float:
    try:
        arr = np.array(input_data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(arr)[:, 1][0]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(arr)[0]
            prob = 1 / (1 + np.exp(-raw))
        else:
            prob = float(model.predict(arr)[0])

        logger.info("Predicted default probability: %.4f", prob)
        return float(prob)

    except Exception as e:
        logger.error("Error predicting default probability: %s", e, exc_info=True)
        raise


def assign_risk_category(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability <= 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def calculate_credit_score(probability: float) -> int:
    score = 850 - probability * 550
    return int(np.clip(score, 300, 850))


def get_risk_color(category: str) -> str:
    mapping = {
        "Low Risk":    "#00c853",
        "Medium Risk": "#ff9100",
        "High Risk":   "#ff1744",
    }
    return mapping.get(category, "#ffffff")


def generate_sample_inputs(n: int = 5) -> pd.DataFrame:
    try:
        rng = np.random.RandomState(42)

        data = {
            "checking_account":       rng.choice(["A11", "A12", "A13", "A14"], n),
            "duration":               rng.randint(6, 60, n),
            "credit_history":         rng.choice(["A30", "A31", "A32", "A33", "A34"], n),
            "purpose":                rng.choice(["A40", "A41", "A42", "A43", "A44",
                                                  "A45", "A46", "A48", "A49", "A410"], n),
            "credit_amount":          rng.randint(500, 18000, n),
            "savings_account":        rng.choice(["A61", "A62", "A63", "A64", "A65"], n),
            "employment_since":       rng.choice(["A71", "A72", "A73", "A74", "A75"], n),
            "installment_rate":       rng.randint(1, 5, n),
            "personal_status_sex":    rng.choice(["A91", "A92", "A93", "A94", "A95"], n),
            "other_debtors":          rng.choice(["A101", "A102", "A103"], n),
            "residence_since":        rng.randint(1, 5, n),
            "property":              rng.choice(["A121", "A122", "A123", "A124"], n),
            "age":                    rng.randint(19, 75, n),
            "other_installment_plans": rng.choice(["A141", "A142", "A143"], n),
            "housing":               rng.choice(["A151", "A152", "A153"], n),
            "existing_credits":       rng.randint(1, 5, n),
            "job":                    rng.choice(["A171", "A172", "A173", "A174"], n),
            "num_dependents":         rng.randint(1, 3, n),
            "telephone":             rng.choice(["A191", "A192"], n),
            "foreign_worker":        rng.choice(["A201", "A202"], n),
        }

        df = pd.DataFrame(data)
        logger.info("Generated %d synthetic sample inputs.", n)
        return df

    except Exception as e:
        logger.error("Error generating sample inputs: %s", e, exc_info=True)
        raise


def get_shap_explanation(model, X_data, feature_names=None):
    try:
        import shap

        arr = np.array(X_data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if feature_names is not None:
            arr = pd.DataFrame(arr, columns=feature_names)

        try:
            explainer = shap.Explainer(model, arr)
        except Exception:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                    shap.sample(arr, min(50, len(arr))),
                )

        shap_values = explainer(arr)
        logger.info("SHAP values computed successfully.")
        return shap_values

    except ImportError:
        logger.warning("SHAP is not installed. Install via: pip install shap")
        return None
    except Exception as e:
        logger.error("Error computing SHAP values: %s", e, exc_info=True)
        return None
