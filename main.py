
import os
import sys
import io
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.preprocessing import (
    load_german_credit_data,
    clean_data,
    encode_categoricals,
    scale_features,
    split_data,
    apply_smote,
    CATEGORICAL_COLUMNS,
)
from src.features import engineer_features
from src.training import (
    train_all_models,
    tune_lightgbm,
    build_scorecard,
    save_model,
)
from src.evaluation import (
    evaluate_all_models,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_feature_importance,
    plot_correlation_heatmap,
    compare_models_chart,
)
from src.predict import generate_sample_inputs, predict_default_probability, assign_risk_category, calculate_credit_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def ensure_directories():
    for d in [DATA_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)
        logger.info("Directory ready: %s", d)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Credit Risk Modeling - End-to-End Pipeline",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable Optuna hyperparameter tuning for LightGBM.",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of Optuna trials (default: 30).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("   CREDIT RISK MODELING PIPELINE")
    print("=" * 70 + "\n")

    ensure_directories()

    logger.info("STEP 1 - Loading German Credit Dataset")
    cache_path = os.path.join(DATA_DIR, "german_credit.csv")
    df_raw = load_german_credit_data(cache_path=cache_path)
    print(f"   [+] Dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"   [+] Target distribution:\n{df_raw['target'].value_counts().to_string()}\n")

    logger.info("STEP 2 - Cleaning data")
    df_clean = clean_data(df_raw)
    print(f"   [+] Data cleaned: {df_clean.shape[0]} rows remain\n")

    logger.info("STEP 3 - Engineering features")
    df_feat = engineer_features(df_clean)
    new_cols = [c for c in df_feat.columns if c not in df_clean.columns]
    print(f"   [+] {len(new_cols)} new features created: {new_cols}\n")

    logger.info("STEP 4 - Generating correlation heatmap")
    plot_correlation_heatmap(
        df_feat,
        save_path=os.path.join(MODEL_DIR, "correlation_heatmap.png"),
    )
    print("   [+] Correlation heatmap saved\n")

    logger.info("STEP 5 - Encoding categorical variables")
    df_encoded, encoders = encode_categoricals(df_feat, fit=True)
    print(f"   [+] {len(encoders)} categorical columns encoded\n")

    logger.info("STEP 6 - Splitting data")
    target_col = "target"
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col].values
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = split_data(X.values, y)
    print(f"   [+] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

    logger.info("STEP 7 - Applying SMOTE oversampling")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    print(f"   [+] After SMOTE: {X_train_sm.shape[0]} training samples\n")

    logger.info("STEP 8 - Scaling features")
    X_train_scaled, scaler = scale_features(X_train_sm, fit=True)
    X_test_scaled, _ = scale_features(X_test, fit=False, scaler=scaler)
    print("   [+] Features scaled with StandardScaler\n")

    logger.info("STEP 9 - Training models")
    models = train_all_models(X_train_scaled, y_train_sm)
    print(f"   [+] {len(models)} models trained\n")

    if args.tune:
        logger.info("STEP 9b - Optuna hyperparameter tuning for LightGBM")
        tuned_lgbm = tune_lightgbm(X_train_scaled, y_train_sm, n_trials=args.trials)
        models["LightGBM_Tuned"] = tuned_lgbm
        print(f"   [+] Tuned LightGBM added ({args.trials} trials)\n")

    logger.info("STEP 10 - Building WoE Credit Scorecard")
    try:
        sc_model, woe_dict, sc_points = build_scorecard(
            X_train, y_train, X_test, y_test,
            feature_names=feature_names,
        )
        models["Scorecard"] = sc_model
        save_model(woe_dict, os.path.join(MODEL_DIR, "woe_dict.pkl"))
        save_model(sc_points, os.path.join(MODEL_DIR, "scorecard_points.pkl"))
        print("   [+] Credit Scorecard built and saved\n")
    except Exception as e:
        logger.warning("Scorecard build failed (non-critical): %s", e)
        print(f"   [!] Scorecard skipped: {e}\n")

    logger.info("STEP 11 - Evaluating models")
    results = evaluate_all_models(models, X_test_scaled, y_test)
    results_path = os.path.join(MODEL_DIR, "results.csv")
    results.to_csv(results_path)
    print("   [+] Evaluation complete\n")
    print("=" * 70)
    print("   MODEL COMPARISON")
    print("=" * 70)
    print(results.to_string())
    print()

    logger.info("STEP 12 - Generating evaluation plots")

    plot_roc_curves(
        models, X_test_scaled, y_test,
        save_path=os.path.join(MODEL_DIR, "roc_curves.png"),
    )
    print("   [+] ROC curves saved")

    plot_precision_recall_curves(
        models, X_test_scaled, y_test,
        save_path=os.path.join(MODEL_DIR, "pr_curves.png"),
    )
    print("   [+] Precision-Recall curves saved")

    best_model_name = results["roc_auc"].idxmax()
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_scaled)
    plot_confusion_matrix(
        y_test, y_pred_best,
        title=f"Confusion Matrix - {best_model_name}",
        save_path=os.path.join(MODEL_DIR, "confusion_matrix.png"),
    )
    print(f"   [+] Confusion matrix saved (best model: {best_model_name})")

    if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
        plot_feature_importance(
            best_model, feature_names,
            save_path=os.path.join(MODEL_DIR, "feature_importance.png"),
        )
        print("   [+] Feature importance plot saved")

    compare_models_chart(
        results,
        save_path=os.path.join(MODEL_DIR, "model_comparison.png"),
    )
    print("   [+] Model comparison chart saved\n")

    logger.info("STEP 13 - Saving artefacts")

    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        save_model(model, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))

    save_model(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    save_model(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    save_model(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print("   [+] All models and artefacts saved to models/\n")

    logger.info("STEP 14 - Running sample predictions")
    print("=" * 70)
    print("   SAMPLE PREDICTIONS")
    print("=" * 70)

    samples = generate_sample_inputs(n=3)
    for idx, row in samples.iterrows():
        row_df = pd.DataFrame([row])
        row_feat = engineer_features(row_df)
        row_enc, _ = encode_categoricals(row_feat, fit=False, encoders=encoders)
        for col in feature_names:
            if col not in row_enc.columns:
                row_enc[col] = 0
        row_enc = row_enc[feature_names]
        row_scaled, _ = scale_features(row_enc.values, fit=False, scaler=scaler)

        prob = predict_default_probability(best_model, row_scaled)
        risk = assign_risk_category(prob)
        score = calculate_credit_score(prob)

        print(f"\n   Sample {idx + 1}:")
        print(f"     Age: {row['age']}, Amount: {row['credit_amount']}, Duration: {row['duration']}mo")
        print(f"     Default Probability: {prob:.1%}")
        print(f"     Risk Category:       {risk}")
        print(f"     Credit Score:         {score}")

    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n   Best model:    {best_model_name} (ROC-AUC: {results.loc[best_model_name, 'roc_auc']:.4f})")
    print(f"   Artefacts in:  {MODEL_DIR}")
    print(f"   Launch app:    streamlit run app.py\n")


if __name__ == "__main__":
    main()
