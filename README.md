# Credit Risk Modeling

A machine learning project that predicts whether a loan borrower will default or not. Built using Python, scikit-learn, and Streamlit.

**Live Demo:** [Open App](https://riskmodel.streamlit.app/)

## About

This project uses the **German Credit Dataset** (1000 samples, 20 features) from the UCI repository to classify borrowers as good or bad credit risks. I trained multiple models and built a web dashboard to visualize predictions.

**Dataset link:** https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

## Models Used

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- WoE Credit Scorecard

## Project Structure

```
credit-risk-model/
├── data/               # dataset (auto-downloaded)
├── models/             # saved models and plots
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── training.py
│   ├── evaluation.py
│   └── predict.py
├── app.py              # streamlit dashboard
├── main.py             # training pipeline
└── requirements.txt
```

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# train all models
python main.py

# launch the dashboard
streamlit run app.py
```

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 72.5% | 0.789 |
| Random Forest | 78.0% | 0.785 |
| XGBoost | 75.0% | 0.757 |
| LightGBM | 73.5% | 0.768 |

## Features

- Data cleaning and preprocessing with SMOTE for class balancing
- 9 engineered features (credit per month, age ratios, flags etc.)
- Hyperparameter tuning with Optuna (optional: `python main.py --tune`)
- SHAP explainability for individual predictions
- Interactive Streamlit dashboard with risk scoring

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, Plotly, Matplotlib, Seaborn

## Screenshots

Run `streamlit run app.py` and open http://localhost:8502 to see the dashboard.

---

Made as a course project for learning ML and credit risk analysis.
