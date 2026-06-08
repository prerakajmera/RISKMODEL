import streamlit as st

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import os
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

_SRC_AVAILABLE = False
_SRC_ERROR = ""
try:
    from src.predict import (
        predict_default_probability,
        assign_risk_category,
        calculate_credit_score,
        get_risk_color,
        get_shap_explanation,
    )
    from src.features import engineer_features
    from src.preprocessing import encode_categoricals, scale_features
    _SRC_AVAILABLE = True
except Exception as _e:
    _SRC_ERROR = str(_e)
    import traceback
    _SRC_ERROR = traceback.format_exc()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  section[data-testid="stSidebar"] { display: none !important; }
  [data-testid="stSidebarCollapsedControl"] { display: none !important; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  [data-testid="stHeader"] {
    background: rgba(10,10,26,0.95) !important;
    backdrop-filter: blur(12px);
  }

  .stApp {
    background: linear-gradient(160deg, #06060f 0%, #0d1117 30%, #0a0f1a 60%, #0d0d1f 100%);
    color: #e0e0e8;
  }

  .card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
  }
  .card:hover {
    border-color: rgba(56,189,248,0.2);
    box-shadow: 0 8px 32px rgba(56,189,248,0.06);
  }

  .kpi {
    text-align: center;
    padding: 28px 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .kpi-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #64748b;
    margin-bottom: 12px;
  }
  .kpi-value {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 8px;
  }

  .pill {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 99px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
  }
  .pill-green  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
  .pill-amber  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
  .pill-red    { background: rgba(239,68,68,0.15);  color: #ef4444;  border: 1px solid rgba(239,68,68,0.3);  }

  .sec-title {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #38bdf8;
    margin: 20px 0 12px;
  }

  .hero { text-align: center; padding: 40px 0 10px; }
  .hero h1 {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #38bdf8, #34d399, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .hero p { color: #64748b; font-size: 0.95rem; margin-top: 6px; letter-spacing: 0.5px; }

  .divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
    margin: 24px 0;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.85rem;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.1) !important;
    color: #38bdf8 !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px 32px !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.3) !important;
  }

  .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
  }

  .clean-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .clean-table th {
    text-align: left; padding: 12px 14px; color: #38bdf8;
    font-weight: 700; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 1.5px; border-bottom: 1px solid rgba(56,189,248,0.15);
  }
  .clean-table td {
    padding: 10px 14px; color: #cbd5e1;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .clean-table tr:hover td { background: rgba(56,189,248,0.03); }

  .disclaimer {
    text-align: center; color: #475569; font-size: 0.72rem;
    padding: 24px; margin-top: 40px;
    border-top: 1px solid rgba(255,255,255,0.04);
  }
</style>
""", unsafe_allow_html=True)

CAT = {
    "checking_account": {
        "< 0 DM": "A11", "0 - 200 DM": "A12",
        ">= 200 DM": "A13", "No checking account": "A14",
    },
    "credit_history": {
        "No credits / all paid": "A30", "All paid at this bank": "A31",
        "Existing credits paid": "A32", "Delay in past": "A33",
        "Critical account": "A34",
    },
    "purpose": {
        "Car (new)": "A40", "Car (used)": "A41", "Furniture": "A42",
        "Radio / TV": "A43", "Appliances": "A44", "Repairs": "A45",
        "Education": "A46", "Retraining": "A48", "Business": "A49", "Other": "A410",
    },
    "savings_account": {
        "< 100 DM": "A61", "100-500 DM": "A62", "500-1000 DM": "A63",
        ">= 1000 DM": "A64", "Unknown / none": "A65",
    },
    "employment_since": {
        "Unemployed": "A71", "< 1 year": "A72", "1-4 years": "A73",
        "4-7 years": "A74", ">= 7 years": "A75",
    },
    "personal_status_sex": {
        "Male - divorced": "A91", "Female - div/married": "A92",
        "Male - single": "A93", "Male - married": "A94", "Female - single": "A95",
    },
    "other_debtors": {"None": "A101", "Co-applicant": "A102", "Guarantor": "A103"},
    "property": {
        "Real estate": "A121", "Savings/insurance": "A122",
        "Car / other": "A123", "Unknown/none": "A124",
    },
    "other_installment_plans": {"Bank": "A141", "Stores": "A142", "None": "A143"},
    "housing": {"Rent": "A151", "Own": "A152", "For free": "A153"},
    "job": {
        "Unskilled non-resident": "A171", "Unskilled resident": "A172",
        "Skilled": "A173", "Highly skilled / mgmt": "A174",
    },
    "telephone": {"No": "A191", "Yes": "A192"},
    "foreign_worker": {"Yes": "A201", "No": "A202"},
}

SAMPLE = {
    "checking_account": "0 - 200 DM", "credit_history": "Existing credits paid",
    "purpose": "Radio / TV", "savings_account": "100-500 DM",
    "employment_since": "4-7 years", "personal_status_sex": "Male - single",
    "other_debtors": "None", "property": "Real estate",
    "other_installment_plans": "None", "housing": "Own",
    "job": "Skilled", "telephone": "Yes", "foreign_worker": "Yes",
    "duration": 24, "credit_amount": 5000, "installment_rate": 2,
    "residence_since": 3, "age": 35, "existing_credits": 1, "num_dependents": 1,
}

MODEL_FILES = {
    "Logistic Regression": "logisticregression.pkl",
    "Random Forest": "randomforest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
}

@st.cache_resource(show_spinner=False)
def load_models():
    artefacts, missing = {}, []
    for label, fname in MODEL_FILES.items():
        p = MODELS_DIR / fname
        if p.exists():
            artefacts[label] = joblib.load(p)
        else:
            missing.append(str(p))
    for name in ("scaler", "encoders", "feature_names"):
        p = MODELS_DIR / f"{name}.pkl"
        if p.exists():
            artefacts[name] = joblib.load(p)
        else:
            missing.append(str(p))
    return artefacts, missing

@st.cache_data(show_spinner=False)
def load_results():
    p = MODELS_DIR / "results.csv"
    return pd.read_csv(p) if p.exists() else None

def gauge_chart(score):
    if score >= 740: bar_color = "#10b981"
    elif score >= 670: bar_color = "#fbbf24"
    elif score >= 580: bar_color = "#f97316"
    else: bar_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": "#e0e0e8", "family": "Inter"}},
        gauge={
            "axis": {"range": [300, 850], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 0,
            "steps": [
                {"range": [300, 580], "color": "rgba(239,68,68,0.08)"},
                {"range": [580, 670], "color": "rgba(249,115,22,0.08)"},
                {"range": [670, 740], "color": "rgba(251,191,36,0.08)"},
                {"range": [740, 850], "color": "rgba(16,185,129,0.08)"},
            ],
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=0, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig

def risk_pill(category):
    cls = {"Low Risk": "pill-green", "Medium Risk": "pill-amber", "High Risk": "pill-red"}
    icon = {"Low Risk": "&#10003;", "Medium Risk": "&#9888;", "High Risk": "&#10007;"}
    return f'<span class="pill {cls.get(category,"pill-amber")}">{icon.get(category,"")} {category}</span>'

def results_table(df):
    if df is None: return "<p style='color:#475569'>No results available.</p>"
    hdr = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, r in df.iterrows():
        cells = "".join(f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>" for v in r)
        rows += f"<tr>{cells}</tr>"
    return f'<table class="clean-table"><thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table>'

st.markdown("""
<div class="hero">
    <h1>Credit Risk Intelligence</h1>
    <p>AI-Powered Default Prediction System</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

tab_predict, tab_compare = st.tabs(["Predict Risk", "Model Comparison"])

with tab_predict:
    if not _SRC_AVAILABLE:
        st.error("Source modules failed to load. See error below:")
        st.code(_SRC_ERROR)
        st.stop()

    artefacts, missing = load_models()
    if missing:
        st.error("Model files missing. Run `python main.py` to train models first.\n\n" +
                 "\n".join(f"- `{m}`" for m in missing))
        st.stop()

    col_sample, col_model, _ = st.columns([1, 1, 3])
    with col_sample:
        if st.button("Load Sample Data"):
            for k, v in SAMPLE.items():
                st.session_state[f"inp_{k}"] = v
            st.rerun()
    with col_model:
        model_choice = st.selectbox("Model", list(MODEL_FILES.keys()), index=3, label_visibility="collapsed")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Borrower Profile</div>', unsafe_allow_html=True)

    def _sel(key, label, col_container):
        opts = list(CAT[key].keys())
        default = st.session_state.get(f"inp_{key}", opts[0])
        idx = opts.index(default) if default in opts else 0
        return col_container.selectbox(label, opts, index=idx, key=f"inp_{key}")

    c1, c2, c3, c4 = st.columns(4)
    inp_age = c1.slider("Age", 18, 75, int(st.session_state.get("inp_age", 35)), key="inp_age")
    inp_pss = _sel("personal_status_sex", "Status / Sex", c2)
    inp_fw = _sel("foreign_worker", "Foreign Worker", c3)
    inp_tel = _sel("telephone", "Telephone", c4)

    c1, c2, c3, c4 = st.columns(4)
    inp_ca = _sel("checking_account", "Checking Account", c1)
    inp_sa = _sel("savings_account", "Savings Account", c2)
    inp_amt = c3.number_input("Credit Amount (DM)", 250, 20000,
                              int(st.session_state.get("inp_credit_amount", 5000)),
                              step=250, key="inp_credit_amount")
    inp_dur = c4.slider("Duration (months)", 1, 72,
                        int(st.session_state.get("inp_duration", 24)), key="inp_duration")

    c1, c2, c3, c4 = st.columns(4)
    inp_ch = _sel("credit_history", "Credit History", c1)
    inp_pur = _sel("purpose", "Loan Purpose", c2)
    inp_ir = c3.slider("Installment Rate (%)", 1, 4,
                       int(st.session_state.get("inp_installment_rate", 2)), key="inp_installment_rate")
    inp_ec = c4.number_input("Existing Credits", 1, 4,
                             int(st.session_state.get("inp_existing_credits", 1)), key="inp_existing_credits")

    c1, c2, c3, c4 = st.columns(4)
    inp_emp = _sel("employment_since", "Employment", c1)
    inp_prop = _sel("property", "Property", c2)
    inp_hous = _sel("housing", "Housing", c3)
    inp_job = _sel("job", "Job", c4)

    c1, c2, c3, c4 = st.columns(4)
    inp_od = _sel("other_debtors", "Other Debtors", c1)
    inp_oip = _sel("other_installment_plans", "Other Plans", c2)
    inp_res = c3.slider("Residence (years)", 1, 4,
                        int(st.session_state.get("inp_residence_since", 3)), key="inp_residence_since")
    inp_nd = c4.number_input("Dependents", 1, 2,
                             int(st.session_state.get("inp_num_dependents", 1)), key="inp_num_dependents")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    predict_clicked = st.button("Analyze Risk", use_container_width=True, type="primary")

    if predict_clicked:
        raw = {
            "checking_account": CAT["checking_account"][inp_ca],
            "credit_history": CAT["credit_history"][inp_ch],
            "purpose": CAT["purpose"][inp_pur],
            "savings_account": CAT["savings_account"][inp_sa],
            "employment_since": CAT["employment_since"][inp_emp],
            "personal_status_sex": CAT["personal_status_sex"][inp_pss],
            "other_debtors": CAT["other_debtors"][inp_od],
            "property": CAT["property"][inp_prop],
            "other_installment_plans": CAT["other_installment_plans"][inp_oip],
            "housing": CAT["housing"][inp_hous],
            "job": CAT["job"][inp_job],
            "telephone": CAT["telephone"][inp_tel],
            "foreign_worker": CAT["foreign_worker"][inp_fw],
            "duration": inp_dur,
            "credit_amount": inp_amt,
            "installment_rate": inp_ir,
            "residence_since": inp_res,
            "age": inp_age,
            "existing_credits": inp_ec,
            "num_dependents": inp_nd,
        }

        with st.spinner("Running prediction..."):
            try:
                model = artefacts[model_choice]
                scaler = artefacts["scaler"]
                encoders = artefacts["encoders"]
                feature_names = artefacts["feature_names"]

                input_df = pd.DataFrame([raw])
                input_df = engineer_features(input_df)
                input_encoded, _ = encode_categoricals(input_df, fit=False, encoders=encoders)
                for col in feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                input_encoded = input_encoded[feature_names]
                input_scaled, _ = scale_features(input_encoded.values, fit=False, scaler=scaler)

                prob = predict_default_probability(model, input_scaled)
                risk_cat = assign_risk_category(prob)
                credit_score = calculate_credit_score(prob)
                risk_color = get_risk_color(risk_cat)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Prediction Results</div>', unsafe_allow_html=True)

        prob_pct = prob * 100
        if prob_pct < 30: p_color = "#10b981"
        elif prob_pct < 60: p_color = "#f59e0b"
        else: p_color = "#ef4444"

        if credit_score >= 740: sc_color = "#10b981"
        elif credit_score >= 670: sc_color = "#fbbf24"
        elif credit_score >= 580: sc_color = "#f97316"
        else: sc_color = "#ef4444"

        score_pct = (credit_score - 300) / 550
        arc_len = score_pct * 251

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">Default Probability</div>
                <div class="kpi-value" style="color:{p_color}">{prob_pct:.1f}%</div>
                <div style="width:80%;height:6px;background:rgba(255,255,255,0.06);border-radius:6px;margin-top:12px;overflow:hidden">
                    <div style="width:{min(prob_pct, 100)}%;height:100%;background:{p_color};border-radius:6px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">Credit Score</div>
                <div style="position:relative;width:180px;height:100px;margin:8px auto 0">
                    <svg viewBox="0 0 120 70" width="180" height="100">
                        <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="8" stroke-linecap="round"/>
                        <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="{sc_color}" stroke-width="8" stroke-linecap="round"
                              stroke-dasharray="{arc_len} 251" style="filter:drop-shadow(0 0 6px {sc_color}40)"/>
                    </svg>
                    <div style="position:absolute;bottom:0;left:50%;transform:translateX(-50%);text-align:center">
                        <div style="font-size:2.4rem;font-weight:800;color:{sc_color};line-height:1">{credit_score}</div>
                        <div style="font-size:0.65rem;color:#64748b;margin-top:2px">300 — 850</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            icon_map = {"Low Risk": "&#128994;", "Medium Risk": "&#128992;", "High Risk": "&#128308;"}
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">Risk Category</div>
                <div style="font-size:3rem;margin-bottom:14px">{icon_map.get(risk_cat, "")}</div>
                {risk_pill(risk_cat)}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col_shap, col_feat = st.columns(2)

        with col_shap:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">SHAP Explanation</div>', unsafe_allow_html=True)
            try:
                shap_vals = get_shap_explanation(model, input_scaled, feature_names=feature_names)
                if shap_vals is not None:
                    import shap as shap_lib
                    fig_s, ax_s = plt.subplots(figsize=(7, 5))
                    fig_s.patch.set_facecolor("#0d1117")
                    ax_s.set_facecolor("#0d1117")
                    shap_lib.plots.waterfall(shap_vals[0], show=False)
                    for item in ax_s.get_xticklabels() + ax_s.get_yticklabels():
                        item.set_color("#94a3b8")
                    ax_s.xaxis.label.set_color("#94a3b8")
                    ax_s.title.set_color("#94a3b8")
                    plt.tight_layout()
                    st.pyplot(fig_s)
                    plt.close(fig_s)
                else:
                    st.info("SHAP not available for this model.")
            except Exception:
                st.info("SHAP waterfall plot could not be generated for this model type.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_feat:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Feature Importance</div>', unsafe_allow_html=True)
            try:
                if shap_vals is not None and hasattr(shap_vals, "values"):
                    vals = np.abs(shap_vals.values).flatten()
                    if len(vals) == len(feature_names):
                        imp_df = pd.DataFrame({"feature": feature_names, "importance": vals})
                        imp_df = imp_df.nlargest(12, "importance").sort_values("importance")
                        fig_f = px.bar(imp_df, x="importance", y="feature", orientation="h",
                                       color="importance",
                                       color_continuous_scale=["#0ea5e9", "#34d399"])
                        fig_f.update_layout(
                            height=400,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", color="#94a3b8"),
                            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title=""),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title=""),
                            coloraxis_showscale=False,
                            margin=dict(t=10, b=20, l=10, r=10),
                        )
                        st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("Feature count mismatch.")
                else:
                    st.info("Feature importance requires SHAP values.")
            except Exception:
                st.info("Could not generate feature importance chart.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Prediction Summary</div>', unsafe_allow_html=True)
            certainty = abs(prob - 0.5) * 200
            cert_color = "#10b981" if certainty > 60 else ("#f59e0b" if certainty > 30 else "#ef4444")
            st.markdown(f"""
            <table class="clean-table">
                <tr><td>Model</td><td style="color:#38bdf8;font-weight:700">{model_choice}</td></tr>
                <tr><td>Default Probability</td><td style="color:{p_color};font-weight:700">{prob_pct:.2f}%</td></tr>
                <tr><td>Credit Score</td><td style="font-weight:700">{credit_score}</td></tr>
                <tr><td>Risk Category</td><td>{risk_pill(risk_cat)}</td></tr>
                <tr><td>Model Certainty</td><td style="color:{cert_color};font-weight:700">{certainty:.0f}%</td></tr>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Borrower Snapshot</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="clean-table">
                <tr><td>Age</td><td>{inp_age}</td></tr>
                <tr><td>Credit Amount</td><td>{inp_amt:,} DM</td></tr>
                <tr><td>Duration</td><td>{inp_dur} months</td></tr>
                <tr><td>Checking Account</td><td>{inp_ca}</td></tr>
                <tr><td>Credit History</td><td>{inp_ch}</td></tr>
                <tr><td>Employment</td><td>{inp_emp}</td></tr>
                <tr><td>Housing</td><td>{inp_hous}</td></tr>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#475569">
            <div style="font-size:3.5rem;margin-bottom:16px;opacity:0.6">&#128737;</div>
            <h3 style="color:#64748b;font-weight:600;margin-bottom:8px">Ready to Analyze</h3>
            <p style="max-width:480px;margin:auto;line-height:1.8;font-size:0.9rem">
                Fill in the borrower details above and click
                <strong style="color:#38bdf8">Analyze Risk</strong>
                to generate a credit risk assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_compare:
    st.markdown('<div class="sec-title">Model Performance Metrics</div>', unsafe_allow_html=True)

    results_df = load_results()
    if results_df is not None:
        st.markdown(f'<div class="card">{results_table(results_df)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Visual Comparison</div>', unsafe_allow_html=True)
        metric_cols = [c for c in results_df.columns if c not in ("model", "Unnamed: 0", "logloss")]
        if "model" not in results_df.columns and "Unnamed: 0" in results_df.columns:
            results_df = results_df.rename(columns={"Unnamed: 0": "model"})

        if "model" in results_df.columns:
            melt = results_df.melt(id_vars="model", value_vars=metric_cols,
                                    var_name="Metric", value_name="Score")
            fig_bar = px.bar(melt, x="model", y="Score", color="Metric", barmode="group",
                            color_discrete_sequence=["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"])
            fig_bar.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title=""),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Score", range=[0, 1]),
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=20, b=60),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Evaluation Charts</div>', unsafe_allow_html=True)

        plot_files = {
            "ROC Curves": "roc_curves.png",
            "Precision-Recall": "pr_curves.png",
            "Confusion Matrix": "confusion_matrix.png",
            "Feature Importance": "feature_importance.png",
        }
        pc1, pc2 = st.columns(2)
        for idx, (title, fname) in enumerate(plot_files.items()):
            fpath = MODELS_DIR / fname
            if fpath.exists():
                col = pc1 if idx % 2 == 0 else pc2
                with col:
                    st.markdown(f'<div class="card"><div class="sec-title">{title}</div>', unsafe_allow_html=True)
                    st.image(str(fpath), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No results data found. Run `python main.py` to train models.")

st.markdown("""
<div class="disclaimer">
    This dashboard uses ML models trained on the German Credit Dataset for educational purposes only.
    Predictions should not be used as the sole basis for real-world lending decisions.
</div>
""", unsafe_allow_html=True)
