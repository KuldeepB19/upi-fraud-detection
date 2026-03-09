import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_artifacts, engineer_features, FEATURES
from src.train_model import train
from src.data_generator import generate_transactions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.markdown("""
<style>
  .metric-card {
    background: #1C2333; border-radius: 12px; padding: 20px 24px;
    border-left: 4px solid; margin-bottom: 8px; text-align: center;
  }
  .metric-val   { font-size: 2rem; font-weight: 700; margin: 0; }
  .metric-label { font-size: 0.8rem; color: #8B9BB4; margin: 0;
                  text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 Model Performance")
st.caption("Evaluation metrics for Random Forest and XGBoost on test data")
st.divider()

# ── Load artifacts ─────────────────────────────────────────────────────────────
artifacts = load_artifacts()
if artifacts is None:
    st.info("No trained model found. Training now...")
    with st.spinner("Training..."):
        train()
    artifacts = load_artifacts()

# ── Load data + rebuild same test split ───────────────────────────────────────
csv_path = 'data/transactions.csv'
if not os.path.exists(csv_path):
    df_full = generate_transactions()
else:
    df_full = pd.read_csv(csv_path)

X_all = engineer_features(df_full, artifacts['encoders'], artifacts['threshold'])
y_all = df_full['is_fraud']

_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# ── Predictions from each model ───────────────────────────────────────────────
rf_preds   = artifacts['rf'].predict(X_test)
xgb_preds  = artifacts['xgb'].predict(X_test)
rf_proba   = artifacts['rf'].predict_proba(X_test)[:, 1]
xgb_proba  = artifacts['xgb'].predict_proba(X_test)[:, 1]
ens_preds  = ((rf_proba + xgb_proba) / 2 >= 0.5).astype(int)

all_preds = {
    'Random Forest' : rf_preds,
    'XGBoost'       : xgb_preds,
    'Ensemble (Avg)': ens_preds,
}

# ── Metrics for all models ────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred, name):
    return {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_true, y_pred) * 100, 2),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        'Recall'   : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        'F1 Score' : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }

results_df = pd.DataFrame([
    calc_metrics(y_test, rf_preds,  'Random Forest'),
    calc_metrics(y_test, xgb_preds, 'XGBoost'),
    calc_metrics(y_test, ens_preds, 'Ensemble (Avg)'),
])

# ── Radio button — ABOVE everything so selection drives the cards + matrix ────
selected = st.radio("Select model to inspect:",
                    ['Random Forest', 'XGBoost', 'Ensemble (Avg)'],
                    horizontal=True)

chosen_preds = all_preds[selected]
chosen_row   = results_df[results_df['Model'] == selected].iloc[0]

st.divider()

# ── 4 KPI cards — update based on selected model ──────────────────────────────
c1, c2, c3, c4 = st.columns(4)

def card(col, label, value, color):
    col.markdown(f"""
    <div class="metric-card" style="border-color:{color}">
      <p class="metric-label">{label}</p>
      <p class="metric-val" style="color:{color}">{value}%</p>
    </div>""", unsafe_allow_html=True)

card(c1, "Accuracy",  chosen_row['Accuracy'],  "#2E86C1")
card(c2, "Precision", chosen_row['Precision'], "#8E44AD")
card(c3, "Recall",    chosen_row['Recall'],    "#E74C3C")
card(c4, "F1 Score",  chosen_row['F1 Score'],  "#27AE60")

st.write("")

# ── Confusion matrix + comparison chart ──────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"#### Confusion Matrix — {selected}")

    cm = confusion_matrix(y_test, chosen_preds)
    tn, fp, fn, tp = cm.ravel()

    # Correct order: Fraud on top, Legit on bottom (matches sklearn output)
    z    = [[tp, fn], [fp, tn]]
    x    = ['Predicted Fraud', 'Predicted Legit']
    y_ax = ['Actual Fraud',    'Actual Legit']

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y_ax,
        colorscale='Blues',
        text=z, texttemplate="%{text}",
        showscale=False
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#FAFAFA',
        height=320,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
- **True Positives (TP) = {tp}** — Fraud correctly caught ✅
- **True Negatives (TN) = {tn}** — Legit correctly allowed ✅
- **False Negatives (FN) = {fn}** — Fraud missed 🚨 (most dangerous!)
- **False Positives (FP) = {fp}** — Legit wrongly flagged ⚠️
    """)

with col_right:
    st.markdown("#### All Models Comparison")
    fig2 = go.Figure()
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for _, row in results_df.iterrows():
        fig2.add_trace(go.Bar(
            name=row['Model'],
            x=metric_cols,
            y=[row[m] for m in metric_cols],
            text=[f"{row[m]}%" for m in metric_cols],
            textposition='outside',
        ))

    fig2.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#FAFAFA',
        yaxis=dict(range=[0, 115]),
        legend=dict(orientation='h', y=-0.2),
        height=320,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Full metrics table ─────────────────────────────────────────────────────────
st.markdown("#### Full Metrics Table")
st.dataframe(results_df.set_index('Model'), use_container_width=True)

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
st.markdown("#### Feature Importance (Random Forest)")
st.caption("Which features matter most when predicting fraud")

feat_df = pd.DataFrame({
    'Feature'   : FEATURES,
    'Importance': (artifacts['rf'].feature_importances_ * 100).round(2)
}).sort_values('Importance', ascending=True)

fig3 = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
              color='Importance', color_continuous_scale='Blues',
              labels={'Importance': 'Importance (%)'})
fig3.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#FAFAFA',
    coloraxis_showscale=False,
    height=380,
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig3, use_container_width=True)
