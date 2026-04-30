"""
Dashboard Streamlit — Marketing ROI Optimizer
==============================================

Interface utilisateur qui consomme l'API FastAPI (/predict, /health, /model-info)
et affiche les graphiques d'analyse pré-générés (reports/).

Lancement :
    streamlit run dashboard/app.py

Pré-requis : l'API doit tourner en parallèle.
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR  = PROJECT_ROOT / "reports"
DEFAULT_API  = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Marketing ROI Optimizer",
    page_icon="📊",
    layout="wide",
)


# ─────────────────────────────────────────────
# Helpers API
# ─────────────────────────────────────────────
def api_get(api_url: str, path: str, timeout: float = 5.0):
    try:
        r = requests.get(f"{api_url}{path}", timeout=timeout)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    except requests.RequestException as exc:
        return None, str(exc)


def api_post(api_url: str, path: str, json_body: dict, timeout: float = 30.0):
    try:
        r = requests.post(f"{api_url}{path}", json=json_body, timeout=timeout)
        try:
            payload = r.json()
        except ValueError:
            payload = r.text
        return r.status_code, payload
    except requests.RequestException as exc:
        return None, str(exc)


# ─────────────────────────────────────────────
# Sidebar — connexion API
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_url = st.text_input(
        "URL de l'API",
        value=DEFAULT_API,
        help="Endpoint FastAPI servant /predict, /health, /model-info.",
    )

    if st.button("🔄 Tester la connexion", use_container_width=True):
        st.session_state["health_check"] = api_get(api_url, "/health")

    health = st.session_state.get("health_check") or api_get(api_url, "/health")
    st.session_state["health_check"] = health

    code, body = health
    if code == 200 and isinstance(body, dict) and body.get("status") == "ok":
        st.success(f"API OK — v{body.get('version', '?')}")
    elif code is None:
        st.error("API injoignable")
        st.caption(str(body))
    else:
        st.warning(f"API : {body}")

    st.divider()
    st.caption(
        "Le dashboard consomme l'API REST — séparation modèle / interface. "
        "Lance d'abord :\n```\nuvicorn src.api.main:app --port 8000\n```"
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📊 Marketing ROI Optimizer")
st.markdown(
    "Prédiction des ventes (`Sales`) en fonction des budgets marketing, "
    "avec explication SHAP locale pour chaque prédiction."
)

tab_predict, tab_reports, tab_model = st.tabs(
    ["🎯 Prédiction", "📈 Rapports & graphiques", "ℹ️ Modèle"]
)


# ─────────────────────────────────────────────
# Onglet 1 — Prédiction
# ─────────────────────────────────────────────
with tab_predict:
    st.subheader("Lancer une prédiction")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        with st.form("predict_form"):
            tv     = st.slider("Budget TV (k$)",            0.0, 1000.0, 50.0, step=1.0)
            radio  = st.slider("Budget Radio (k$)",         0.0, 1000.0, 20.0, step=0.5)
            social = st.slider("Budget Social Media (k$)",  0.0, 1000.0, 5.0,  step=0.1)
            influencer = st.selectbox("Type d'influenceur", ["Mega", "Macro", "Micro", "Nano"])

            submitted = st.form_submit_button("🚀 Prédire", use_container_width=True)

    with col_out:
        if submitted:
            payload = {
                "TV":           tv,
                "Radio":        radio,
                "Social Media": social,
                "Influencer":   influencer,
            }
            with st.spinner("Appel de l'API..."):
                code, body = api_post(api_url, "/predict", payload)

            if code == 200 and isinstance(body, dict):
                pred  = body["sales_pred"]
                bv    = body.get("base_value")
                model = body.get("model_name", "?")
                st.metric("Ventes prédites", f"{pred:,.2f} k$")
                st.caption(f"Modèle : {model}")

                contributions = body.get("contributions") or []
                if bv is not None and contributions:
                    st.markdown("#### Décomposition SHAP (waterfall local)")
                    st.caption(
                        f"Reconstruction : `base_value ({bv:.2f}) + Σ shap → prédiction`"
                    )

                    df = pd.DataFrame(contributions)
                    df["abs_shap"] = df["shap"].abs()
                    df = df.sort_values("abs_shap", ascending=True)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in df["shap"]]
                    ax.barh(df["feature"], df["shap"], color=colors, edgecolor="white")
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_xlabel("Contribution SHAP")
                    ax.set_title(f"Contributions par feature — {model}")
                    plt.tight_layout()
                    st.pyplot(fig)

                    with st.expander("Détails numériques"):
                        st.dataframe(
                            pd.DataFrame(contributions)[["feature", "value", "shap"]],
                            use_container_width=True,
                        )
                elif body.get("explainer") is None:
                    st.info("Pas d'explication SHAP disponible pour ce modèle.")

            elif code in (400, 422):
                st.error(f"Entrée invalide : {body}")
            elif code == 503:
                st.error(f"Service indisponible : {body}")
            elif code is None:
                st.error(f"Impossible de joindre l'API : {body}")
            else:
                st.error(f"Erreur ({code}) : {body}")
        else:
            st.info("👈 Renseigne les budgets et clique sur **Prédire**.")


# ─────────────────────────────────────────────
# Onglet 2 — Rapports / graphiques
# ─────────────────────────────────────────────
with tab_reports:
    st.subheader("Graphiques générés par les notebooks")

    if not REPORTS_DIR.exists():
        st.warning(f"Dossier introuvable : {REPORTS_DIR}")
    else:
        images = sorted(REPORTS_DIR.glob("*.png"))
        if not images:
            st.info("Aucun graphique disponible. Lance les notebooks pour les générer.")
        else:
            sections = {
                "EDA": [
                    "sales_distribution.png",
                    "budgets_distributions.png",
                    "scatter_budgets_vs_sales.png",
                    "correlation_matrix.png",
                    "pairplot.png",
                    "influencer_analysis.png",
                ],
                "Preprocessing": [
                    "standardisation_before_after.png",
                    "feature_engineering_exploration.png",
                    "train_test_split_distribution.png",
                ],
                "Modélisation": [
                    "model_comparison.png",
                    "predictions_vs_real.png",
                    "residuals_analysis.png",
                ],
                "Interprétabilité": [
                    "feature_importance.png",
                    "feature_importance_native.png",
                    "permutation_importance.png",
                    "shap_summary.png",
                    "shap_bar.png",
                    "shap_waterfall.png",
                    "shap_dependence.png",
                ],
            }

            shown = set()
            for section_name, fnames in sections.items():
                section_imgs = [REPORTS_DIR / f for f in fnames if (REPORTS_DIR / f).exists()]
                if not section_imgs:
                    continue
                st.markdown(f"### {section_name}")
                for img_path in section_imgs:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
                    shown.add(img_path.name)

            extras = [p for p in images if p.name not in shown]
            if extras:
                st.markdown("### Autres")
                for img_path in extras:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)

        # SHAP ranking CSV s'il existe
        ranking_csv = REPORTS_DIR / "shap_ranking.csv"
        if ranking_csv.exists():
            st.markdown("### Ranking SHAP")
            st.dataframe(pd.read_csv(ranking_csv), use_container_width=True)


# ─────────────────────────────────────────────
# Onglet 3 — Model info
# ─────────────────────────────────────────────
with tab_model:
    st.subheader("Informations sur le modèle servi")

    code, body = api_get(api_url, "/model-info")

    if code == 200 and isinstance(body, dict):
        c1, c2 = st.columns(2)
        c1.metric("Modèle", body.get("model_name", "?"))
        c2.metric("Cible",  body.get("target", "?"))

        st.markdown("**Features attendues :** " + ", ".join(body.get("features", [])))
        st.markdown(f"**Type :** `{body.get('model_type', '?')}`")

        metrics = body.get("metrics")
        if metrics:
            st.markdown("#### Métriques (test)")
            test_metrics = metrics.get("test", {})
            cols = st.columns(len(test_metrics) or 1)
            for col, (k, v) in zip(cols, test_metrics.items()):
                col.metric(k, f"{v}")

            with st.expander("Toutes les métriques (train + CV)"):
                st.json(metrics)

        with st.expander("Artefacts"):
            st.json(body.get("artifacts", {}))

    elif code is None:
        st.error(f"API injoignable : {body}")
    else:
        st.warning(f"Erreur ({code}) : {body}")
