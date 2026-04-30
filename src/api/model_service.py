"""
Service modèle — encapsule le chargement des artefacts (preprocessor + modèle)
et l'inférence + explication SHAP locale (cf. notebook 04_Interpretability).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Chemins des artefacts
# ─────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
MODELS_DIR    = os.path.join(_PROJECT_ROOT, "models")
REPORTS_DIR   = os.path.join(_PROJECT_ROOT, "reports")

DEFAULT_MODEL_PATH        = os.path.join(MODELS_DIR, "best_model.pkl")
DEFAULT_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
DEFAULT_METRICS_PATH      = os.path.join(REPORTS_DIR, "metrics.json")

# Schéma de features attendu par le preprocessor (cf. src/preprocessing/pipeline.py)
NUM_FEATURES = ["TV", "Radio", "Social Media"]
CAT_FEATURES = ["Influencer"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET       = "Sales"

# Mapping classe sklearn/xgb → clé utilisée dans reports/metrics.json
_METRICS_KEY_BY_CLASS = {
    "LinearRegression":      "LinearRegression",
    "RandomForestRegressor": "RandomForest",
    "XGBRegressor":          "XGBoost",
    "MLPRegressor":          "MLP",
}

# Modèles supportés par shap.TreeExplainer (rapide, sans données de fond)
_TREE_BASED = {"RandomForestRegressor", "XGBRegressor", "GradientBoostingRegressor"}


class ModelNotLoadedError(RuntimeError):
    """Levée quand un appel d'inférence est tenté sans modèle chargé."""


class ModelService:
    """
    Singleton applicatif : charge une seule fois le modèle et le preprocessor,
    puis sert prédictions et explications locales.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
        metrics_path: str = DEFAULT_METRICS_PATH,
    ) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metrics_path = metrics_path

        self.model = None
        self.preprocessor = None
        self.metrics: Optional[dict] = None
        self.model_name: str = "unknown"

        # Initialisés au premier appel à explain()
        self._explainer = None
        self._explainer_kind: Optional[str] = None
        self._feature_names: Optional[List[str]] = None

    # ─────────────────────────────────────────
    # Chargement
    # ─────────────────────────────────────────
    def load(self) -> None:
        """Charge les artefacts depuis le disque. Idempotent."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modèle introuvable : {self.model_path}. "
                f"Lance d'abord `python src/models/train.py`."
            )
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor introuvable : {self.preprocessor_path}. "
                f"Lance d'abord `python src/models/train.py`."
            )

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model_name = type(self.model).__name__

        # Reproduction du calcul de feature_names fait dans pipeline.py
        try:
            ohe = self.preprocessor.named_transformers_["cat"]["encoder"]
            ohe_cats = list(ohe.categories_[0])
            self._feature_names = NUM_FEATURES + ohe_cats
        except Exception as exc:
            logger.warning("Impossible d'extraire les noms de features : %s", exc)
            self._feature_names = None

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        # Reset explainer (sera reconstruit à la demande)
        self._explainer = None
        self._explainer_kind = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names) if self._feature_names else []

    # ─────────────────────────────────────────
    # Inférence
    # ─────────────────────────────────────────
    def _to_dataframe(self, items: Iterable[dict]) -> pd.DataFrame:
        rows = []
        for it in items:
            rows.append({
                "TV":           it["TV"],
                "Radio":        it["Radio"],
                "Social Media": it.get("Social Media", it.get("Social_Media")),
                "Influencer":   it["Influencer"],
            })
        return pd.DataFrame(rows, columns=ALL_FEATURES)

    def predict(self, items: Iterable[dict]) -> List[float]:
        if not self.is_ready:
            raise ModelNotLoadedError("Le modèle n'est pas chargé.")
        df = self._to_dataframe(items)
        X = self.preprocessor.transform(df)
        y_pred = np.asarray(self.model.predict(X)).ravel()
        return [float(v) for v in y_pred]

    # ─────────────────────────────────────────
    # Explication SHAP locale (cf. notebook 04, cell 14)
    # ─────────────────────────────────────────
    def _build_explainer(self):
        """
        Construit l'explainer SHAP adapté au modèle.
        TreeExplainer pour RF/XGBoost (rapide, exact), KernelExplainer sinon
        (plus lent, nécessite un échantillon de fond).
        """
        try:
            import shap
        except ImportError:
            logger.warning("Le package `shap` n'est pas installé — pas d'explication.")
            return None, None

        if self.model_name in _TREE_BASED:
            return shap.TreeExplainer(self.model), "TreeExplainer"

        # KernelExplainer : besoin d'un background. On en construit un petit
        # à partir des centres du scaler (médiane) — pas idéal mais évite de
        # devoir embarquer X_train dans l'API.
        try:
            n_features = len(self._feature_names) if self._feature_names else None
            if n_features is None:
                return None, None
            background = np.zeros((1, n_features))  # tout à zéro = origine standardisée
            return (
                shap.KernelExplainer(self.model.predict, background),
                "KernelExplainer",
            )
        except Exception as exc:
            logger.warning("Impossible de construire un KernelExplainer : %s", exc)
            return None, None

    def explain_one(self, item: dict) -> dict:
        """
        Reproduit le bloc waterfall du notebook 04 pour UNE prédiction :
        retourne la prédiction, base_value et la contribution SHAP par feature.
        """
        if not self.is_ready:
            raise ModelNotLoadedError("Le modèle n'est pas chargé.")

        df = self._to_dataframe([item])
        X = self.preprocessor.transform(df)
        pred = float(np.asarray(self.model.predict(X)).ravel()[0])

        if self._explainer is None:
            self._explainer, self._explainer_kind = self._build_explainer()

        if self._explainer is None or not self._feature_names:
            return {
                "sales_pred": pred,
                "base_value": None,
                "contributions": None,
                "explainer": None,
            }

        shap_values = self._explainer.shap_values(X)
        # shap_values peut être (1, n_features) ou liste — on normalise
        sv = np.asarray(shap_values)
        if sv.ndim == 3:        # certains explainers : (n_outputs, n_samples, n_features)
            sv = sv[0]
        sv = sv[0]              # 1ère (et seule) ligne

        ev = self._explainer.expected_value
        base_value = float(ev[0]) if hasattr(ev, "__len__") and not np.isscalar(ev) else float(ev)

        x_row = np.asarray(X)[0]
        contributions = [
            {"feature": name, "value": float(v), "shap": float(s)}
            for name, v, s in zip(self._feature_names, x_row, sv)
        ]

        return {
            "sales_pred":    pred,
            "base_value":    base_value,
            "contributions": contributions,
            "explainer":     self._explainer_kind,
        }

    # ─────────────────────────────────────────
    # Méta-informations
    # ─────────────────────────────────────────
    def info(self) -> dict:
        if not self.is_ready:
            raise ModelNotLoadedError("Le modèle n'est pas chargé.")

        model_metrics = None
        if self.metrics:
            metrics_key = _METRICS_KEY_BY_CLASS.get(self.model_name, self.model_name)
            model_metrics = self.metrics.get(metrics_key) or self.metrics.get(self.model_name)

        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__module__ + "." + type(self.model).__name__,
            "features":   ALL_FEATURES,
            "target":     TARGET,
            "metrics":    model_metrics,
            "artifacts": {
                "model_path":        self.model_path,
                "preprocessor_path": self.preprocessor_path,
            },
        }


# Instance partagée par l'application
service = ModelService()
