"""
Schémas Pydantic — validation des entrées / sorties de l'API.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# Catégories acceptées pour la variable Influencer (cf. dataset)
InfluencerType = Literal["Mega", "Macro", "Micro", "Nano"]


class ClientFeatures(BaseModel):
    """Données client pour une prédiction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "TV": 50.0,
                "Radio": 20.0,
                "Social_Media": 5.0,
                "Influencer": "Macro",
            }
        }
    )

    TV: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Budget TV (en milliers de $).",
    )
    Radio: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Budget Radio (en milliers de $).",
    )
    Social_Media: float = Field(
        ...,
        ge=0,
        le=1000,
        alias="Social Media",
        description="Budget Social Media (en milliers de $).",
    )
    Influencer: InfluencerType = Field(
        ...,
        description="Type d'influenceur : Mega / Macro / Micro / Nano.",
    )


class FeatureContribution(BaseModel):
    """Contribution SHAP d'une feature à la prédiction (équivalent d'une ligne du waterfall)."""

    feature: str = Field(..., description="Nom de la feature (après preprocessing).")
    value:   float = Field(..., description="Valeur transformée passée au modèle.")
    shap:    float = Field(..., description="Contribution SHAP : impact sur la prédiction.")


class PredictResponse(BaseModel):
    """Réponse /predict — prédiction + explication locale (cf. notebook 04)."""

    sales_pred:    float = Field(..., description="Ventes prédites (milliers de $).")
    model_name:    str   = Field(..., description="Nom du modèle utilisé.")
    base_value:    Optional[float] = Field(
        None,
        description="Valeur attendue du modèle (E[f(X)]) — point de départ du waterfall SHAP.",
    )
    contributions: Optional[List[FeatureContribution]] = Field(
        None,
        description="Décomposition SHAP : sales_pred ≈ base_value + Σ contributions.shap.",
    )
    explainer:     Optional[str] = Field(
        None,
        description="Explainer SHAP utilisé (TreeExplainer / KernelExplainer / None).",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    preprocessor_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    target: str
    metrics: Optional[dict] = None
    artifacts: dict


class ErrorResponse(BaseModel):
    detail: str
