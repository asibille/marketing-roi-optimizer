"""
API REST — Marketing ROI Optimizer
===================================

Endpoints :
  • POST /predict      → prédiction + explication SHAP locale (cf. notebook 04)
  • GET  /health       → statut du service
  • GET  /model-info   → informations sur le modèle servi

Lancement local :
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Documentation interactive :
    http://localhost:8000/docs   (Swagger)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_service import ModelNotLoadedError, service
from src.api.schemas import (
    ClientFeatures,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)

API_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("marketing-roi-api")


# ─────────────────────────────────────────────
# Cycle de vie : on charge le modèle au démarrage
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Chargement des artefacts ML...")
        service.load()
        logger.info("Modèle chargé : %s", service.model_name)
    except Exception as exc:  # pragma: no cover — surface au démarrage
        logger.exception("Échec du chargement du modèle : %s", exc)
        # On laisse l'app démarrer en mode dégradé : /health renverra "down"
    yield


app = FastAPI(
    title="Marketing ROI Optimizer API",
    description=(
        "Service d'inférence pour prédire les ventes (`Sales`) à partir des "
        "budgets marketing (TV, Radio, Social Media) et du type d'influenceur. "
        "Chaque prédiction est accompagnée de son explication SHAP locale."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)

# CORS large par défaut — utile pour le dashboard (à restreindre en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Gestion globale des erreurs
# ─────────────────────────────────────────────
@app.exception_handler(ModelNotLoadedError)
async def _model_not_loaded_handler(_, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Marketing ROI Optimizer API",
        "version": API_VERSION,
        "docs":    "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["monitoring"],
    summary="Vérifie l'état du service",
)
def health() -> HealthResponse:
    model_loaded        = service.model is not None
    preprocessor_loaded = service.preprocessor is not None

    if model_loaded and preprocessor_loaded:
        st = "ok"
    elif model_loaded or preprocessor_loaded:
        st = "degraded"
    else:
        st = "down"

    return HealthResponse(
        status=st,
        model_loaded=model_loaded,
        preprocessor_loaded=preprocessor_loaded,
        version=API_VERSION,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["monitoring"],
    summary="Informations sur le modèle servi",
)
def model_info() -> ModelInfoResponse:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé.",
        )
    return ModelInfoResponse(**service.info())


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Prédiction + explication SHAP locale",
    responses={
        400: {"model": ErrorResponse, "description": "Entrée invalide."},
        503: {"model": ErrorResponse, "description": "Modèle non chargé."},
    },
)
def predict(payload: ClientFeatures) -> PredictResponse:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé.",
        )

    try:
        item = payload.model_dump(by_alias=True)
        result = service.explain_one(item)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Entrée invalide : {exc}",
        )
    except Exception as exc:
        logger.exception("Erreur d'inférence")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'inférence : {exc}",
        )

    return PredictResponse(
        sales_pred=round(result["sales_pred"], 4),
        model_name=service.model_name,
        base_value=(
            round(result["base_value"], 4) if result["base_value"] is not None else None
        ),
        contributions=result["contributions"],
        explainer=result["explainer"],
    )
