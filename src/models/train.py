"""
Modélisation — Marketing ROI Optimizer
=======================================
Ce module entraîne, évalue et sauvegarde 4 modèles :
  1. Régression Linéaire  (baseline)
  2. Random Forest
  3. XGBoost (Gradient Boosting)
  4. MLP (réseau de neurones — Deep Learning)

Usage :
    python src/models/train.py
    # ou depuis un notebook :
    from src.models.train import train_all_models
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Chemin vers src/ pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.pipeline import load_and_prepare_data, save_preprocessor

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
RANDOM_STATE = 42
CV_FOLDS     = 5


# ─────────────────────────────────────────────
# 1. DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────
def get_models() -> dict:
    """Retourne les 4 modèles avec leurs hyperparamètres de base."""
    return {
        'LinearRegression': LinearRegression(),

        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        'XGBoost': XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0
        ),

        'MLP': MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE
        )
    }


# ─────────────────────────────────────────────
# 2. MÉTRIQUES
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    """Calcule MAE, RMSE et R² pour un modèle."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}


# ─────────────────────────────────────────────
# 3. ENTRAÎNEMENT ET ÉVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test) -> dict:
    """
    Entraîne un modèle et retourne ses métriques train/test + cross-validation.
    """
    print(f'\n Entraînement : {model_name}...')

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Métriques train / test
    metrics_train = compute_metrics(y_train, y_pred_train)
    metrics_test  = compute_metrics(y_test,  y_pred_test)

    # Cross-validation (sur le train set uniquement)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2   = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    cv_rmse = cross_val_score(model, X_train, y_train, cv=kf,
                               scoring='neg_root_mean_squared_error')

    result = {
        'model_name'  : model_name,
        'train'       : metrics_train,
        'test'        : metrics_test,
        'cv_r2_mean'  : round(cv_r2.mean(), 4),
        'cv_r2_std'   : round(cv_r2.std(),  4),
        'cv_rmse_mean': round(-cv_rmse.mean(), 4),
        'cv_rmse_std' : round(cv_rmse.std(),   4),
        'y_pred_test' : y_pred_test.tolist()
    }

    print(f'  Test  — MAE: {metrics_test["MAE"]:.4f} | RMSE: {metrics_test["RMSE"]:.4f} | R²: {metrics_test["R2"]:.4f}')
    print(f'  CV R² — {cv_r2.mean():.4f} ± {cv_r2.std():.4f}')

    return result


# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT DE TOUS LES MODÈLES
# ─────────────────────────────────────────────
def train_all_models(data_path=None):
    """
    Pipeline complet :
      - Charge et prépare les données
      - Entraîne les 4 modèles
      - Sauvegarde les modèles et les métriques
      - Retourne le résumé des résultats
    """
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Données
    kwargs = {'path': data_path} if data_path else {}
    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_prepare_data(**kwargs)
    save_preprocessor(preprocessor)

    models  = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        result = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        results[name] = result
        trained_models[name] = model

        # Sauvegarde du modèle
        model_path = os.path.join(MODELS_DIR, f'{name}.pkl')
        joblib.dump(model, model_path)
        print(f'  Modèle sauvegardé → {model_path}')

    # Sauvegarde des métriques en JSON
    metrics_export = {}
    for name, res in results.items():
        metrics_export[name] = {
            'train'       : res['train'],
            'test'        : res['test'],
            'cv_r2_mean'  : res['cv_r2_mean'],
            'cv_r2_std'   : res['cv_r2_std'],
            'cv_rmse_mean': res['cv_rmse_mean'],
            'cv_rmse_std' : res['cv_rmse_std'],
        }

    metrics_path = os.path.join(REPORTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_export, f, indent=2)
    print(f'\n Métriques sauvegardées → {metrics_path}')

    # Tableau récapitulatif
    summary = _build_summary(results)
    print('\n' + '='*65)
    print('COMPARAISON DES MODÈLES')
    print('='*65)
    print(summary.to_string(index=False))

    # Meilleur modèle (R² test)
    best_name = summary.loc[summary['R² Test'].idxmax(), 'Modèle']
    best_model = trained_models[best_name]
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    print(f'\nMeilleur modèle (R² Test) : {best_name}')
    print(f'   Sauvegardé → models/best_model.pkl')

    return results, trained_models, feature_names, X_test, y_test


# ─────────────────────────────────────────────
# 5. TABLEAU RÉCAPITULATIF
# ─────────────────────────────────────────────
def _build_summary(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({
            'Modèle'      : name,
            'MAE Test'    : res['test']['MAE'],
            'RMSE Test'   : res['test']['RMSE'],
            'R² Test'     : res['test']['R2'],
            'CV R² (mean)': res['cv_r2_mean'],
            'CV R² (std)' : res['cv_r2_std'],
        })
    df = pd.DataFrame(rows)
    return df.sort_values('R² Test', ascending=False).reset_index(drop=True)


def load_best_model():
    """Charge le meilleur modèle sauvegardé."""
    path = os.path.join(MODELS_DIR, 'best_model.pkl')
    return joblib.load(path)


# ─────────────────────────────────────────────
# EXÉCUTION DIRECTE
# ─────────────────────────────────────────────
if __name__ == '__main__':
    train_all_models()
