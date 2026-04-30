# Marketing ROI Optimizer

Prédit les ventes (`Sales`) à partir des budgets marketing
(`TV`, `Radio`, `Social Media`) et du type d'influenceur, avec **explication
SHAP locale** pour chaque prédiction (cf. notebook `04_Interpretability`).

## Structure

```
data/          jeux de données bruts
notebooks/     EDA, preprocessing, modélisation, interprétabilité
src/
  preprocessing/   pipeline sklearn (ColumnTransformer + scaler/OHE)
  models/          entraînement des 4 modèles + sélection du meilleur
  api/             service REST FastAPI
models/        artefacts sérialisés (best_model.pkl, preprocessor.pkl)
reports/       métriques (metrics.json) et figures
```

## Installation

```bash
pip install -r requirements.txt
```

## Entraînement

```bash
python src/models/train.py
```

Génère `models/best_model.pkl`, `models/preprocessor.pkl` et
`reports/metrics.json`.

## API REST

Lancement local :

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Documentation interactive :
- Swagger UI → http://localhost:8000/docs
- ReDoc     → http://localhost:8000/redoc

### Endpoints

| Méthode | Route          | Description                                        |
|---------|----------------|----------------------------------------------------|
| GET     | `/health`      | État du service (modèle chargé ?).                 |
| GET     | `/model-info`  | Type de modèle, features, métriques.               |
| POST    | `/predict`     | Prédiction + explication SHAP locale.              |

### Exemple — `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TV": 50, "Radio": 20, "Social Media": 5, "Influencer": "Macro"}'
```

Réponse (extrait) :

```json
{
  "sales_pred": 178.2954,
  "model_name": "RandomForestRegressor",
  "base_value": 192.7019,
  "explainer":  "TreeExplainer",
  "contributions": [
    { "feature": "TV",           "value": -0.16, "shap": -15.06 },
    { "feature": "Radio",        "value":  0.19, "shap":   0.48 },
    { "feature": "Social Media", "value":  0.76, "shap":   0.08 },
    { "feature": "Macro",        "value":  1.00, "shap":   0.01 },
    { "feature": "Mega",         "value":  0.00, "shap":  -0.02 },
    { "feature": "Micro",        "value":  0.00, "shap":   0.05 },
    { "feature": "Nano",         "value":  0.00, "shap":   0.05 }
  ]
}
```

> Décomposition waterfall : `sales_pred ≈ base_value + Σ contributions.shap`
> (équivalent du `shap.plots.waterfall` de la cell 14 du notebook 04).

### Validation des entrées

- `TV`, `Radio`, `Social Media` : nombres ≥ 0 et ≤ 1000
- `Influencer` : `Mega` | `Macro` | `Micro` | `Nano`

Toute valeur invalide → réponse `422` avec détail du champ fautif.
Si les artefacts ne sont pas chargés → `503`.

## Dashboard

Frontend Streamlit qui consomme l'API (séparation modèle / interface) :

- formulaire de prédiction → appelle `/predict` et affiche le **waterfall SHAP**
- galerie des graphiques de `reports/` (EDA, modèles, SHAP)
- onglet "Modèle" qui affiche `/model-info` et les métriques

Lancement (avec l'API déjà démarrée) :

```bash
# Terminal 1 — l'API
uvicorn src.api.main:app --port 8000

# Terminal 2 — le dashboard
streamlit run dashboard/app.py
```

Dashboard accessible sur http://localhost:8501. L'URL de l'API est
configurable dans la sidebar (par défaut `http://localhost:8000`, ou via
la variable d'environnement `API_URL`).
