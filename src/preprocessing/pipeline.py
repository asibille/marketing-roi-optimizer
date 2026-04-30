import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
DATA_PATH      = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'marketing_and_sales.csv')
ARTIFACTS_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

TARGET_COL     = 'Sales'
NUM_FEATURES   = ['TV', 'Radio', 'Social Media']
CAT_FEATURES   = ['Influencer']
RANDOM_STATE   = 42
TEST_SIZE      = 0.2


# ─────────────────────────────────────────────
# 1. CHARGEMENT ET NETTOYAGE
# ─────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV et effectue le nettoyage de base."""
    df = pd.read_csv(path)

    # Supprimer les doublons
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f'  [nettoyage] {n_dropped} doublon(s) supprimé(s)')

    # Supprimer les lignes où la cible est manquante
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f'  [nettoyage] {n_dropped} ligne(s) supprimée(s) (Sales manquant)')

    print(f'  [nettoyage] Dataset final : {df.shape[0]} lignes × {df.shape[1]} colonnes')
    return df


# ─────────────────────────────────────────────
# 2. CONSTRUCTION DU PIPELINE SKLEARN
# ─────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer :
      - Variables numériques : imputation médiane + standardisation
      - Variables catégorielles : imputation mode + One-Hot Encoding
    """

    # Pipeline pour les variables numériques
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),   # remplace les NaN par la médiane
        ('scaler',  StandardScaler())                    # standardisation (moyenne=0, std=1)
    ])

    # Pipeline pour les variables catégorielles
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # remplace les NaN par le mode
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Assemblage
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline,      NUM_FEATURES),
        ('cat', categorical_pipeline,  CAT_FEATURES)
    ])

    return preprocessor


# ─────────────────────────────────────────────
# 3. FONCTION PRINCIPALE
# ─────────────────────────────────────────────
def load_and_prepare_data(path: str = DATA_PATH, test_size: float = TEST_SIZE):
    """
    Pipeline complet : charge, nettoie, split, et prépare les données.

    Retourne :
        X_train, X_test, y_train, y_test (arrays numpy)
        preprocessor (ColumnTransformer fitté sur le train set uniquement)
        feature_names (liste des noms de colonnes après transformation)
    """
    print('Chargement et nettoyage...')
    df = load_data(path)

    # Séparation features / cible
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET_COL].values

    # Split train / test — AVANT tout preprocessing pour éviter le data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE
    )
    print(f'  [split] Train : {X_train.shape[0]} | Test : {X_test.shape[0]}')

    # Construction et fit du preprocessor SUR LE TRAIN SET UNIQUEMENT
    print('Fitting du preprocessor sur le train set...')
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Transformation
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Récupération des noms de features après OHE
    ohe_categories = preprocessor.named_transformers_['cat']['encoder'].categories_[0]
    feature_names = NUM_FEATURES + list(ohe_categories)

    print(f'  [features] {len(feature_names)} features finales : {feature_names}')
    print('Preprocessing terminé.')

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names


# ─────────────────────────────────────────────
# 4. SAUVEGARDE DU PREPROCESSOR
# ─────────────────────────────────────────────
def save_preprocessor(preprocessor, path: str = None):
    """Sérialise le preprocessor fitté pour l'API et le dashboard."""
    if path is None:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        path = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
    joblib.dump(preprocessor, path)
    print(f'Preprocessor sauvegardé → {path}')


def load_preprocessor(path: str = None):
    """Charge le preprocessor sérialisé."""
    if path is None:
        path = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
    return joblib.load(path)


# ─────────────────────────────────────────────
# 5. EXÉCUTION DIRECTE (test rapide)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_prepare_data()
    save_preprocessor(preprocessor)

    print(f'\n Shapes :')
    print(f'  X_train : {X_train.shape}')
    print(f'  X_test  : {X_test.shape}')
    print(f'  y_train : {y_train.shape}')
    print(f'  y_test  : {y_test.shape}')
