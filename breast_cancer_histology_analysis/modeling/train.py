import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import joblib
from loguru import logger
import typer

from breast_cancer_histology_analysis.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_COLUMNS, 
    DEFAULT_RANDOM_STATE, DEFAULT_NAN_HANDLING_STRATEGY
)

app = typer.Typer()

@app.command()
def train_all_models_for_magnification(
    target_magnification: str = typer.Option(..., "--magnification", "-m", help="Target magnification features to use."),
    features_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Directory containing training features CSV."),
    output_models_dir: Path = typer.Option(MODELS_DIR, help="Directory to save trained models/transformers."),
    nan_strategy: str = typer.Option(DEFAULT_NAN_HANDLING_STRATEGY, help="'drop' or 'impute_mean'."),
    random_state: int = typer.Option(DEFAULT_RANDOM_STATE, help="Random state.")
):
    """Trains all specified models for a given magnification's feature set."""
    logger.info(f"Starting model training for magnification: {target_magnification}")
    train_features_path = features_dir / f"train_features_{target_magnification}.csv"

    if not train_features_path.exists():
        logger.error(f"Training features file not found: {train_features_path}"); raise typer.Exit(code=1)

    df_train_features = pd.read_csv(train_features_path)
    logger.info(f"Loaded training features ({target_magnification}). Shape: {df_train_features.shape}")

    output_models_dir_mag = output_models_dir / target_magnification # Save models in subfolders per magnification
    output_models_dir_mag.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler(); le = LabelEncoder(); imputer = SimpleImputer(strategy='mean')
    X_train_scaled = pd.DataFrame(); y_train = np.array([])

    if df_train_features.empty: logger.error("Training features empty."); raise typer.Exit(code=1)

    logger.info("--- Preprocessing Training Data ---")
    df_train_clean = df_train_features.copy()
    initial_rows = df_train_clean.shape[0]
    
    nan_count = df_train_clean[FEATURE_COLUMNS].isnull().any(axis=1).sum()
    if nan_count > 0:
        if nan_strategy == 'drop':
            df_train_clean.dropna(subset=FEATURE_COLUMNS, inplace=True)
        elif nan_strategy == 'impute_mean':
            df_train_clean[FEATURE_COLUMNS] = imputer.fit_transform(df_train_clean[FEATURE_COLUMNS])
            joblib.dump(imputer, output_models_dir_mag / f"imputer_{target_magnification}.joblib")
    logger.info(f"Training rows after NaN handling: {df_train_clean.shape[0]} (out of {initial_rows})")

    if df_train_clean.empty: logger.error("Training data empty post-NaN handling."); raise typer.Exit(code=1)
    
    X_train = df_train_clean[FEATURE_COLUMNS]
    y_train = le.fit_transform(df_train_clean['Diagnosis'])
    logger.info(f"Target encoded. Classes: {le.classes_}")
    joblib.dump(le, output_models_dir_mag / f"label_encoder_{target_magnification}.joblib")
    
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=FEATURE_COLUMNS, index=X_train.index)
    logger.info("Training features scaled.")
    joblib.dump(scaler, output_models_dir_mag / f"scaler_{target_magnification}.joblib")

    models_to_train_spec = {
        "SVC": (SVC(probability=True, random_state=random_state, class_weight='balanced'), 
                {'C': [1, 10, 50], 'gamma': ['scale', 0.01, 0.1], 'kernel': ['rbf', 'linear']}), # Reduced grid for CLI speed
        "RandomForest": (RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced_subsample'), {}),
        "LogisticRegression": (LogisticRegression(solver='liblinear', random_state=random_state, class_weight='balanced', max_iter=200), {}),
        "MLP": (MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=random_state, early_stopping=True, n_iter_no_change=10), {}),
        "XGBoost": (xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=random_state, scale_pos_weight=(sum(y_train==0)/sum(y_train==1) if sum(y_train==1)>0 else 1)), {})
    }
    cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    for name, (model, param_grid) in models_to_train_spec.items():
        logger.info(f"--- Training {name} for {target_magnification} ---")
        try:
            if name == "SVC" and param_grid: # Only SVC has grid search for now
                search = GridSearchCV(model, param_grid, cv=cv_stratified, scoring='f1_weighted', verbose=0, n_jobs=-1)
                search.fit(X_train_scaled, y_train)
                logger.info(f"{name} Best Params: {search.best_params_}, Best CV Score: {search.best_score_:.4f}")
                final_model = search.best_estimator_
            else:
                model.fit(X_train_scaled, y_train)
                final_model = model
            
            joblib.dump(final_model, output_models_dir_mag / f"{name.lower()}_model_{target_magnification}.joblib")
            logger.success(f"{name} model for {target_magnification} trained and saved.")
        except Exception as e: logger.error(f"Error training {name} for {target_magnification}: {e}")
    logger.success(f"Model training for {target_magnification} complete.")

if __name__ == "__main__":
    app()