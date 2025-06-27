import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import typer

from breast_cancer_histology_analysis.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, FEATURE_COLUMNS,
    DEFAULT_NAN_HANDLING_STRATEGY
)

app = typer.Typer()

# evaluate_model_performance function (copy from your notebook's Cell 9, slightly adapted)
def evaluate_model_performance_cli(model_name: str, model, X_test_data: pd.DataFrame, y_true_encoded: np.ndarray, 
                               le_encoder, target_magnification: str, figures_dir: Path, reports_dir: Path):
    logger.info(f"--- Eval: {model_name} ({target_magnification}) ---")
    if model is None: logger.warning(f"{model_name} not loaded."); return 0.0,None,None
    if X_test_data.empty or y_true_encoded.size==0 or np.all(y_true_encoded==-1): logger.warning("Test data/labels invalid."); return 0.0,None,None
    
    y_pred, y_prob = None,None
    try:
        y_pred = model.predict(X_test_data)
        if hasattr(model,"predict_proba"): y_prob = model.predict_proba(X_test_data)[:,1]
    except Exception as e: logger.error(f"Predict error {model_name}: {e}"); return 0.0,None,None

    accuracy = accuracy_score(y_true_encoded,y_pred); logger.info(f"Acc: {accuracy:.4f}")
    
    r_path=reports_dir/f"{model_name.lower()}_{target_magnification}_report.txt"
    cm_path=figures_dir/f"{model_name.lower()}_{target_magnification}_cm.png"
    roc_path=figures_dir/f"{model_name.lower()}_{target_magnification}_roc.png"
    
    valid_lbls=[l for l in np.unique(np.concatenate((y_true_encoded,y_pred))) if l!=-1]
    class_rep_str="No valid labels"
    if valid_lbls:
        tgt_names=le_encoder.inverse_transform(valid_lbls)
        class_rep_str=classification_report(y_true_encoded,y_pred,labels=valid_lbls,target_names=tgt_names,zero_division=0)
    logger.info(f"\n{class_rep_str}")
    with open(r_path,'w') as f:f.write(f"Report {model_name} ({target_magnification}):\n{class_rep_str}\nAcc: {accuracy:.4f}\n")
    
    cm_lbls_sk=[l for l in np.unique(y_true_encoded) if l!=-1] or list(le_encoder.transform(le_encoder.classes_))
    cm=None
    if cm_lbls_sk:
        cm=confusion_matrix(y_true_encoded,y_pred,labels=cm_lbls_sk);plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=le_encoder.inverse_transform(cm_lbls_sk),yticklabels=le_encoder.inverse_transform(cm_lbls_sk))
        plt.xlabel('Predicted');plt.ylabel('True');plt.title(f'CM - {model_name} ({target_magnification})');plt.savefig(cm_path);plt.close()
    
    roc_auc=None
    valid_y_roc=y_true_encoded[y_true_encoded!=-1]
    if y_prob is not None and len(np.unique(valid_y_roc))>1:
        y_prob_f=y_prob[y_true_encoded!=-1] if len(y_prob)==len(y_true_encoded) else y_prob
        fpr,tpr,_=roc_curve(valid_y_roc,y_prob_f);roc_auc=auc(fpr,tpr);logger.info(f"ROC AUC: {roc_auc:.4f}")
        plt.figure(figsize=(7,5));plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC (area={roc_auc:.2f})')
        plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--');plt.xlim([0,1]);plt.ylim([0,1.05])
        plt.xlabel('FPR');plt.ylabel('TPR');plt.title(f'ROC - {model_name} ({target_magnification})');plt.legend(loc='lower right');plt.savefig(roc_path);plt.close()
        with open(r_path,'a') as f:f.write(f"ROC AUC: {roc_auc:.4f}\n")
    return accuracy, cm, roc_auc

@app.command()
def predict_magnification(
    target_magnification: str = typer.Option(..., "--magnification", "-m", help="Target magnification features to use."),
    features_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Directory containing test features CSV."),
    models_input_dir: Path = typer.Option(MODELS_DIR, help="Directory of saved models/transformers."),
    nan_strategy: str = typer.Option(DEFAULT_NAN_HANDLING_STRATEGY, help="'drop' or 'impute_mean'.")
):
    """Predicts and evaluates models for a specific magnification."""
    logger.info(f"Starting prediction & evaluation for magnification: {target_magnification}")
    test_features_path = features_dir / f"test_features_{target_magnification}.csv"
    models_dir_mag = models_input_dir / target_magnification
    figures_dir_mag = FIGURES_DIR / target_magnification
    reports_dir_mag = REPORTS_DIR / target_magnification
    figures_dir_mag.mkdir(parents=True, exist_ok=True)
    reports_dir_mag.mkdir(parents=True, exist_ok=True)

    if not test_features_path.exists(): logger.error(f"Test features not found: {test_features_path}"); raise typer.Exit(code=1)
    df_test_features = pd.read_csv(test_features_path)
    
    try:
        scaler = joblib.load(models_dir_mag / f"scaler_{target_magnification}.joblib")
        le = joblib.load(models_dir_mag / f"label_encoder_{target_magnification}.joblib")
        imputer = joblib.load(models_dir_mag / f"imputer_{target_magnification}.joblib") if nan_strategy == 'impute_mean' else None
    except FileNotFoundError as e: logger.error(f"Transformer missing: {e}"); raise typer.Exit(code=1)

    X_test_scaled = pd.DataFrame(); y_test_encoded = np.array([])
    if df_test_features.empty: logger.warning("Test features empty."); raise typer.Exit()

    df_test_clean = df_test_features.copy()
    if df_test_clean[FEATURE_COLUMNS].isnull().any().any():
        if nan_strategy == 'impute_mean' and imputer: df_test_clean[FEATURE_COLUMNS] = imputer.transform(df_test_clean[FEATURE_COLUMNS])
        elif nan_strategy == 'drop': df_test_clean.dropna(subset=FEATURE_COLUMNS, inplace=True)
    
    if df_test_clean.empty: logger.warning("Test data empty post-NaN."); raise typer.Exit()
    X_test = df_test_clean[FEATURE_COLUMNS]
    try: y_test_encoded = le.transform(df_test_clean['Diagnosis'])
    except: y_test_encoded = np.array([-1]*len(df_test_clean['Diagnosis']))
    
    X_test_scaled_np = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=FEATURE_COLUMNS, index=X_test.index)

    model_files = list(models_dir_mag.glob(f"*_model_{target_magnification}.joblib"))
    if not model_files: logger.error(f"No models in {models_dir_mag}"); raise typer.Exit(code=1)

    all_model_performances = []
    for model_path in model_files:
        model_name_full = model_path.stem # e.g., svm_model_400X
        model_name_base = model_name_full.replace(f"_model_{target_magnification}", "").upper()
        model = joblib.load(model_path)
        logger.info(f"Evaluating {model_name_base} for {target_magnification}...")
        acc, _, roc = evaluate_model_performance_cli(model_name_base, model, X_test_scaled, y_test_encoded, le, target_magnification, figures_dir_mag, reports_dir_mag)
        all_model_performances.append({'Magnification': target_magnification, 'Model': model_name_base, 'Accuracy': acc, 'ROC AUC': roc})
    
    df_perf = pd.DataFrame(all_model_performances)
    summary_path = reports_dir_mag.parent / f"model_performance_summary_{target_magnification}.csv" # Save one level up
    df_perf.to_csv(summary_path, index=False)
    logger.success(f"Performance summary for {target_magnification} saved to {summary_path}")
    print(f"\n--- Performance Summary for {target_magnification} ---")
    print(df_perf.to_string(index=False))

if __name__ == "__main__":
    app()