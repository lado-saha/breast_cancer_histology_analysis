import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
import typer

# Use absolute import from the project's src directory
from breast_cancer_histology_analysis.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    SEGMENTATION_CONFIG_FILE,
    DEFAULT_TEST_SET_SIZE,
    DEFAULT_RANDOM_STATE,
    ALL_MAGNIFICATIONS_CONFIG # Added this to config.py for consistency
)

app = typer.Typer()

def _load_segmentation_config(config_path: Path) -> dict:
    """Loads the segmentation configuration from a JSON file."""
    if not config_path.exists():
        logger.error(f"Segmentation config file not found at {config_path}")
        return {} # Return empty dict if not found for tuner context
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Segmentation config loaded successfully from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        return {} # Return empty for tuner context
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}: {e}")
        return {}


def get_all_image_paths_from_config(root_dir: Path, segmentation_config: dict) -> list:
    """
    Finds all relevant image paths based on ALL magnifications defined in segmentation_config.
    This version is for the segmentation_tuner.py or creating a master list.
    """
    all_image_infos = []
    logger.info(f"Scanning directory: {root_dir} for all configured magnifications...")
    
    # Use magnifications from the config, or ALL_MAGNIFICATIONS_CONFIG as a broader default if not specific
    valid_magnifications = list(segmentation_config.get('contour_filters_by_magnification', {}).keys())
    if not valid_magnifications:
        logger.warning(f"No 'contour_filters_by_magnification' found in segmentation_config. Falling back to ALL_MAGNIFICATIONS_CONFIG: {ALL_MAGNIFICATIONS_CONFIG}")
        valid_magnifications = ALL_MAGNIFICATIONS_CONFIG
        if not valid_magnifications:
             logger.error("No valid magnifications to scan. Please check config.")
             return []


    for r_str, _, files in os.walk(root_dir):
        r_path = Path(r_str)
        path_parts = list(r_path.parts)
        
        # Path structure: .../raw/<benign_or_malignant>/<SOB_or_CNB>/<subtype>/<patient_slide_id>/<magnification>/image.png
        # Check if path is deep enough and current directory name is a valid magnification
        if len(path_parts) >= (len(root_dir.parts) + 4) and path_parts[-1] in valid_magnifications and files:
            try:
                magnification = path_parts[-1]
                patient_slide_id = path_parts[-2]
                main_diagnosis_folder_index = len(root_dir.parts) # index of folder after root_dir
                if len(path_parts) > main_diagnosis_folder_index:
                     main_diagnosis_folder = path_parts[main_diagnosis_folder_index]
                else:
                    continue


                if main_diagnosis_folder.lower() not in ['benign', 'malignant']:
                    continue
                
                diagnosis_label = 'M' if main_diagnosis_folder.lower() == 'malignant' else 'B'

                for f_name in files:
                    if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                        all_image_infos.append({
                            'path': str(r_path / f_name),
                            'filename': f_name,
                            'diagnosis': diagnosis_label,
                            'magnification': magnification,
                            'patient_slide_id': patient_slide_id
                        })
            except IndexError:
                # logger.debug(f"Skipping path due to unexpected structure: {r_path}")
                continue
            except Exception as e:
                logger.error(f"Error parsing path info for {r_path}: {e}")

    total_unique_ids = len(set(item['patient_slide_id'] for item in all_image_infos))
    logger.info(f"Found {len(all_image_infos)} total relevant images across configured magnifications from {total_unique_ids} unique patient/slides.")
    return sorted(all_image_infos, key=lambda x: x['path'])


def get_image_paths_for_magnification(root_dir: Path, target_magnification: str, all_image_infos_master: list) -> list:
    """
    Filters a master list of image_infos for a specific target_magnification.
    """
    logger.info(f"Filtering master image list for {target_magnification} images...")
    
    filtered_image_infos = [
        info for info in all_image_infos_master if info['magnification'] == target_magnification
    ]
    
    total_unique_ids_filtered = len(set(item['patient_slide_id'] for item in filtered_image_infos))
    logger.info(f"Found {len(filtered_image_infos)} {target_magnification} images from {total_unique_ids_filtered} unique patient/slides after filtering.")
    return filtered_image_infos # Already sorted if master list was


@app.command()
def create_splits(
    target_magnification: str = typer.Option(..., "--magnification", "-m", help="Target magnification (e.g., '40X', '400X')."),
    root_data_dir: Path = typer.Option(RAW_DATA_DIR, help="Root directory of raw image data."),
    config_path: Path = typer.Option(SEGMENTATION_CONFIG_FILE, help="Path to segmentation_config.json."),
    output_dir: Path = typer.Option(INTERIM_DATA_DIR, help="Directory for output CSV files."),
    test_size: float = typer.Option(DEFAULT_TEST_SET_SIZE, help="Test set proportion."),
    random_state: int = typer.Option(DEFAULT_RANDOM_STATE, help="Random state.")
):
    """
    Gathers ALL images based on config, then filters for TARGET_MAGNIFICATION, 
    splits by patient, and saves train/test CSVs for that magnification.
    """
    logger.info(f"Starting dataset split for magnification: {target_magnification}")
    segmentation_config = _load_segmentation_config(config_path)
    if not segmentation_config: # If loading failed and returned empty dict
        logger.error("Segmentation config is empty or could not be loaded. Exiting.")
        raise typer.Exit(code=1)

    # First, get all image paths based on the full config (for consistent patient ID splitting)
    all_image_infos_master = get_all_image_paths_from_config(root_data_dir, segmentation_config)
    if not all_image_infos_master:
        logger.error("No images found for ANY configured magnification. Exiting.")
        raise typer.Exit(code=1)
    
    df_all_master = pd.DataFrame(all_image_infos_master)
    unique_pids_master = df_all_master['patient_slide_id'].unique()
    patient_diagnoses_master = df_all_master.groupby('patient_slide_id')['diagnosis'].first()

    master_train_ids, master_test_ids = np.array([]), np.array([])
    if len(unique_pids_master) < 2 or \
       (len(unique_pids_master) >= 2 and len(patient_diagnoses_master.loc[unique_pids_master].unique()) < 2):
        logger.warning("Not enough unique patient IDs or diagnosis groups for stratified split across all data. Using random split or assigning all to train for patient ID pool.")
        if len(unique_pids_master) == 1: master_train_ids = unique_pids_master
        elif len(unique_pids_master) > 1 : master_train_ids, master_test_ids = train_test_split(unique_pids_master, test_size=test_size, random_state=random_state)
    else:
        try:
            master_train_ids, master_test_ids = train_test_split(unique_pids_master, test_size=test_size, random_state=random_state, stratify=patient_diagnoses_master.loc[unique_pids_master])
        except ValueError as e:
            logger.warning(f"Stratified split error on master ID list: {e}. Falling to random."); 
            master_train_ids, master_test_ids = train_test_split(unique_pids_master, test_size=test_size, random_state=random_state)
    
    logger.info(f"Master patient ID split: {len(master_train_ids)} train IDs, {len(master_test_ids)} test IDs.")

    # Now filter the master DataFrame for the target magnification and then by train/test IDs
    df_target_mag = df_all_master[df_all_master['magnification'] == target_magnification]
    if df_target_mag.empty:
        logger.error(f"No images found for the TARGET MAGNIFICATION '{target_magnification}' after initial scan. Exiting.")
        raise typer.Exit(code=1)

    df_train = df_target_mag[df_target_mag['patient_slide_id'].isin(master_train_ids)] if master_train_ids.size > 0 else pd.DataFrame()
    df_test = df_target_mag[df_target_mag['patient_slide_id'].isin(master_test_ids)] if master_test_ids.size > 0 else pd.DataFrame()

    logger.info(f"Train images for {target_magnification}: {len(df_train)}")
    logger.info(f"Test images for {target_magnification}: {len(df_test)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output_path = output_dir / f"train_image_info_{target_magnification}.csv"
    test_output_path = output_dir / f"test_image_info_{target_magnification}.csv"
    
    if not df_train.empty: df_train.to_csv(train_output_path, index=False)
    else: logger.warning(f"Train set for {target_magnification} is empty.")
    
    if not df_test.empty: df_test.to_csv(test_output_path, index=False)
    else: logger.warning(f"Test set for {target_magnification} is empty.")
        
    logger.success(f"Dataset splits for {target_magnification} saved to {output_dir}")

if __name__ == "__main__":
    app()