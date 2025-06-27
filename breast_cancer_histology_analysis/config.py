from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger

# --- Project Root ---
# __file__ is .../breast_cancer_histology_analysis/breast_cancer_histology_analysis/config.py
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Project Root (PROJ_ROOT) detected as: {PROJ_ROOT}")

# --- Load .env file from project root ---
dotenv_path = PROJ_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
    logger.info(f".env file loaded from {dotenv_path}")
else:
    logger.info(f".env file not found at {dotenv_path}, skipping dotenv loading.")

# --- Configuration Files Directory ---
CONFIG_DIR = PROJ_ROOT / "config"
SEGMENTATION_CONFIG_FILE = CONFIG_DIR / "segmentation_config.json"

# --- Data Directories ---
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"    # For train/test image info lists
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # For features, predictions
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# --- Models Directory ---
MODELS_DIR = PROJ_ROOT / "models"

# --- Reports and Figures ---
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" # General figures
# Subdirectories for magnification-specific figures will be created by scripts

# --- Create directories if they don't exist (optional, scripts can also do this) ---
for dir_path in [CONFIG_DIR, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, 
                 PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR, 
                 REPORTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# --- TQDM Integration with Loguru ---
try:
    from tqdm import tqdm
    logger.remove(0) 
    logger.add(lambda msg: tqdm.write(msg, end="") if False else print(msg, end=""), colorize=True)
    logger.info("Loguru configured with tqdm.write")
except Exception as e:
    logger.info(f"Could not configure tqdm with Loguru: {e}. Using default logger.")

# --- Global Modeling Parameters (from notebook, can be overridden by CLI args) ---
# These can be accessed by other modules if needed as defaults
DEFAULT_TEST_SET_SIZE = 0.25
DEFAULT_RANDOM_STATE = 42
DEFAULT_NAN_HANDLING_STRATEGY = 'impute_mean'

# --- Feature Column Definitions (from notebook) ---
FEATURE_NAMES_BASE = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                      'compactness', 'concavity', 'concave_points', 'symmetry',
                      'fractal_dimension']
FEATURE_COLUMNS = [f'{name}{suffix}' for name in FEATURE_NAMES_BASE for suffix in ['_mean', '_se', '_max']]

ALL_MAGNIFICATIONS_CONFIG = ['40X', '100X', '200X', '400X'] # Used by dataset.py for robust path gathering if needed