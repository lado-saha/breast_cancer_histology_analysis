import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from skimage import exposure
import math
from loguru import logger
from tqdm import tqdm
import typer
from typing import Optional


from breast_cancer_histology_analysis.config import (
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SEGMENTATION_CONFIG_FILE, FEATURE_COLUMNS, FEATURE_NAMES_BASE
)
from breast_cancer_histology_analysis.utils import get_hematoxylin_channel, calculate_circularity

app = typer.Typer()

# _load_segmentation_config, segment_nuclei_pipeline, calculate_contour_features 
# (These functions are the same as in your updated pipeline_final.ipynb's Cell 3. Copy them here.)
# For brevity, I'm assuming they are copied correctly.
# Make sure segment_nuclei_pipeline and calculate_contour_features use logger for warnings/errors.

def _load_segmentation_config(config_path: Path) -> dict: # Copied from dataset.py for consistency
    if not config_path.exists():
        logger.error(f"Segmentation config file not found: {config_path}"); raise typer.Exit(code=1)
    with open(config_path, 'r') as f: config = json.load(f)
    logger.info(f"Segmentation config loaded from {config_path}")
    return config

# Copy segment_nuclei_pipeline and calculate_contour_features from your notebook's Cell 3 here
# (or the versions I provided previously if they are up-to-date)
# Ensure they are defined at the module level or imported if they are in utils.py fully
# For example:
# from .utils import segment_nuclei_pipeline, calculate_contour_features # If you move them

# --- Start of copied/adapted functions from notebook cell 3 ---
def segment_nuclei_pipeline(image_bgr, magnification: str, config: dict) -> list:
    # ... (Full implementation from your notebook) ...
    if image_bgr is None: return []
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_channel = get_hematoxylin_channel(image_rgb, is_bgr=False)
    if h_channel is None: return []
    h_channel_processed = h_channel.copy()
    if config.get('contrast_stretch', False):
        p_low,p_high=config.get('contrast_percentiles_low',2),config.get('contrast_percentiles_high',98)
        if 0<=p_low<p_high<=100:
            p_low_val,p_high_val=np.percentile(h_channel_processed,(p_low,p_high))
            if p_low_val<p_high_val:
                h_channel_processed=exposure.rescale_intensity(h_channel_processed,in_range=(p_low_val,p_high_val))
                h_channel_processed=cv2.normalize(h_channel_processed,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    if config.get('threshold_method')=='otsu':
        _,binary_mask=cv2.threshold(h_channel_processed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        manual_thresh=config.get('manual_threshold_value',100)
        _,binary_mask=cv2.threshold(h_channel_processed,manual_thresh,255,cv2.THRESH_BINARY)
    open_k_size=config.get('morph_open_kernel_size',3)
    if open_k_size%2==0:open_k_size=max(1,open_k_size-1)
    close_k_size=config.get('morph_close_kernel_size',3)
    if close_k_size%2==0:close_k_size=max(1,close_k_size-1)
    kernel_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_k_size,open_k_size))
    opened_mask=cv2.morphologyEx(binary_mask,cv2.MORPH_OPEN,kernel_open,iterations=config.get('morph_open_iterations',1))
    kernel_close=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_k_size,close_k_size))
    cleaned_mask=cv2.morphologyEx(opened_mask,cv2.MORPH_CLOSE,kernel_close,iterations=config.get('morph_close_iterations',1))
    final_contours_list=[]
    mag_filters_dict=config.get('contour_filters_by_magnification',{})
    mag_specific_filters=mag_filters_dict.get(magnification,mag_filters_dict.get('200X',{}))
    if not mag_specific_filters:
        logger.warning(f"No filters for {magnification}. Using defaults.")
        mag_specific_filters={'min_area':10,'max_area':100000,'min_circularity':0.01,'dist_transform_thresh_ratio':0.3}
    if config.get('use_watershed',False): # As per your config, watershed is False
        sure_bg_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        sure_bg=cv2.dilate(cleaned_mask,sure_bg_kernel,iterations=3)
        dist_transform=cv2.distanceTransform(cleaned_mask,cv2.DIST_L2,5)
        default_global_dt_ratio=config.get('dist_transform_thresh_ratio',0.3)
        current_dist_thresh_ratio=mag_specific_filters.get('dist_transform_thresh_ratio',default_global_dt_ratio)
        _,sure_fg=cv2.threshold(dist_transform,current_dist_thresh_ratio*dist_transform.max(),255,0)
        sure_fg=np.uint8(sure_fg);unknown=cv2.subtract(sure_bg,sure_fg)
        _,markers=cv2.connectedComponents(sure_fg);markers+=1;markers[unknown==255]=0
        try:
            markers_copy=markers.copy();ws_input=image_rgb.copy();cv2.watershed(ws_input,markers_copy)
            for val in np.unique(markers_copy):
                if val<=1:continue
                mask_ws=np.zeros(cleaned_mask.shape,dtype=np.uint8);mask_ws[markers_copy==val]=255
                cnts,_=cv2.findContours(mask_ws,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if cnts:final_contours_list.extend(cnts)
        except cv2.error as e:
            logger.warning(f"Watershed error: {e}. Fallback.");cnts,_=cv2.findContours(cleaned_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);final_contours_list=cnts
    else:
        cnts,_=cv2.findContours(cleaned_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);final_contours_list=cnts
    min_a,max_a,min_c=mag_specific_filters.get('min_area',0),mag_specific_filters.get('max_area',float('inf')),mag_specific_filters.get('min_circularity',0.0)
    return [c for c in final_contours_list if min_a<cv2.contourArea(c)<max_a and calculate_circularity(c)>=min_c]


def calculate_contour_features(contour, gray_image: np.ndarray) -> Optional[dict]:
    # ... (Full implementation from your notebook) ...
    features={};area=cv2.contourArea(contour)
    if area<=0:return None
    try:
        moments=cv2.moments(contour);perimeter=cv2.arcLength(contour,True)
        eq_diam=np.sqrt(4*area/np.pi);features['radius']=eq_diam/2.0
        mask=np.zeros(gray_image.shape,dtype=np.uint8);cv2.drawContours(mask,[contour],-1,255,-1)
        _,std_val=cv2.meanStdDev(gray_image,mask=mask)
        features['texture']=std_val[0][0] if std_val is not None and std_val.size>0 else 0.0
        features['perimeter'],features['area']=perimeter,area
        if moments['m00']>0 and len(contour)>1:
            center=np.array([int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])])
            dist=np.sqrt(np.sum((contour.reshape(-1,2)-center)**2,axis=1))
            features['smoothness']=np.std(dist) if dist.size>0 else 0.0
        else:features['smoothness']=0.0
        features['compactness']=(perimeter**2)/area if area>0 else 0.0
        hull=cv2.convexHull(contour);hull_area=cv2.contourArea(hull)
        features['concavity']=1.0-(area/hull_area) if hull_area>0 else 0.0
        try:
            if len(contour)>3:
                hull_idx=cv2.convexHull(contour,returnPoints=False)
                if hull_idx is not None and len(hull_idx)>3:
                    defects=cv2.convexityDefects(contour,hull_idx);count=0
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            if(defects[i,0][3]/256.0)>(0.05*eq_diam):count+=1
                        features['concave_points']=count
                    else:features['concave_points']=0
                else:features['concave_points']=0
            else:features['concave_points']=0
        except cv2.error:features['concave_points']=0
        if len(contour)>=5:
            try:(_,(ma,MA),_)=cv2.fitEllipse(contour);features['symmetry']=ma/MA if MA>0 else 1.0
            except cv2.error:features['symmetry']=0.0
        else:features['symmetry']=0.0
        features['fractal_dimension']=perimeter/np.sqrt(area) if area>0 else 0.0
        return features
    except Exception as e:logger.warning(f"Feature calc error: {e}");return None
# --- End of copied functions ---

def extract_features_for_image_list_cli(image_info_list: list, segmentation_config: dict, desc="Extracting Features") -> pd.DataFrame:
    # ... (This is your extract_features_for_paths from the notebook, renamed for clarity)
    all_features_data = []
    output_cols = ['ID', 'Diagnosis', 'Magnification', 'PatientSlideID'] + FEATURE_COLUMNS
    for img_info in tqdm(image_info_list, desc=desc):
        img_path, img_id, diagnosis, magnification, patient_id = \
            img_info['path'], img_info['filename'], img_info['diagnosis'], img_info['magnification'], img_info['patient_slide_id']
        row_res = {'ID':img_id, 'Diagnosis':diagnosis, 'Magnification':magnification, 'PatientSlideID':patient_id, **{c:np.nan for c in FEATURE_COLUMNS}}
        try:
            img = cv2.imread(img_path)
            if img is None: all_features_data.append(row_res); continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cnts = segment_nuclei_pipeline(img, magnification, segmentation_config)
            if not cnts: all_features_data.append(row_res); continue
            nuc_feats = [f for c in cnts if (f:=calculate_contour_features(c,gray)) is not None]
            if not nuc_feats: all_features_data.append(row_res); continue
            df_nuc = pd.DataFrame(nuc_feats); num_nuc = len(df_nuc)
            for base_f in FEATURE_NAMES_BASE:
                if base_f in df_nuc.columns:
                    vals = pd.to_numeric(df_nuc[base_f],errors='coerce').dropna()
                    if not vals.empty:
                        row_res[f'{base_f}_mean']=vals.mean()
                        row_res[f'{base_f}_se']=vals.std()/math.sqrt(num_nuc) if num_nuc>1 else 0.0
                        row_res[f'{base_f}_max']=vals.max()
            all_features_data.append(row_res)
        except Exception as e: logger.error(f"Err processing {img_path}: {e}"); all_features_data.append(row_res)
    return pd.DataFrame(all_features_data, columns=output_cols)

@app.command()
def build_magnification_features(
    target_magnification: str = typer.Option(..., "--magnification", "-m", help="Target magnification for feature extraction."),
    config_path: Path = typer.Option(SEGMENTATION_CONFIG_FILE, help="Path to segmentation_config.json."),
    input_dir: Path = typer.Option(INTERIM_DATA_DIR, help="Directory containing train/test image info CSVs."),
    output_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Directory to save output feature CSVs.")
):
    """Extracts features for a specific magnification from train and test image lists."""
    logger.info(f"Starting feature extraction for magnification: {target_magnification}")
    segmentation_config = _load_segmentation_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_type in ["train", "test"]:
        info_csv_path = input_dir / f"{split_type}_image_info_{target_magnification}.csv"
        features_csv_path = output_dir / f"{split_type}_features_{target_magnification}.csv"
        
        if info_csv_path.exists():
            df_info = pd.read_csv(info_csv_path)
            logger.info(f"Loaded {len(df_info)} {split_type} image records for {target_magnification}.")
            if not df_info.empty:
                df_features = extract_features_for_image_list_cli(
                    df_info.to_dict('records'), segmentation_config, 
                    desc=f"{split_type.capitalize()} Set Features ({target_magnification})"
                )
                df_features.to_csv(features_csv_path, index=False)
                logger.success(f"{split_type.capitalize()} features ({target_magnification}) saved. Shape: {df_features.shape}. Path: {features_csv_path}")
            else:
                logger.warning(f"{split_type.capitalize()} image info list for {target_magnification} is empty. No features extracted.")
        else:
            logger.warning(f"{split_type.capitalize()} image info file not found at {info_csv_path}. Skipping.")
    logger.success(f"Feature extraction for {target_magnification} complete.")

if __name__ == "__main__":
    app()