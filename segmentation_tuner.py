import cv2
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from skimage import color, exposure
from scipy import ndimage
import gradio as gr
import json
from collections import defaultdict
from loguru import logger  # Added logger

# --- Set display options for plots (Gradio handles its own display) ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- Project Paths ---
PROJECT_ROOT_DIR_TUNER = pathlib.Path(__file__).parent.resolve()
ROOT_DATA_DIR = PROJECT_ROOT_DIR_TUNER / "data" / "raw"
CONFIG_DIR_TUNER = PROJECT_ROOT_DIR_TUNER / "config"
SAVE_FILE_PATH = CONFIG_DIR_TUNER / "segmentation_config.json"

# Define available magnifications
ALL_MAGNIFICATIONS = ['40X', '100X', '200X', '400X']

# --- Default Configuration (used if no saved file is found or if it's invalid) ---
DEFAULT_INITIAL_SEGMENTATION_CONFIG = {
    'contrast_stretch': False, 'contrast_percentiles_low': 2, 'contrast_percentiles_high': 98,
    'threshold_method': 'otsu', 'manual_threshold_value': 100,
    'morph_open_kernel_size': 3, 'morph_open_iterations': 1,
    'morph_close_kernel_size': 3, 'morph_close_iterations': 1,
    'use_watershed': True, 'dist_transform_thresh_ratio': 0.3,
    'show_intermediate_steps': False,
    'contour_filters_by_magnification': {
        '40X':  {'min_area': 10,   'max_area': 300,  'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.2},
        '100X': {'min_area': 40,  'max_area': 1000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.3},
        '200X': {'min_area': 80, 'max_area': 3000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.5},
        '400X': {'min_area': 200, 'max_area': 7000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.6}
    }
}

# --- Attempt to Load Saved Configuration ---
# Start with default
INITIAL_SEGMENTATION_CONFIG = DEFAULT_INITIAL_SEGMENTATION_CONFIG.copy()
# Ensure config directory exists
CONFIG_DIR_TUNER.mkdir(parents=True, exist_ok=True)

if SAVE_FILE_PATH.exists():
    try:
        with open(SAVE_FILE_PATH, 'r') as f:
            loaded_config = json.load(f)
        # Basic validation: check if essential keys are present
        if "contour_filters_by_magnification" in loaded_config and \
           all(mag in loaded_config["contour_filters_by_magnification"] for mag in ALL_MAGNIFICATIONS):
            INITIAL_SEGMENTATION_CONFIG = loaded_config
            logger.info(
                f"Successfully loaded saved configuration from: {SAVE_FILE_PATH}")
        else:
            logger.warning(
                f"Saved configuration file {SAVE_FILE_PATH} is missing essential keys. Using default config.")

    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from {SAVE_FILE_PATH}. Using default config.")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading config: {e}. Using default config.")
else:
    logger.info(
        f"No saved configuration file found at {SAVE_FILE_PATH}. Using default config and will create the file on first save.")


def get_all_image_paths(root_dir):
    all_image_infos = []
    # valid_magnifications will now get all keys from the (potentially loaded) config
    valid_magnifications = list(
        INITIAL_SEGMENTATION_CONFIG.get('contour_filters_by_magnification', {}).keys())

    for r, _, files in os.walk(root_dir):
        path_parts = pathlib.Path(r).parts
        if len(path_parts) > 1 and path_parts[-1] in valid_magnifications:
            magnification = path_parts[-1]
            for f_name in files:
                if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    all_image_infos.append({'path': os.path.join(r, f_name),
                                            'magnification': magnification,
                                            'filename': f_name})
    if not all_image_infos:
        logger.warning(
            f"No images found in {root_dir} or its subdirectories with valid magnification folders: {', '.join(valid_magnifications)}")
    return sorted(all_image_infos, key=lambda x: x['path'])


def display_image_grid_for_gradio(images, titles, grid_shape, main_title=""):
    # ... (implementation as before) ...
    num_images = len(images)
    if num_images == 0:
        plot_rows, plot_cols = 1, 1
    else:
        plot_cols = max(1, grid_shape[1])
        plot_rows = max(1, (num_images + plot_cols - 1) // plot_cols)
    fig, axes = plt.subplots(plot_rows, plot_cols,
                             figsize=(4 * plot_cols, 4 * plot_rows))
    if plot_rows == 1 and plot_cols == 1:
        flat_axes = [axes]
    elif isinstance(axes, np.ndarray):
        flat_axes = axes.ravel()
    else:
        flat_axes = [axes]
    last_image_idx = -1
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(flat_axes):
            cmap = 'gray' if len(img.shape) == 2 else None
            flat_axes[i].imshow(img, cmap=cmap)
            flat_axes[i].set_title(title, fontsize=8)
            flat_axes[i].axis('off')
            last_image_idx = i
        else:  # pragma: no cover
            logger.warning("Not enough subplots allocated.")
            break
    for j in range(last_image_idx + 1, len(flat_axes)):
        flat_axes[j].axis('off')
    if main_title:
        fig.suptitle(main_title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]
                     if main_title else [0, 0, 1, 0.97])
    plt.close(fig)
    return fig


def get_hematoxylin_channel(image_rgb):
    image_rgb_safe = np.clip(image_rgb, 1, 255)
    ihc_hed = color.rgb2hed(image_rgb_safe)
    h_channel = ihc_hed[:, :, 0]
    return cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def calculate_circularity(contour):
    # ... (implementation as before) ...
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0 or area == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def segment_nuclei_advanced_for_gradio(image_bgr, magnification, current_config, show_intermediate_plot: bool):
    # ... (implementation as before, ensure it uses `current_config` passed to it) ...
    if image_bgr is None:
        return [], None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    vis_images, vis_titles = [], []
    if show_intermediate_plot:
        vis_images.append(image_rgb.copy())
        vis_titles.append('1. Original RGB')
    h_channel = get_hematoxylin_channel(image_rgb)  # Ensure safe call
    if h_channel is None:
        return [], None, image_rgb  # Cannot proceed if H-channel fails
    h_channel_processed = h_channel.copy()
    if show_intermediate_plot:
        vis_images.append(h_channel_processed.copy())
        vis_titles.append('2. H-Channel')
    if current_config.get('contrast_stretch', False):
        p_low, p_high = current_config.get(
            'contrast_percentiles_low', 2), current_config.get('contrast_percentiles_high', 98)
        if 0 <= p_low < p_high <= 100:
            p_low_val, p_high_val = np.percentile(
                h_channel_processed, (p_low, p_high))
            if p_low_val < p_high_val:
                h_channel_processed = exposure.rescale_intensity(
                    h_channel_processed, in_range=(p_low_val, p_high_val))
                h_channel_processed = cv2.normalize(
                    h_channel_processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if show_intermediate_plot:
                    vis_images.append(h_channel_processed.copy())
                    vis_titles.append('2b. H-Stretched')
    if current_config.get('threshold_method') == 'otsu':
        _, binary_mask = cv2.threshold(
            h_channel_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        manual_thresh = current_config.get('manual_threshold_value', 100)
        _, binary_mask = cv2.threshold(
            h_channel_processed, manual_thresh, 255, cv2.THRESH_BINARY)
    if show_intermediate_plot:
        vis_images.append(binary_mask.copy())
        vis_titles.append('3. Binary Mask')
    open_k_size = current_config.get('morph_open_kernel_size', 3)
    if open_k_size % 2 == 0:
        open_k_size = max(1, open_k_size - 1)
    close_k_size = current_config.get('morph_close_kernel_size', 3)
    if close_k_size % 2 == 0:
        close_k_size = max(1, close_k_size-1)
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open,
                                   iterations=current_config.get('morph_open_iterations', 1))
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close,
                                    iterations=current_config.get('morph_close_iterations', 1))
    if show_intermediate_plot:
        vis_images.append(cleaned_mask.copy())
        vis_titles.append('4. Cleaned Mask')
    final_contours_list = []
    mag_filters_all = current_config.get(
        'contour_filters_by_magnification', {})
    mag_specific_filters = mag_filters_all.get(
        magnification, mag_filters_all.get('200X', {}))
    if not mag_specific_filters:
        mag_specific_filters = {'min_area': 10, 'max_area': 100000,
                                'min_circularity': 0.01, 'dist_transform_thresh_ratio': 0.3}

    # Default to False as per your updated config
    if current_config.get('use_watershed', False):
        if show_intermediate_plot:
            vis_images.append(cleaned_mask.copy())
            # Indicate watershed is off
            vis_titles.append('5x. Using Cleaned Mask (Watershed Off)')
        contours_no_ws, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours_list = contours_no_ws
    else:  # This is the current default path
        contours_no_ws, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours_list = contours_no_ws
        if show_intermediate_plot:
            vis_images.append(cleaned_mask.copy())
            vis_titles.append('5. Contours from Cleaned Mask (Watershed Off)')

    min_area = mag_specific_filters.get('min_area', 0)
    max_area = mag_specific_filters.get('max_area', float('inf'))
    min_circ = mag_specific_filters.get('min_circularity', 0.0)
    filtered_contours = [cnt for cnt in final_contours_list if min_area < cv2.contourArea(
        cnt) < max_area and calculate_circularity(cnt) >= min_circ]
    image_final_contours_display = image_rgb.copy()
    cv2.drawContours(image_final_contours_display,
                     filtered_contours, -1, (0, 255, 0), 1)
    plot_fig = None
    if show_intermediate_plot and vis_images:
        num_vis_images = len(vis_images)
        grid_cols_display = 3
        grid_rows_display = (
            num_vis_images + grid_cols_display - 1) // grid_cols_display
        plot_fig = display_image_grid_for_gradio(
            vis_images, vis_titles, grid_shape=(grid_rows_display, grid_cols_display))
    return filtered_contours, plot_fig, image_final_contours_display


# --- Data Loading ---
PATIENT_SLIDE_IMAGES = defaultdict(
    lambda: {mag: None for mag in ALL_MAGNIFICATIONS})
# Uses the (potentially loaded) INITIAL_SEGMENTATION_CONFIG
ALL_RAW_IMAGE_INFOS = get_all_image_paths(ROOT_DATA_DIR)

if not ALL_RAW_IMAGE_INFOS:
    logger.error(
        f"CRITICAL ERROR: No images found. Check ROOT_DATA_DIR and segmentation config.")
else:
    for img_info in ALL_RAW_IMAGE_INFOS:
        # ... (Patient/Slide ID parsing logic remains the same) ...
        parts = img_info['filename'].split('-')
        if len(parts) >= 4:
            try:
                filename_no_ext = os.path.splitext(img_info['filename'])[0]
                last_mag_part_idx = -1
                for mag_key_iter in ALL_MAGNIFICATIONS:
                    mag_str_part_search = f"-{mag_key_iter.replace('X', '')}-"
                    try:
                        idx = filename_no_ext.rindex(mag_str_part_search)
                        if idx > last_mag_part_idx:
                            last_mag_part_idx = idx
                        break
                    except ValueError:
                        continue
                if last_mag_part_idx != -1:
                    patient_slide_id = filename_no_ext[:last_mag_part_idx]
                else:
                    patient_slide_id = "-".join(parts[:-2])
                if img_info['magnification'] in PATIENT_SLIDE_IMAGES[patient_slide_id]:
                    PATIENT_SLIDE_IMAGES[patient_slide_id][img_info['magnification']
                                                           ] = img_info['path']
            except Exception as e:
                logger.error(
                    f"Could not parse patient_slide_id from {img_info['filename']}: {e}")
UNIQUE_PATIENT_SLIDE_IDS = sorted(list(PATIENT_SLIDE_IMAGES.keys()))

# --- Gradio UI Definition ---
# process_patient_slide and navigate_patient_id functions remain the same
# ... (process_patient_slideAction and navigate_patient_idAction definitions) ...


def process_patient_slideAction(
    selected_patient_id, selected_magnification, show_intermediate_steps_value,
    contrast_stretch, contrast_p_low, contrast_p_high, threshold_method, manual_threshold_value,
    morph_open_kernel, morph_open_iter, morph_close_kernel, morph_close_iter, use_watershed,
    min_area_40X, max_area_40X, min_circ_40X, dt_ratio_40X,
    min_area_100X, max_area_100X, min_circ_100X, dt_ratio_100X,
    min_area_200X, max_area_200X, min_circ_200X, dt_ratio_200X,
    min_area_400X, max_area_400X, min_circ_400X, dt_ratio_400X
):
    # ... (implementation as before, ensure INITIAL_SEGMENTATION_CONFIG in empty_plot case is the loaded/default one) ...
    if not selected_patient_id or not PATIENT_SLIDE_IMAGES or selected_patient_id not in PATIENT_SLIDE_IMAGES:
        empty_plot = None
        if show_intermediate_steps_value:
            empty_plot = display_image_grid_for_gradio(
                [], [], (1, 1), "No Data")
        # Use INITIAL_SEGMENTATION_CONFIG which is now the loaded or default one
        return selected_patient_id, "No Patient ID or images.", None, empty_plot, None, json.dumps(INITIAL_SEGMENTATION_CONFIG, indent=2)

    live_config = {
        'contrast_stretch': contrast_stretch, 'contrast_percentiles_low': int(contrast_p_low),
        'contrast_percentiles_high': int(contrast_p_high), 'threshold_method': threshold_method,
        'manual_threshold_value': int(manual_threshold_value), 'morph_open_kernel_size': int(morph_open_kernel),
        'morph_open_iterations': int(morph_open_iter), 'morph_close_kernel_size': int(morph_close_kernel),
        'morph_close_iterations': int(morph_close_iter), 'use_watershed': use_watershed,
        'dist_transform_thresh_ratio': INITIAL_SEGMENTATION_CONFIG.get('dist_transform_thresh_ratio', 0.3),
        'contour_filters_by_magnification': {
            '40X': {'min_area': int(min_area_40X), 'max_area': int(max_area_40X), 'min_circularity': float(min_circ_40X), 'dist_transform_thresh_ratio': float(dt_ratio_40X)},
            '100X': {'min_area': int(min_area_100X), 'max_area': int(max_area_100X), 'min_circularity': float(min_circ_100X), 'dist_transform_thresh_ratio': float(dt_ratio_100X)},
            '200X': {'min_area': int(min_area_200X), 'max_area': int(max_area_200X), 'min_circularity': float(min_circ_200X), 'dist_transform_thresh_ratio': float(dt_ratio_200X)},
            '400X': {'min_area': int(min_area_400X), 'max_area': int(max_area_400X), 'min_circularity': float(min_circ_400X), 'dist_transform_thresh_ratio': float(dt_ratio_400X)}
        }
    }
    output_orig_img, output_plot_fig, output_final_img = None, None, None
    processing_status_info = f"{selected_magnification} (Not processed)"
    patient_data = PATIENT_SLIDE_IMAGES.get(selected_patient_id)
    if patient_data:
        image_path = patient_data.get(selected_magnification)
        if image_path and os.path.exists(image_path):  # Check if path exists
            image_bgr = cv2.imread(image_path)
            if image_bgr is not None:
                _, plot_fig, final_img = segment_nuclei_advanced_for_gradio(
                    image_bgr, selected_magnification, live_config, show_intermediate_plot=show_intermediate_steps_value)
                output_orig_img, output_plot_fig, output_final_img = cv2.cvtColor(
                    image_bgr, cv2.COLOR_BGR2RGB), plot_fig, final_img
                processing_status_info = selected_magnification
            else:
                processing_status_info = f"{selected_magnification} (Error loading image)"
        else:
            processing_status_info = f"{selected_magnification} (Image path not found/invalid)"
    info_text = f"Patient: {selected_patient_id}\nProcessing: {processing_status_info}"
    # If plot is None (e.g. show_intermediate_steps is False), Gradio handles it.
    # If image loading failed, plot_fig might be None.
    if output_plot_fig is None and show_intermediate_steps_value:  # Create empty plot if needed
        output_plot_fig = display_image_grid_for_gradio(
            [], [], (1, 1), processing_status_info)

    # Return live_config as dict
    return [selected_patient_id, info_text, output_orig_img, output_plot_fig, output_final_img, live_config]


def navigate_patient_idAction(current_id, direction, id_list):
    # ... (implementation as before) ...
    if not id_list:
        return None
    if current_id not in id_list:
        return id_list[0] if id_list else None
    current_idx = id_list.index(current_id)
    if direction == "next":
        new_idx = (current_idx + 1) % len(id_list)
    elif direction == "prev":
        new_idx = (current_idx - 1 + len(id_list)) % len(id_list)
    else:
        return current_id
    return id_list[new_idx]


with gr.Blocks(theme=gr.themes.Ocean(), title=f"Nuclei Segmentation Tuner") as app:
    gr.Markdown(f"# Interactive Nuclei Segmentation Tuner")
    initial_patient_id = UNIQUE_PATIENT_SLIDE_IDS[0] if UNIQUE_PATIENT_SLIDE_IDS else None
    selected_patient_id_state = gr.State(initial_patient_id)
    all_ids_state = gr.State(UNIQUE_PATIENT_SLIDE_IDS)
    initial_magnification = ALL_MAGNIFICATIONS[3]  # Default to 400X

    with gr.Row():
        with gr.Column(scale=1, elem_id="control-panel", min_width=350):
            gr.Markdown("## Controls")
            patient_id_dropdown = gr.Dropdown(
                label="Select Patient/Slide ID", choices=UNIQUE_PATIENT_SLIDE_IDS, value=initial_patient_id)
            with gr.Row():
                prev_btn = gr.Button("Previous Patient")
                next_btn = gr.Button("Next Patient")

            magnification_selector_radio = gr.Radio(
                label="Select Magnification to Process",
                choices=ALL_MAGNIFICATIONS,
                # Use value from (potentially loaded) INITIAL_SEGMENTATION_CONFIG if you add a key for it,
                # or stick to a fixed default like initial_magnification
                value=initial_magnification
            )
            # Initialize checkbox from (potentially loaded) INITIAL_SEGMENTATION_CONFIG
            show_intermediate_steps_check = gr.Checkbox(
                label="Show Intermediate Steps",
                value=INITIAL_SEGMENTATION_CONFIG.get(
                    'show_intermediate_steps', False)
            )
            image_info_display = gr.Textbox(
                label="Processing Info", interactive=False, lines=2)
            gr.Markdown("---")
            gr.Markdown("### Segmentation Parameters (Loaded/Default)")

            # UI components initialized from INITIAL_SEGMENTATION_CONFIG
            with gr.Accordion("Global Settings", open=True):
                contrast_stretch_check = gr.Checkbox(
                    label="Contrast Stretch H-Channel", value=INITIAL_SEGMENTATION_CONFIG['contrast_stretch'])
                use_watershed_check = gr.Checkbox(
                    label="Use Watershed", value=INITIAL_SEGMENTATION_CONFIG['use_watershed'])
                with gr.Row():
                    contrast_p_low_slider = gr.Slider(
                        label="Contrast Low %", minimum=0, maximum=49, step=1, value=INITIAL_SEGMENTATION_CONFIG['contrast_percentiles_low'])
                    contrast_p_high_slider = gr.Slider(
                        label="Contrast High %", minimum=51, maximum=100, step=1, value=INITIAL_SEGMENTATION_CONFIG['contrast_percentiles_high'])
                threshold_method_radio = gr.Radio(label="Threshold Method", choices=[
                                                  'otsu', 'manual'], value=INITIAL_SEGMENTATION_CONFIG['threshold_method'])
                manual_thresh_slider = gr.Slider(label="Manual Threshold", minimum=0, maximum=255, step=1, value=INITIAL_SEGMENTATION_CONFIG['manual_threshold_value'],
                                                 interactive=(INITIAL_SEGMENTATION_CONFIG['threshold_method'] == 'manual'))

                def update_manual_thresh_interactiveAction(
                    method): return gr.Slider(interactive=(method == 'manual'))
                threshold_method_radio.change(fn=update_manual_thresh_interactiveAction,
                                              inputs=threshold_method_radio, outputs=manual_thresh_slider, show_progress="hidden")

            with gr.Accordion("Morphological Operations", open=False):
                morph_open_k_slider = gr.Slider(
                    label="Open Kernel", minimum=1, maximum=15, step=2, value=INITIAL_SEGMENTATION_CONFIG['morph_open_kernel_size'])
                morph_open_iter_slider = gr.Slider(
                    label="Open Iterations", minimum=1, maximum=5, step=1, value=INITIAL_SEGMENTATION_CONFIG['morph_open_iterations'])
                morph_close_k_slider = gr.Slider(
                    label="Close Kernel", minimum=1, maximum=15, step=2, value=INITIAL_SEGMENTATION_CONFIG['morph_close_kernel_size'])
                morph_close_iter_slider = gr.Slider(
                    label="Close Iterations", minimum=1, maximum=5, step=1, value=INITIAL_SEGMENTATION_CONFIG['morph_close_iterations'])

            mag_ui_elements = {}
            for mag_idx, mag_key in enumerate(ALL_MAGNIFICATIONS):
                # Ensure contour_filters_by_magnification and specific mag_key exist, otherwise use defaults
                conf_mag_specific = INITIAL_SEGMENTATION_CONFIG.get('contour_filters_by_magnification', {}).get(mag_key,
                                                                                                                # Fallback to default if key missing
                                                                                                                DEFAULT_INITIAL_SEGMENTATION_CONFIG['contour_filters_by_magnification'][mag_key])

                with gr.Accordion(f"Filters & Seed Ratio for {mag_key}", open=(mag_key == initial_magnification)):
                    mag_ui_elements[f'min_area_{mag_key}'] = gr.Slider(
                        label="Min Area", minimum=1, maximum=10000, step=1, value=conf_mag_specific['min_area'])
                    mag_ui_elements[f'max_area_{mag_key}'] = gr.Slider(
                        label="Max Area", minimum=100, maximum=50000, step=10, value=conf_mag_specific['max_area'])
                    mag_ui_elements[f'min_circ_{mag_key}'] = gr.Slider(
                        label="Min Circularity", minimum=0.0, maximum=1.0, step=0.01, value=conf_mag_specific['min_circularity'])
                    mag_ui_elements[f'dt_ratio_{mag_key}'] = gr.Slider(
                        label="Watershed Seed Ratio", minimum=0.01, maximum=0.99, step=0.01, value=conf_mag_specific['dist_transform_thresh_ratio'])

            gr.Markdown("---")
            gr.Markdown("### Configuration Management")
            with gr.Row():
                save_config_btn = gr.Button(
                    "Save Current Settings to File", variant="primary")
            # current_config_textbox displays the live_config from process_patient_slide
            current_config_textbox = gr.JSON(
                label="Current Processed Config (Live)", visible=True)
            # saved_config_textbox shows the content of the JSON file if it was loaded
            saved_config_display = gr.JSON(
                label="Config from File (on Load/Save)", value=INITIAL_SEGMENTATION_CONFIG, visible=True)

        with gr.Column(scale=3, elem_id="display-panel"):
            gr.Markdown(f"## Segmentation Results")
            output_ui = {}  # Original, Plot, Final
            with gr.Row():
                output_ui['orig_img'] = gr.Image(
                    label="Original Image", type="numpy", height=400, interactive=False, scale=1)
                output_ui['final_img'] = gr.Image(
                    label="Segmented Image", type="numpy", height=400, interactive=False, scale=1)
            intermediate_steps_plot = gr.Plot(label="Intermediate Steps",
                                              visible=INITIAL_SEGMENTATION_CONFIG.get(
                                                  'show_intermediate_steps', False),
                                              elem_id="plot-container")

    param_inputs_from_global_and_morph = [
        contrast_stretch_check, contrast_p_low_slider, contrast_p_high_slider,
        threshold_method_radio, manual_thresh_slider,
        morph_open_k_slider, morph_open_iter_slider,
        morph_close_k_slider, morph_close_iter_slider,
        use_watershed_check,
    ]
    param_inputs_from_mag_filters = []
    for mag_key in ALL_MAGNIFICATIONS:
        param_inputs_from_mag_filters.extend([
            mag_ui_elements[f'min_area_{mag_key}'], mag_ui_elements[f'max_area_{mag_key}'],
            mag_ui_elements[f'min_circ_{mag_key}'], mag_ui_elements[f'dt_ratio_{mag_key}']
        ])

    event_inputs_for_processing = [
        patient_id_dropdown, magnification_selector_radio, show_intermediate_steps_check
    ] + param_inputs_from_global_and_morph + param_inputs_from_mag_filters

    all_outputs_for_processing = [
        selected_patient_id_state, image_info_display,
        output_ui['orig_img'], intermediate_steps_plot, output_ui['final_img'],
        current_config_textbox  # This will now display the live_config dictionary
    ]

    all_event_triggers = [patient_id_dropdown, magnification_selector_radio, show_intermediate_steps_check] + \
        param_inputs_from_global_and_morph + param_inputs_from_mag_filters

    for trigger_component in all_event_triggers:
        trigger_component.change(
            fn=process_patient_slideAction, inputs=event_inputs_for_processing,
            outputs=all_outputs_for_processing, show_progress="full")

    def toggle_plot_visibilityAction(
        show_plot_val): return gr.Plot(visible=show_plot_val)
    show_intermediate_steps_check.change(fn=toggle_plot_visibilityAction, inputs=[
                                         show_intermediate_steps_check], outputs=[intermediate_steps_plot], show_progress="hidden")

    prev_btn.click(fn=navigate_patient_idAction, inputs=[patient_id_dropdown, gr.State(
        "prev"), all_ids_state], outputs=[patient_id_dropdown])
    next_btn.click(fn=navigate_patient_idAction, inputs=[patient_id_dropdown, gr.State(
        "next"), all_ids_state], outputs=[patient_id_dropdown])

    def save_config_to_file_action(config_to_save):
        # config_to_save is the dictionary from current_config_textbox (live_config)
        if not isinstance(config_to_save, dict):
            logger.error("Config to save is not a dictionary. Aborting save.")
            return config_to_save  # Return current state of saved_config_display

        # Add/Update the 'show_intermediate_steps' value from the checkbox to the config being saved
        # This requires passing show_intermediate_steps_check.value to this function.
        # For simplicity, let's assume it's part of config_to_save if you include it in live_config.
        # Or, fetch it from the UI component if possible, but Gradio functions are best with explicit inputs.
        # A cleaner way: `process_patient_slideAction` already forms `live_config`.
        # We could make `live_config` also include the `show_intermediate_steps` value.
        # Let's assume for now `live_config` passed to `config_to_save` IS what we want to save.

        try:
            CONFIG_DIR_TUNER.mkdir(
                parents=True, exist_ok=True)  # Ensure dir exists
            with open(SAVE_FILE_PATH, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(
                f"Configuration successfully saved to {SAVE_FILE_PATH}")
            return config_to_save  # Update the "Saved Config" display with what was just saved
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            # Optionally, return an error message or the previous state of saved_config_display
            error_msg = {"error": f"Failed to save config: {str(e)}"}
            return error_msg

    save_config_btn.click(fn=save_config_to_file_action, inputs=[
                          current_config_textbox], outputs=[saved_config_display])

    initial_load_inputs = [gr.State(initial_patient_id), gr.State(initial_magnification), show_intermediate_steps_check] + \
        param_inputs_from_global_and_morph + param_inputs_from_mag_filters

    app.load(fn=process_patient_slideAction,
             inputs=initial_load_inputs, outputs=all_outputs_for_processing)

if __name__ == '__main__':
    if not ALL_RAW_IMAGE_INFOS or not UNIQUE_PATIENT_SLIDE_IDS:
        logger.critical(
            "No images or patient IDs found. UI might not function as expected.")
    else:
        logger.info(
            f"Found {len(ALL_RAW_IMAGE_INFOS)} images, {len(UNIQUE_PATIENT_SLIDE_IDS)} unique patient/slide IDs. Launching Gradio UI...")
    app.launch(debug=True, share=False)
