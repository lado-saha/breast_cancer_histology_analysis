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

# --- Set display options for plots (Gradio handles its own display) ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- Configuration (Restored for all magnifications) ---
ROOT_DATA_DIR = './raw/'
SAVE_FILE_PATH = './config/segmentation_config.json'

# Define available magnifications
ALL_MAGNIFICATIONS = ['40X', '100X', '200X', '400X']
INITIAL_SEGMENTATION_CONFIG = {
    'contrast_stretch': False, 'contrast_percentiles_low': 2, 'contrast_percentiles_high': 98,
    'threshold_method': 'otsu', 'manual_threshold_value': 100,
    'morph_open_kernel_size': 3, 'morph_open_iterations': 1,
    'morph_close_kernel_size': 3, 'morph_close_iterations': 1,
    'use_watershed': True, 'dist_transform_thresh_ratio': 0.3,
    ' show_intermediate_steps': False,
    'contour_filters_by_magnification': {
        '40X':  {'min_area': 10,   'max_area': 300,  'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.2},
        '100X': {'min_area': 40,  'max_area': 1000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.3},
        '200X': {'min_area': 80, 'max_area': 3000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.5},
        '400X': {'min_area': 200, 'max_area': 7000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.6}
    }
}


def get_all_image_paths(root_dir):
    all_image_infos = []
    # valid_magnifications will now get all keys from the restored config
    valid_magnifications = list(
        INITIAL_SEGMENTATION_CONFIG['contour_filters_by_magnification'].keys())

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
        print(
            f"Warning: No images found in {root_dir} or its subdirectories with valid magnification folders: {', '.join(valid_magnifications)}")
    return sorted(all_image_infos, key=lambda x: x['path'])


def display_image_grid_for_gradio(images, titles, grid_shape, main_title=""):
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
            print("Warning: Not enough subplots allocated.")
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
    ihc_hed = color.rgb2hed(image_rgb)
    h_channel = ihc_hed[:, :, 0]
    return cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def calculate_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0 or area == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def segment_nuclei_advanced_for_gradio(image_bgr, magnification, current_config, show_intermediate_plot: bool):
    if image_bgr is None:
        return [], None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    vis_images, vis_titles = [], []
    if show_intermediate_plot:
        vis_images.append(image_rgb.copy())
        vis_titles.append('1. Original RGB')
    h_channel = get_hematoxylin_channel(image_rgb)
    h_channel_processed = h_channel.copy()
    if show_intermediate_plot:
        vis_images.append(h_channel_processed.copy())
        vis_titles.append('2. H-Channel')
    if current_config.get('contrast_stretch', False):
        p_low, p_high = current_config['contrast_percentiles_low'], current_config['contrast_percentiles_high']
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
    if current_config['threshold_method'] == 'otsu':
        _, binary_mask = cv2.threshold(
            h_channel_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_mask = cv2.threshold(
            h_channel_processed, current_config['manual_threshold_value'], 255, cv2.THRESH_BINARY)
    if show_intermediate_plot:
        vis_images.append(binary_mask.copy())
        vis_titles.append('3. Binary Mask')
    open_k_size = current_config['morph_open_kernel_size']
    if open_k_size % 2 == 0:
        open_k_size += 1
    close_k_size = current_config['morph_close_kernel_size']
    if close_k_size % 2 == 0:
        close_k_size += 1
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
    opened_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=current_config['morph_open_iterations'])
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))
    cleaned_mask = cv2.morphologyEx(
        opened_mask, cv2.MORPH_CLOSE, kernel_close, iterations=current_config['morph_close_iterations'])
    if show_intermediate_plot:
        vis_images.append(cleaned_mask.copy())
        vis_titles.append('4. Cleaned Mask')
    final_contours_list = []
    # This is key: magnification argument selects the correct sub-config
    mag_specific_filters = current_config['contour_filters_by_magnification'][magnification]
    if current_config.get('use_watershed', True):
        sure_bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(cleaned_mask, sure_bg_kernel, iterations=3)
        if show_intermediate_plot:
            vis_images.append(sure_bg.copy())
            vis_titles.append('5a. Sure BG')
        dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        if show_intermediate_plot:
            dist_vis = cv2.normalize(
                dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
            vis_images.append(dist_vis.copy())
            vis_titles.append('5b. Distance Tx')
        default_global_dt_ratio = current_config.get(
            'dist_transform_thresh_ratio', 0.3)  # Global fallback
        current_dist_thresh_ratio = mag_specific_filters.get(
            'dist_transform_thresh_ratio', default_global_dt_ratio)
        _, sure_fg = cv2.threshold(
            dist_transform, current_dist_thresh_ratio * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        if show_intermediate_plot:
            vis_images.append(sure_fg.copy())
            vis_titles.append('5c. Sure FG')
        unknown = cv2.subtract(sure_bg, sure_fg)
        if show_intermediate_plot:
            vis_images.append(unknown.copy())
            vis_titles.append('5d. Unknown')
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        watershed_input_img = image_rgb.copy()
        try:
            markers_watershed = cv2.watershed(
                watershed_input_img, markers.copy())
            if show_intermediate_plot:
                watershed_lines_vis = image_rgb.copy()
                watershed_lines_vis[markers_watershed == -1] = [255, 0, 0]
                vis_images.append(watershed_lines_vis)
                vis_titles.append('5e. Watershed Lines')
            unique_marker_values = np.unique(markers_watershed)
            for marker_val in unique_marker_values:
                if marker_val <= 1:
                    continue
                nucleus_mask_ws = np.zeros(cleaned_mask.shape, dtype=np.uint8)
                nucleus_mask_ws[markers_watershed == marker_val] = 255
                contours, _ = cv2.findContours(
                    nucleus_mask_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    final_contours_list.extend(contours)
        except cv2.error:  # pragma: no cover
            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_contours_list = contours
    else:
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours_list = contours
    min_area = mag_specific_filters['min_area']
    max_area = mag_specific_filters['max_area']
    min_circ = mag_specific_filters['min_circularity']
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


# PATIENT_SLIDE_IMAGES will now hold paths for all magnifications
PATIENT_SLIDE_IMAGES = defaultdict(
    lambda: {mag: None for mag in ALL_MAGNIFICATIONS})
ALL_RAW_IMAGE_INFOS = get_all_image_paths(ROOT_DATA_DIR)

if not ALL_RAW_IMAGE_INFOS:
    print(f"CRITICAL ERROR: No images found for any configured magnification.")
else:
    for img_info in ALL_RAW_IMAGE_INFOS:
        parts = img_info['filename'].split('-')
        if len(parts) >= 4:
            try:
                filename_no_ext = os.path.splitext(img_info['filename'])[0]
                last_mag_part_idx = -1
                # Try to find any of the magnification patterns in the filename
                for mag_key_iter in ALL_MAGNIFICATIONS:
                    mag_str_part_search = f"-{mag_key_iter.replace('X', '')}-"
                    try:
                        idx = filename_no_ext.rindex(mag_str_part_search)
                        if idx > last_mag_part_idx:
                            last_mag_part_idx = idx
                        break  # Found one, assume it's the correct one
                    except ValueError:
                        continue

                if last_mag_part_idx != -1:
                    patient_slide_id = filename_no_ext[:last_mag_part_idx]
                else:  # Fallback if no specific mag pattern like "-100-" found
                    patient_slide_id = "-".join(parts[:-2])

                # Ensure the magnification from directory matches the one in PATIENT_SLIDE_IMAGES keys
                if img_info['magnification'] in PATIENT_SLIDE_IMAGES[patient_slide_id]:
                    PATIENT_SLIDE_IMAGES[patient_slide_id][img_info['magnification']
                                                           ] = img_info['path']
                else:  # pragma: no cover
                    print(
                        f"Warning: Image {img_info['filename']} has magnification {img_info['magnification']} not in ALL_MAGNIFICATIONS for PATIENT_SLIDE_IMAGES structure.")

            except Exception as e:  # pragma: no cover
                print(
                    f"Could not parse patient_slide_id from {img_info['filename']}: {e}")
                continue
UNIQUE_PATIENT_SLIDE_IDS = sorted(list(PATIENT_SLIDE_IMAGES.keys()))

# MODIFIED: process_patient_slide


def process_patient_slide(
    selected_patient_id,
    selected_magnification,  # New input: current magnification to process
    show_intermediate_steps_value,
    contrast_stretch, contrast_p_low, contrast_p_high,
    threshold_method, manual_threshold_value,
    morph_open_kernel, morph_open_iter,
    morph_close_kernel, morph_close_iter,
    use_watershed,
    # All mag-specific filter inputs are passed
    min_area_40X, max_area_40X, min_circ_40X, dt_ratio_40X,
    min_area_100X, max_area_100X, min_circ_100X, dt_ratio_100X,
    min_area_200X, max_area_200X, min_circ_200X, dt_ratio_200X,
    min_area_400X, max_area_400X, min_circ_400X, dt_ratio_400X
):
    if not selected_patient_id or not PATIENT_SLIDE_IMAGES or selected_patient_id not in PATIENT_SLIDE_IMAGES:
        empty_plot = None
        if show_intermediate_steps_value:
            empty_plot = display_image_grid_for_gradio(
                [], [], (1, 1), "No Data / Invalid Patient ID")
        return selected_patient_id, f"No Patient ID selected or invalid.", None, empty_plot, None, json.dumps(INITIAL_SEGMENTATION_CONFIG, indent=2)

    # Construct live_config WITH ALL magnification filter settings from UI
    # segment_nuclei_advanced_for_gradio will pick the one matching 'selected_magnification'
    live_config = {
        'contrast_stretch': contrast_stretch, 'contrast_percentiles_low': int(contrast_p_low),
        'contrast_percentiles_high': int(contrast_p_high), 'threshold_method': threshold_method,
        'manual_threshold_value': int(manual_threshold_value), 'morph_open_kernel_size': int(morph_open_kernel),
        'morph_open_iterations': int(morph_open_iter), 'morph_close_kernel_size': int(morph_close_kernel),
        'morph_close_iterations': int(morph_close_iter), 'use_watershed': use_watershed,
        # Global fallback
        'dist_transform_thresh_ratio': INITIAL_SEGMENTATION_CONFIG['dist_transform_thresh_ratio'],
        'contour_filters_by_magnification': {
            '40X':  {'min_area': int(min_area_40X),   'max_area': int(max_area_40X),  'min_circularity': float(min_circ_40X), 'dist_transform_thresh_ratio': float(dt_ratio_40X)},
            '100X': {'min_area': int(min_area_100X),  'max_area': int(max_area_100X), 'min_circularity': float(min_circ_100X), 'dist_transform_thresh_ratio': float(dt_ratio_100X)},
            '200X': {'min_area': int(min_area_200X), 'max_area': int(max_area_200X), 'min_circularity': float(min_circ_200X), 'dist_transform_thresh_ratio': float(dt_ratio_200X)},
            '400X': {'min_area': int(min_area_400X), 'max_area': int(max_area_400X), 'min_circularity': float(min_circ_400X), 'dist_transform_thresh_ratio': float(dt_ratio_400X)}
        }
    }

    output_orig_img = None
    output_plot_fig = None
    output_final_img = None
    processing_status_info = f"{selected_magnification} (Not processed)"

    patient_data = PATIENT_SLIDE_IMAGES.get(selected_patient_id)
    if patient_data:
        # Get image path for the SELECTED magnification
        image_path = patient_data.get(selected_magnification)
        if image_path:
            image_bgr = cv2.imread(image_path)
            if image_bgr is not None:
                _, plot_fig, final_img = segment_nuclei_advanced_for_gradio(
                    image_bgr,
                    selected_magnification,  # Pass the currently selected magnification
                    live_config,
                    show_intermediate_plot=show_intermediate_steps_value
                )
                original_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                output_orig_img, output_plot_fig, output_final_img = original_img_rgb, plot_fig, final_img
                processing_status_info = selected_magnification
            else:  # pragma: no cover
                processing_status_info = f"{selected_magnification} (Error loading image: {os.path.basename(image_path)})"
                if show_intermediate_steps_value:
                    output_plot_fig = display_image_grid_for_gradio(
                        [], [], (1, 1), processing_status_info)
        else:  # pragma: no cover
            processing_status_info = f"{selected_magnification} (Image not found for patient at this magnification)"
            if show_intermediate_steps_value:
                output_plot_fig = display_image_grid_for_gradio(
                    [], [], (1, 1), processing_status_info)

    info_text = f"Patient: {selected_patient_id}\nProcessing: {processing_status_info}"
    return [selected_patient_id, info_text, output_orig_img, output_plot_fig, output_final_img, json.dumps(live_config, indent=2)]


def navigate_patient_id(current_id, direction, id_list):
    if not id_list:
        return None
    if current_id not in id_list:
        return id_list[0]
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

    # Default magnification to process
    # Default to 400X, or choose another
    initial_magnification = ALL_MAGNIFICATIONS[3]

    with gr.Row():
        with gr.Column(scale=1, elem_id="control-panel", min_width=350):
            gr.Markdown("## Controls")
            patient_id_dropdown = gr.Dropdown(
                label="Select Patient/Slide ID", choices=UNIQUE_PATIENT_SLIDE_IDS, value=initial_patient_id)
            with gr.Row():
                prev_btn = gr.Button("Previous Patient")
                next_btn = gr.Button("Next Patient")

            # --- Magnification Selector ---
            magnification_selector_radio = gr.Radio(
                label="Select Magnification to Process",
                choices=ALL_MAGNIFICATIONS,
                value=initial_magnification
            )

            show_intermediate_steps_check = gr.Checkbox(
                label="Show Intermediate Steps", value=INITIAL_SEGMENTATION_CONFIG.get('show_intermediate_steps', False))
            image_info_display = gr.Textbox(
                label="Processing Info", interactive=False, lines=2)
            gr.Markdown("---")
            gr.Markdown("### Segmentation Parameters")

            with gr.Accordion("Global Settings", open=True):
                # ... (Global settings UI remains the same) ...
                with gr.Row():
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
                manual_thresh_slider = gr.Slider(label="Manual Threshold (if manual)", minimum=0, maximum=255, step=1,
                                                 value=INITIAL_SEGMENTATION_CONFIG['manual_threshold_value'], interactive=(INITIAL_SEGMENTATION_CONFIG['threshold_method'] == 'manual'))
                def update_manual_thresh_interactive(
                    method): return gr.Slider(interactive=(method == 'manual'))
                threshold_method_radio.change(
                    fn=update_manual_thresh_interactive, inputs=threshold_method_radio, outputs=manual_thresh_slider, show_progress="hidden")

            with gr.Accordion("Morphological Operations", open=False):
                # ... (Morph settings UI remains the same) ...
                morph_open_k_slider = gr.Slider(label="Open Kernel (Size)", minimum=1, maximum=15,
                                                step=2, value=INITIAL_SEGMENTATION_CONFIG['morph_open_kernel_size'])
                morph_open_iter_slider = gr.Slider(
                    label="Open Iterations", minimum=1, maximum=5, step=1, value=INITIAL_SEGMENTATION_CONFIG['morph_open_iterations'])
                morph_close_k_slider = gr.Slider(label="Close Kernel (Size)", minimum=1, maximum=15,
                                                 step=2, value=INITIAL_SEGMENTATION_CONFIG['morph_close_kernel_size'])
                morph_close_iter_slider = gr.Slider(
                    label="Close Iterations", minimum=1, maximum=5, step=1, value=INITIAL_SEGMENTATION_CONFIG['morph_close_iterations'])

            # --- Magnification-specific filter UI components ---
            # These will all be present in the UI, but only the ones corresponding
            # to selected_magnification will be effectively used by the backend.
            mag_ui_elements = {}
            # Loop through ALL_MAGNIFICATIONS
            for mag_idx, mag_key in enumerate(ALL_MAGNIFICATIONS):
                # Open current default mag
                with gr.Accordion(f"Filters & Seed Ratio for {mag_key}", open=(mag_key == initial_magnification)):
                    conf = INITIAL_SEGMENTATION_CONFIG['contour_filters_by_magnification'][mag_key]
                    mag_ui_elements[f'min_area_{mag_key}'] = gr.Slider(
                        label=f"Min Area", minimum=1, maximum=10000, step=1, value=conf['min_area'])
                    mag_ui_elements[f'max_area_{mag_key}'] = gr.Slider(
                        label=f"Max Area", minimum=100, maximum=50000, step=10, value=conf['max_area'])
                    mag_ui_elements[f'min_circ_{mag_key}'] = gr.Slider(
                        label=f"Min Circularity", minimum=0.0, maximum=1.0, step=0.01, value=conf['min_circularity'])
                    mag_ui_elements[f'dt_ratio_{mag_key}'] = gr.Slider(
                        label=f"Watershed Seed Ratio", minimum=0.01, maximum=0.99, step=0.01, value=conf['dist_transform_thresh_ratio'])

            gr.Markdown("---")
            gr.Markdown("### Configuration Management")
            # ... (Config management UI remains the same) ...
            with gr.Row():
                save_config_btn = gr.Button(
                    "Save Current Settings", variant="primary")
            current_config_textbox = gr.JSON(
                label="Current Config", value=INITIAL_SEGMENTATION_CONFIG, visible=True)
            saved_config_textbox = gr.JSON(
                label="Saved Config", value=INITIAL_SEGMENTATION_CONFIG, visible=True)

        with gr.Column(scale=3, elem_id="display-panel"):
            gr.Markdown(f"## Segmentation Results")  # Title is generic now
            output_ui = {}
            with gr.Row():
                output_ui['orig_img'] = gr.Image(
                    label=f"Original Image", type="numpy", height=400, interactive=False, scale=1)
                output_ui['final_img'] = gr.Image(
                    label=f"Segmented Image", type="numpy", height=400, interactive=False, scale=1)
            intermediate_steps_plot = gr.Plot(label=f"Intermediate Steps", visible=INITIAL_SEGMENTATION_CONFIG.get(
                'show_intermediate_steps', False), elem_id=f"plot-container")

    # --- Update input list for process_patient_slide ---
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

    # Order of inputs for process_patient_slide:
    # selected_patient_id, selected_magnification, show_intermediate_steps_value, global_params..., mag_filter_params...
    event_inputs_for_processing = [
        patient_id_dropdown,
        magnification_selector_radio,  # New
        show_intermediate_steps_check
    ] + param_inputs_from_global_and_morph + param_inputs_from_mag_filters

    all_outputs_for_processing = [
        selected_patient_id_state, image_info_display,
        output_ui['orig_img'], intermediate_steps_plot, output_ui['final_img'],
        current_config_textbox]

    # All components that can trigger a re-processing
    all_event_triggers = [
        patient_id_dropdown,
        magnification_selector_radio,  # New trigger
        show_intermediate_steps_check
    ] + param_inputs_from_global_and_morph + param_inputs_from_mag_filters

    for trigger_component in all_event_triggers:
        trigger_component.change(
            fn=process_patient_slide,
            inputs=event_inputs_for_processing,
            outputs=all_outputs_for_processing,
            show_progress="full")

    def toggle_plot_visibility(show_plot_checkbox_value):
        return gr.Plot(visible=show_plot_checkbox_value)
    show_intermediate_steps_check.change(fn=toggle_plot_visibility, inputs=[
                                         show_intermediate_steps_check], outputs=[intermediate_steps_plot], show_progress="hidden")

    prev_btn.click(fn=navigate_patient_id, inputs=[patient_id_dropdown, gr.State(
        "prev"), all_ids_state], outputs=[patient_id_dropdown])
    next_btn.click(fn=navigate_patient_id, inputs=[patient_id_dropdown, gr.State(
        "next"), all_ids_state], outputs=[patient_id_dropdown])

    def save_config_action(current_config_obj):
        
        if isinstance(current_config_obj, (dict, list)):
            with open(SAVE_FILE_PATH) as f:
                f.write(json.dumps(current_config_obj, indent=2))
        try:
            parsed = json.loads(str(current_config_obj))
            return json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):  # pragma: no cover
            return str(current_config_obj) if not isinstance(current_config_obj, (dict, list)) else "{ \"error\": \"Invalid config format for saving.\" }"
    save_config_btn.click(fn=save_config_action, inputs=[
                          current_config_textbox], outputs=[saved_config_textbox])

    # Initial inputs for app.load
    initial_inputs_for_load = [
        gr.State(initial_patient_id),  # For selected_patient_id
        gr.State(initial_magnification),  # For selected_magnification
        # The rest match the order in event_inputs_for_processing starting from show_intermediate_steps_check
        show_intermediate_steps_check
    ] + param_inputs_from_global_and_morph + param_inputs_from_mag_filters

    app.load(fn=process_patient_slide, inputs=initial_inputs_for_load,
             outputs=all_outputs_for_processing)

if __name__ == '__main__':
    if not ALL_RAW_IMAGE_INFOS or not UNIQUE_PATIENT_SLIDE_IDS:
        print(f"CRITICAL: No images or patient IDs found for any configured magnification.")
    else:
        print(f"Found {len(ALL_RAW_IMAGE_INFOS)} images across all magnifications, {len(UNIQUE_PATIENT_SLIDE_IDS)} unique patient/slide IDs. Launching Gradio UI...")
    app.launch(debug=True, share=False)
