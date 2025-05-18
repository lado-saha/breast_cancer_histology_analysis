import cv2
import numpy as np
import os
import pathlib
import random
import matplotlib.pyplot as plt
from skimage import color, exposure
from scipy import ndimage
import gradio as gr
import json
from collections import defaultdict

# --- Set display options for plots (Gradio handles its own display) ---
plt.style.use('seaborn-v0_8-whitegrid')


def get_all_image_paths(root_dir):
    all_image_infos = []
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
            f"Warning: No images found in {root_dir} or its subdirectories with valid magnification folders.")
    return sorted(all_image_infos, key=lambda x: x['path'])


# --- Configuration (Copied and adapted from your main.ipynb) ---
ROOT_DATA_DIR = './raw/'
INITIAL_SEGMENTATION_CONFIG = {  # Same as before
    'contrast_stretch': False, 'contrast_percentiles_low': 2, 'contrast_percentiles_high': 98,
    'threshold_method': 'otsu', 'manual_threshold_value': 100,
    'morph_open_kernel_size': 3, 'morph_open_iterations': 1,
    'morph_close_kernel_size': 3, 'morph_close_iterations': 1,
    'use_watershed': True, 'dist_transform_thresh_ratio': 0.3,  # Global fallback
    'contour_filters_by_magnification': {
        '40X':  {'min_area': 10,   'max_area': 300,  'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.2},
        '100X': {'min_area': 40,  'max_area': 1000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.3},
        '200X': {'min_area': 80, 'max_area': 3000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.5},
        '400X': {'min_area': 200, 'max_area': 7000, 'min_circularity': 0.3, 'dist_transform_thresh_ratio': 0.6}
    }
}

# --- Helper Functions (display_image_grid_for_gradio, get_hematoxylin_channel, calculate_circularity are the same) ---


def display_image_grid_for_gradio(images, titles, grid_shape, main_title=""):
    num_images = len(images)
    cols = grid_shape[1]
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.ravel() if rows > 1 or cols > 1 else [axes]
    for i, (img, title) in enumerate(zip(images, titles)):
        cmap = 'gray' if len(img.shape) == 2 else None
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
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

# --- (segment_nuclei_advanced_for_gradio is the same as your last working version) ---


def segment_nuclei_advanced_for_gradio(image_bgr, magnification, current_config):
    if image_bgr is None:
        return [], None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    vis_images = [image_rgb.copy()]
    vis_titles = ['1. Original RGB']
    h_channel = get_hematoxylin_channel(image_rgb)
    h_channel_processed = h_channel.copy()
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
                vis_images.append(h_channel_processed.copy())
                vis_titles.append('2b. H-Stretched')

    if current_config['threshold_method'] == 'otsu':
        _, binary_mask = cv2.threshold(
            h_channel_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_mask = cv2.threshold(
            h_channel_processed, current_config['manual_threshold_value'], 255, cv2.THRESH_BINARY)
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
    vis_images.append(cleaned_mask.copy())
    vis_titles.append('4. Cleaned Mask')

    final_contours_list = []
    mag_specific_filters = current_config['contour_filters_by_magnification'].get(
        magnification, current_config['contour_filters_by_magnification']['200X'])

    if current_config.get('use_watershed', True):
        sure_bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(cleaned_mask, sure_bg_kernel, iterations=3)
        vis_images.append(sure_bg.copy())
        vis_titles.append('5a. Sure BG')
        dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        dist_vis = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        vis_images.append(dist_vis.copy())
        vis_titles.append('5b. Distance Tx')
        default_global_dt_ratio = current_config.get(
            'dist_transform_thresh_ratio', 0.3)
        current_dist_thresh_ratio = mag_specific_filters.get(
            'dist_transform_thresh_ratio', default_global_dt_ratio)
        _, sure_fg = cv2.threshold(
            dist_transform, current_dist_thresh_ratio * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        vis_images.append(sure_fg.copy())
        vis_titles.append('5c. Sure FG')
        unknown = cv2.subtract(sure_bg, sure_fg)
        vis_images.append(unknown.copy())
        vis_titles.append('5d. Unknown')
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        watershed_input_img = image_rgb.copy()
        try:
            markers_watershed = cv2.watershed(
                watershed_input_img, markers.copy())
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
        except cv2.error:
            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_contours_list = contours
    else:
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours_list = contours

    min_area, max_area, min_circ = mag_specific_filters['min_area'], mag_specific_filters[
        'max_area'], mag_specific_filters['min_circularity']
    filtered_contours = [cnt for cnt in final_contours_list if min_area < cv2.contourArea(
        cnt) < max_area and calculate_circularity(cnt) >= min_circ]

    image_final_contours_display = image_rgb.copy()
    cv2.drawContours(image_final_contours_display,
                     filtered_contours, -1, (0, 255, 0), 1)
    plot_fig = display_image_grid_for_gradio(
        vis_images, vis_titles, grid_shape=((len(vis_images)+2)//3, 3))
    return filtered_contours, plot_fig, image_final_contours_display


# --- New Data Loading: Group images by Patient/Slide ID ---
PATIENT_SLIDE_IMAGES = defaultdict(
    lambda: {mag: None for mag in ['40X', '100X', '200X', '400X']})
ALL_RAW_IMAGE_INFOS = get_all_image_paths(ROOT_DATA_DIR)

if not ALL_RAW_IMAGE_INFOS:
    print("CRITICAL ERROR: No images found. Please check ROOT_DATA_DIR in the script.")
else:
    for img_info in ALL_RAW_IMAGE_INFOS:
        # Assuming filename format like: SOB_B_A_14-22549AB-100-001.png
        # Or SOB_M_DC_14-10926-100-001.png
        parts = img_info['filename'].split('-')
        # SOB_M_DC_14-10926  (parts[0] to parts[3] usually)
        if len(parts) >= 4:
            # Patient ID might be a combination of parts before magnification
            # Example: "SOB_B_A_14-22549AB" or "SOB_M_DC_14-10926"
            # This needs to be robust to your filename structure.
            # A common pattern is "TYPE_CLASS_SUBTYPE_PATIENTID"
            # Let's try to capture up to the part before magnification string like "100"
            try:
                # Find the part that contains the magnification (e.g., "100X", "40X")
                # and take everything before it as part of the patient_slide_id_prefix
                filename_no_ext = os.path.splitext(img_info['filename'])[0]
                # Find last occurrence of magnification string like "-40-" or "-100-"
                last_mag_part_idx = -1
                for mag_str_part in ["-40-", "-100-", "-200-", "-400-"]:
                    try:
                        idx = filename_no_ext.rindex(mag_str_part)
                        if idx > last_mag_part_idx:
                            last_mag_part_idx = idx
                    except ValueError:
                        continue

                if last_mag_part_idx != -1:
                    patient_slide_id = filename_no_ext[:last_mag_part_idx]
                else:  # Fallback if pattern not perfectly matched
                    # Heuristic: join all but last two parts (magnification, sequence)
                    patient_slide_id = "-".join(parts[:-2])

                PATIENT_SLIDE_IMAGES[patient_slide_id][img_info['magnification']
                                                       ] = img_info['path']
            except Exception as e:
                print(
                    f"Could not parse patient_slide_id from {img_info['filename']}: {e}")
                continue

UNIQUE_PATIENT_SLIDE_IDS = sorted(list(PATIENT_SLIDE_IMAGES.keys()))

# --- Gradio UI Definition ---


def process_patient_slide(
    selected_patient_id,  # This is now the primary input
    contrast_stretch, contrast_p_low, contrast_p_high,
    threshold_method, manual_threshold_value,
    morph_open_kernel, morph_open_iter,
    morph_close_kernel, morph_close_iter,
    use_watershed,
    min_area_40X, max_area_40X, min_circ_40X, dt_ratio_40X,
    min_area_100X, max_area_100X, min_circ_100X, dt_ratio_100X,
    min_area_200X, max_area_200X, min_circ_200X, dt_ratio_200X,
    min_area_400X, max_area_400X, min_circ_400X, dt_ratio_400X
):
    if not selected_patient_id or not PATIENT_SLIDE_IMAGES:
        # Create empty outputs for all 4 magnifications
        empty_plot = display_image_grid_for_gradio([], [], (1, 1), "No Data")
        return selected_patient_id, "No Patient ID selected or no images.", (None, empty_plot, None) * 4, "{}"

    live_config = {  # Same construction as before
        'contrast_stretch': contrast_stretch, 'contrast_percentiles_low': int(contrast_p_low),
        'contrast_percentiles_high': int(contrast_p_high), 'threshold_method': threshold_method,
        'manual_threshold_value': int(manual_threshold_value), 'morph_open_kernel_size': int(morph_open_kernel),
        'morph_open_iterations': int(morph_open_iter), 'morph_close_kernel_size': int(morph_close_kernel),
        'morph_close_iterations': int(morph_close_iter), 'use_watershed': use_watershed,
        'dist_transform_thresh_ratio': INITIAL_SEGMENTATION_CONFIG['dist_transform_thresh_ratio'],
        'contour_filters_by_magnification': {
            '40X':  {'min_area': int(min_area_40X),   'max_area': int(max_area_40X),  'min_circularity': float(min_circ_40X), 'dist_transform_thresh_ratio': float(dt_ratio_40X)},
            '100X': {'min_area': int(min_area_100X),  'max_area': int(max_area_100X), 'min_circularity': float(min_circ_100X), 'dist_transform_thresh_ratio': float(dt_ratio_100X)},
            '200X': {'min_area': int(min_area_200X), 'max_area': int(max_area_200X), 'min_circularity': float(min_circ_200X), 'dist_transform_thresh_ratio': float(dt_ratio_200X)},
            '400X': {'min_area': int(min_area_400X), 'max_area': int(max_area_400X), 'min_circularity': float(min_circ_400X), 'dist_transform_thresh_ratio': float(dt_ratio_400X)}
        }
    }

    outputs_for_tabs = []
    magnifications_processed = []

    for mag in ['40X', '100X', '200X', '400X']:
        image_path = PATIENT_SLIDE_IMAGES[selected_patient_id].get(mag)
        if image_path:
            image_bgr = cv2.imread(image_path)
            if image_bgr is not None:
                _, intermediate_fig, final_img = segment_nuclei_advanced_for_gradio(
                    image_bgr, mag, live_config)
                original_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                outputs_for_tabs.extend(
                    [original_img_rgb, intermediate_fig, final_img])
                magnifications_processed.append(mag)
            else:  # Image path exists but couldn't be read
                empty_plot = display_image_grid_for_gradio(
                    [], [], (1, 1), f"{mag}: Error loading")
                outputs_for_tabs.extend([None, empty_plot, None])
        else:  # No image for this magnification
            empty_plot = display_image_grid_for_gradio(
                [], [], (1, 1), f"{mag}: Not found")
            outputs_for_tabs.extend([None, empty_plot, None])

    info_text = f"Displaying: {selected_patient_id} (Magnifications processed: {', '.join(magnifications_processed) if magnifications_processed else 'None'})"
    return [selected_patient_id, info_text] + outputs_for_tabs + [live_config]


# --- UI Layout Improvements ---
# --- UI Layout Improvements ---

with gr.Blocks(theme=gr.themes.Soft(), title="Nuclei Segmentation Tuner") as app:
    gr.Markdown(
        "# Interactive Nuclei Segmentation Tuner (All Magnifications View)")

    # Updated CSS to fix the layout issues
    # Store the currently selected patient ID as state
    selected_patient_id_state = gr.State(
        UNIQUE_PATIENT_SLIDE_IDS[0] if UNIQUE_PATIENT_SLIDE_IDS else None)

    with gr.Row():
        # Left column (control panel) with fixed width
        with gr.Column(scale=1, elem_id="control-panel", min_width=300):
            gr.Markdown("## Controls")
            patient_id_dropdown = gr.Dropdown(
                label="Select Patient/Slide ID",
                choices=UNIQUE_PATIENT_SLIDE_IDS,
                value=UNIQUE_PATIENT_SLIDE_IDS[0] if UNIQUE_PATIENT_SLIDE_IDS else None
            )
            image_info_display = gr.Textbox(
                label="Processing Info",
                interactive=False,
                lines=2
            )

            gr.Markdown("---")
            gr.Markdown("### Segmentation Parameters")

            # Global settings accordion
            with gr.Accordion("Global Settings", open=True):
                with gr.Row():
                    contrast_stretch_check = gr.Checkbox(
                        label="Contrast Stretch H-Channel",
                        value=INITIAL_SEGMENTATION_CONFIG['contrast_stretch']
                    )
                    use_watershed_check = gr.Checkbox(
                        label="Use Watershed",
                        value=INITIAL_SEGMENTATION_CONFIG['use_watershed']
                    )
                with gr.Row():
                    contrast_p_low_slider = gr.Slider(
                        label="Contrast Low %",
                        minimum=0,
                        maximum=49,
                        step=1,
                        value=INITIAL_SEGMENTATION_CONFIG['contrast_percentiles_low']
                    )
                    contrast_p_high_slider = gr.Slider(
                        label="Contrast High %",
                        minimum=51,
                        maximum=100,
                        step=1,
                        value=INITIAL_SEGMENTATION_CONFIG['contrast_percentiles_high']
                    )

                threshold_method_radio = gr.Radio(
                    label="Threshold Method",
                    choices=['otsu', 'manual'],
                    value=INITIAL_SEGMENTATION_CONFIG['threshold_method']
                )

                manual_thresh_slider = gr.Slider(
                    label="Manual Threshold (if manual)",
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=INITIAL_SEGMENTATION_CONFIG['manual_threshold_value'],
                    interactive=(
                        INITIAL_SEGMENTATION_CONFIG['threshold_method'] == 'manual')
                )

                def update_manual_thresh_interactive(method):
                    return gr.Slider(interactive=(method == 'manual'))

                threshold_method_radio.change(
                    fn=update_manual_thresh_interactive,
                    inputs=threshold_method_radio,
                    outputs=manual_thresh_slider,
                    show_progress="hidden"
                )

            # Morphological Operations accordion
            with gr.Accordion("Morphological Operations", open=False):
                morph_open_k_slider = gr.Slider(
                    label="Open Kernel (Size)",
                    minimum=1,
                    maximum=15,
                    step=2,
                    value=INITIAL_SEGMENTATION_CONFIG['morph_open_kernel_size']
                )

                morph_open_iter_slider = gr.Slider(
                    label="Open Iterations",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=INITIAL_SEGMENTATION_CONFIG['morph_open_iterations']
                )

                morph_close_k_slider = gr.Slider(
                    label="Close Kernel (Size)",
                    minimum=1,
                    maximum=15,
                    step=2,
                    value=INITIAL_SEGMENTATION_CONFIG['morph_close_kernel_size']
                )

                morph_close_iter_slider = gr.Slider(
                    label="Close Iterations",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=INITIAL_SEGMENTATION_CONFIG['morph_close_iterations']
                )

            # Magnification-specific settings
            mag_ui_elements = {}
            for mag_idx, mag_key in enumerate(['40X', '100X', '200X', '400X']):
                with gr.Accordion(f"Filters & Seed Ratio for {mag_key}", open=(mag_idx == 0)):
                    conf = INITIAL_SEGMENTATION_CONFIG['contour_filters_by_magnification'][mag_key]
                    mag_ui_elements[f'min_area_{mag_key}'] = gr.Slider(
                        label=f"Min Area",
                        minimum=1,
                        maximum=10000,
                        step=1,
                        value=conf['min_area']
                    )

                    mag_ui_elements[f'max_area_{mag_key}'] = gr.Slider(
                        label=f"Max Area",
                        minimum=100,
                        maximum=50000,
                        step=10,
                        value=conf['max_area']
                    )

                    mag_ui_elements[f'min_circ_{mag_key}'] = gr.Slider(
                        label=f"Min Circularity",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=conf['min_circularity']
                    )

                    mag_ui_elements[f'dt_ratio_{mag_key}'] = gr.Slider(
                        label=f"Watershed Seed Ratio",
                        minimum=0.01,
                        maximum=0.99,
                        step=0.01,
                        value=conf['dist_transform_thresh_ratio']
                    )

            gr.Markdown("---")
            gr.Markdown("### Configuration Management")

            # Config management section
            with gr.Row():
                save_config_btn = gr.Button(
                    "Save Current Settings",
                    variant="primary"
                )

            current_config_textbox = gr.JSON(
                label="Current Config",
                value=INITIAL_SEGMENTATION_CONFIG,
                visible=True
            )

            saved_config_textbox = gr.JSON(
                label="Saved Config",
                value=INITIAL_SEGMENTATION_CONFIG,
                visible=True
            )

        # Right column (display panel)
        with gr.Column(scale=3, elem_id="display-panel"):
            gr.Markdown("## Segmentation Results by Magnification")

            # Create UI elements for each magnification within Tabs
            output_ui_elements_per_mag = {}
            with gr.Tabs() as magnification_tabs:
                for mag_key in ['40X', '100X', '200X', '400X']:
                    with gr.TabItem(label=mag_key, id=mag_key):
                        with gr.Row():
                            # Original and segmented images side by side
                            output_ui_elements_per_mag[f'orig_{mag_key}'] = gr.Image(
                                label=f"Original {mag_key}",
                                type="numpy",
                                height=300,
                                interactive=False,
                                scale=1
                            )

                            output_ui_elements_per_mag[f'final_{mag_key}'] = gr.Image(
                                label=f"Segmented {mag_key}",
                                type="numpy",
                                height=300,
                                interactive=False,
                                scale=1
                            )

                        # Intermediate steps plot (below the images)
                        output_ui_elements_per_mag[f'plot_{mag_key}'] = gr.Plot(
                            label=f"Intermediate Steps {mag_key}",
                            elem_id=f"plot-container-{mag_key}"
                        )

    # Collect all parameter UI components (inputs to process_patient_slide)
    all_param_inputs_for_event_handling = [
        contrast_stretch_check, contrast_p_low_slider, contrast_p_high_slider,
        threshold_method_radio, manual_thresh_slider,
        morph_open_k_slider, morph_open_iter_slider,
        morph_close_k_slider, morph_close_iter_slider,
        use_watershed_check,
        mag_ui_elements['min_area_40X'], mag_ui_elements['max_area_40X'], mag_ui_elements['min_circ_40X'], mag_ui_elements['dt_ratio_40X'],
        mag_ui_elements['min_area_100X'], mag_ui_elements['max_area_100X'], mag_ui_elements['min_circ_100X'], mag_ui_elements['dt_ratio_100X'],
        mag_ui_elements['min_area_200X'], mag_ui_elements['max_area_200X'], mag_ui_elements['min_circ_200X'], mag_ui_elements['dt_ratio_200X'],
        mag_ui_elements['min_area_400X'], mag_ui_elements['max_area_400X'], mag_ui_elements['min_circ_400X'], mag_ui_elements['dt_ratio_400X']
    ]

    # Collect all output UI components for the tabs
    all_outputs_for_processing = [
        selected_patient_id_state,  # This will be updated if dropdown changes selection
        image_info_display,
        output_ui_elements_per_mag['orig_40X'], output_ui_elements_per_mag['plot_40X'], output_ui_elements_per_mag['final_40X'],
        output_ui_elements_per_mag['orig_100X'], output_ui_elements_per_mag[
            'plot_100X'], output_ui_elements_per_mag['final_100X'],
        output_ui_elements_per_mag['orig_200X'], output_ui_elements_per_mag[
            'plot_200X'], output_ui_elements_per_mag['final_200X'],
        output_ui_elements_per_mag['orig_400X'], output_ui_elements_per_mag[
            'plot_400X'], output_ui_elements_per_mag['final_400X'],
        current_config_textbox
    ]

    # Event handling
    # When a parameter or the patient_id_dropdown changes, call process_patient_slide
    event_triggers = [patient_id_dropdown] + \
        all_param_inputs_for_event_handling

    # Debounced event handling to prevent excessive recomputation
    for trigger_component in event_triggers:
        trigger_component.change(
            fn=process_patient_slide,
            inputs=[patient_id_dropdown] + all_param_inputs_for_event_handling,
            outputs=all_outputs_for_processing,
            show_progress="full"
        )

    # Config save button handler
    def save_config_action(live_config_value):
        return live_config_value

    save_config_btn.click(
        fn=save_config_action,
        inputs=[current_config_textbox],
        outputs=[saved_config_textbox]
    )

    # Load initial data
    initial_inputs_for_load = [
        gr.State(UNIQUE_PATIENT_SLIDE_IDS[0]
                 if UNIQUE_PATIENT_SLIDE_IDS else None)
    ] + all_param_inputs_for_event_handling

    app.load(
        fn=process_patient_slide,
        inputs=initial_inputs_for_load,
        outputs=all_outputs_for_processing
    )
if __name__ == '__main__':
    if not ALL_RAW_IMAGE_INFOS or not UNIQUE_PATIENT_SLIDE_IDS:
        print("CRITICAL: No images or patient IDs found. UI cannot start meaningfully.")
        print(
            f"Please check the path: {os.path.abspath(ROOT_DATA_DIR)} and file naming conventions.")
    else:
        print(f"Found {len(ALL_RAW_IMAGE_INFOS)} images, {len(UNIQUE_PATIENT_SLIDE_IDS)} unique patient/slide IDs. Launching Gradio UI...")
        app.launch(debug=True, share=False)

