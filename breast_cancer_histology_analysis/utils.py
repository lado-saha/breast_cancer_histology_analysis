import cv2
import numpy as np
from skimage import color
from loguru import logger

def get_hematoxylin_channel(image_rgb_or_bgr, is_bgr=True):
    """Extracts and normalizes the Hematoxylin channel."""
    if image_rgb_or_bgr is None:
        logger.warning("Input image to get_hematoxylin_channel is None.")
        return None
    
    image_rgb = cv2.cvtColor(image_rgb_or_bgr, cv2.COLOR_BGR2RGB) if is_bgr else image_rgb_or_bgr

    image_rgb_safe = np.clip(image_rgb, 1, 255)
    try:
        ihc_hed = color.rgb2hed(image_rgb_safe)
        h_channel = ihc_hed[:, :, 0]
        return cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    except Exception as e:
        logger.error(f"Color deconvolution failed: {e}. Falling back to grayscale.")
        if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        elif len(image_rgb.shape) == 2:
            return image_rgb # Already grayscale
        return None

def calculate_circularity(contour):
    """Calculates the circularity of a contour."""
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0 or area == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)