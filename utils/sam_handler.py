import torch
import numpy as np
import cv2
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time

class SamHandler:
    def __init__(self, model_name="facebook/sam2.1-hiera-large", device="cuda"):
        print("[SAMHandler] Initializing SAM model once on device:", device)
        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name, device=device)
        # Caching removed to ensure new coordinates update the segmentation.
        self.cached_image = None
        self.cached_image_shape = None

    def run_sam_overlay(self, pil_img, coords, active_idx=0, max_dim=640):
        """
        coords: list of normalized coordinates [(nx, ny)] in [0,1]
        active_idx: which coordinate to use for segmentation

        The function downscales the input image to speed up processing,
        runs SAM on the smaller image, then resizes the segmentation mask
        back to the original resolution.
        """
        t0 = time.time()

        if not coords or active_idx < 0 or active_idx >= len(coords):
            return pil_img  # no valid points

        # Convert the original image to a NumPy array.
        np_img = np.array(pil_img)
        orig_h, orig_w, _ = np_img.shape

        # Determine scaling factor if needed.
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pil_img_small = pil_img.resize((new_w, new_h), Image.BILINEAR)
            np_img_small = np.array(pil_img_small)
        else:
            new_w, new_h = orig_w, orig_h
            pil_img_small = pil_img
            np_img_small = np_img

        # Remap the normalized coordinate for the downscaled image.
        (nx, ny) = coords[active_idx]
        px_small = int(nx * new_w)
        py_small = int(ny * new_h)

        # Always update the predictor for a new segmentation request.
        self.predictor.set_image(np_img_small)
        self.cached_image_shape = np_img_small.shape
        self.cached_image = np_img_small.copy()

        const_point_coords = np.array([[px_small, py_small]], dtype=np.float32)
        const_point_labels = np.array([1], dtype=np.int64)

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=const_point_coords,
                point_labels=const_point_labels
            )
        if masks is None or len(masks) == 0:
            return pil_img

        # Get the segmentation mask from the downscaled image.
        mask_small = masks[0]
        # Resize the mask to the original image resolution.
        mask_orig = cv2.resize(mask_small.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_orig = mask_orig.astype(bool)

        # Overlay the segmentation mask (with a green tint) on the original image.
        cv_img = np_img.copy()
        for rr in range(orig_h):
            for cc in range(orig_w):
                if mask_orig[rr, cc]:
                    cv_img[rr, cc] = (0.5 * cv_img[rr, cc] + 0.5 * np.array([0,255,0])).astype(np.uint8)
                else:
                    cv_img[rr, cc] = (0.8 * cv_img[rr, cc]).astype(np.uint8)

        elapsed = time.time() - t0
        print(f"[SAM] Inference time (with resizing): {elapsed:.3f} seconds")
        return Image.fromarray(cv_img)
