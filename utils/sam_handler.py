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
        # Remove caching (if any) so new coordinates update the segmentation.
        self.cached_image = None
        self.cached_image_shape = None
        # Internal state for toggling segmentation
        self.segmentation_active = False
        self.last_seg_output = None
        self.last_click = None

    def process_seg(self, pil_img, click_coords=None, max_dim=640):
        """
        Always run SAM at the given point (no toggling).
        """
        if not click_coords:
            return pil_img
        
        print("[SAMHandler] Running segmentation at:", click_coords)
        seg_img = self.run_sam_overlay(pil_img, [click_coords], active_idx=0, max_dim=max_dim)
        return seg_img

    def run_sam_overlay(self, pil_img, coords, active_idx=0, max_dim=640):
        """
        Run SAM on a downscaled image using the active normalized coordinate.
        """
        t0 = time.time()

        if not coords or active_idx < 0 or active_idx >= len(coords):
            return pil_img  # no valid points provided

        # Convert PIL image to NumPy array.
        np_img = np.array(pil_img)
        orig_h, orig_w, _ = np_img.shape

        # Downscale if needed.
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

        # Map the active normalized coordinate to the downscaled image.
        (nx, ny) = coords[active_idx]
        px_small = int(nx * new_w)
        py_small = int(ny * new_h)

        # Set the predictor with the small image.
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

        # Resize the segmentation mask back to original dimensions.
        mask_small = masks[0]
        mask_orig = cv2.resize(mask_small.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_orig = mask_orig.astype(bool)

        # Overlay the segmentation mask (with green tint) on the original image.
        cv_img = np_img.copy()
        for rr in range(orig_h):
            for cc in range(orig_w):
                if mask_orig[rr, cc]:
                    # Blend the pixel with green color.
                    cv_img[rr, cc] = (0.5 * cv_img[rr, cc] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
                else:
                    # Dim non-segmented areas slightly.
                    cv_img[rr, cc] = (0.8 * cv_img[rr, cc]).astype(np.uint8)

        elapsed = time.time() - t0
        print(f"[SAM] Inference time (with resizing): {elapsed:.3f} seconds")
        return Image.fromarray(cv_img)

    def toggle_segmentation(self, pil_img, click_coords=None, max_dim=640):
        """
        Toggle segmentation:
          - If segmentation is off, run SAM with the provided coordinate and cache the output.
          - If segmentation is already active, reset and return the original live frame.
        """
        if not self.segmentation_active:
            if click_coords is None:
                raise ValueError("No coordinate provided for starting segmentation.")
            print("[SAMHandler] Starting segmentation at:", click_coords)
            seg_img = self.run_sam_overlay(pil_img, [click_coords], active_idx=0, max_dim=max_dim)
            self.segmentation_active = True
            self.last_seg_output = seg_img
            self.last_click = click_coords
            return seg_img
        else:
            print("[SAMHandler] Toggling segmentation off, resuming live feed.")
            self.segmentation_active = False
            self.last_seg_output = None
            self.last_click = None
            return pil_img
