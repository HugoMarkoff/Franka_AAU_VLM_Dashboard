# depth_handler.py
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import pipeline

try:
    import pyrealsense2 as rs
    RS_OK = True
except ImportError:
    RS_OK = False


class DepthHandler:
    """
    Handles three things:

      1) "Depth-Anything" monocular depth inference for any RGB image
      2) RealSense D400-series RGB-D capture and helpers
      3) Local webcam capture via OpenCV

    New in this version
    -------------------
    * calculate_object_info(mask_bool, depth_arr)   →  dict
        Compute centre X-Y-Z, Euclidean distance, physical width/height and
        bounding-box pixels for a segmented object, given the Boolean mask
        from SAM and the raw 16-bit depth frame from RealSense.
    """

    def __init__(self, device="cuda"):
        # 1 — Depth-Anything
        self.depth_anything = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-base-hf",
            device=device,
        )

        # 2 — RealSense state
        self.rs_pipeline = None
        self.rs_config   = None
        self.rs_align    = None
        self.rs_active   = False
        self.frame_timeout = 2_000  # ms

        # 3 — Local webcam state
        self.local_cap        = None
        self.local_cam_index  = None
        self.local_cameras_info = self.enumerate_local_cameras()

    # ------------------------------------------------------------------
    # Camera enumeration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def enumerate_local_cameras(max_cams: int = 8):
        cams = []
        for idx in range(max_cams):
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                cams.append(idx)
                cap.release()
        return cams

    def list_realsense_devices(self):
        if not RS_OK:
            return []
        try:
            ctx = rs.context()
            return [
                {
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                }
                for dev in ctx.devices
            ]
        except Exception as e:
            print("[DepthHandler] Error listing RealSense devices:", e)
            return []

    # ------------------------------------------------------------------
    # RealSense management
    # ------------------------------------------------------------------
    def start_realsense(self):
        if not RS_OK:
            print("[DepthHandler] pyrealsense2 not installed.")
            return False
        if self.rs_active:
            return True
        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config   = rs.config()
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.rs_pipeline.start(self.rs_config)
            self.rs_align   = rs.align(rs.stream.color)
            self.rs_active  = True
            return True
        except Exception as e:
            print("[DepthHandler] Failed to start RealSense:", e)
            return False

    def stop_realsense(self):
        if self.rs_active and self.rs_pipeline:
            self.rs_pipeline.stop()
        self.rs_pipeline = self.rs_config = self.rs_align = None
        self.rs_active = False

    def get_realsense_frames(self):
        if not (self.rs_active and self.rs_pipeline):
            return (None, None)
        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=self.frame_timeout)
            aligned = self.rs_align.process(frames)
            color   = aligned.get_color_frame()
            depth   = aligned.get_depth_frame()
            if not color or not depth:
                return (None, None)
            return (
                np.asanyarray(color.get_data()),
                np.asanyarray(depth.get_data()),  # uint16, millimetres
            )
        except Exception as e:
            print("[DepthHandler] RealSense error:", e)
            return (None, None)

    def realsense_color_to_depthanything(self):
        color_arr, _ = self.get_realsense_frames()
        if color_arr is None:
            return None
        return self.run_depth_anything(Image.fromarray(color_arr))

    def realsense_depth_colormap(self):
        _, depth_arr = self.get_realsense_frames()
        if depth_arr is None:
            return None
        depth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_arr, alpha=0.03), cv2.COLORMAP_JET
        )
        return Image.fromarray(depth_color[..., ::-1])

    # ------------------------------------------------------------------
    # Local webcam management
    # ------------------------------------------------------------------
    def start_local_camera(self, index: int):
        if self.local_cap and self.local_cam_index == index:
            return True
        self.stop_local_camera()
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[DepthHandler] Could not open camera {index}")
            return False
        self.local_cap = cap
        self.local_cam_index = index
        return True

    def stop_local_camera(self):
        if self.local_cap:
            self.local_cap.release()
        self.local_cap = self.local_cam_index = None

    def grab_local_frame(self):
        if not self.local_cap:
            return None
        ok, frame = self.local_cap.read()
        return frame if ok else None

    def run_depth_anything(self, pil_img: Image.Image) -> Image.Image:
        depth_out   = self.depth_anything(pil_img)
        depth_arr   = np.array(depth_out["depth"], dtype=np.float32)
        depth_norm  = cv2.normalize(depth_arr, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return Image.fromarray(depth_color[..., ::-1])

    def calculate_object_info(self, mask_bool: np.ndarray, depth_arr: np.ndarray):
        import numpy as np, cv2, pyrealsense2 as rs

        # 1) Align mask to depth resolution
        h_m, w_m = mask_bool.shape
        h_d, w_d = depth_arr.shape
        if (h_m, w_m) != (h_d, w_d):
            mask_depth = cv2.resize(
                mask_bool.astype(np.uint8),
                (w_d, h_d),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_depth = mask_bool

        # 2) Compute centroid in mask space
        ys, xs = np.where(mask_depth)
        if ys.size == 0:
            print("[DepthHandler] No mask pixels found.")
            return None
        cy_init, cx_init = int(ys.mean() + 0.5), int(xs.mean() + 0.5)

        # 3) Dynamic patch expansion until >=3 valid depths
        max_patch = 25  # up to 25x25
        patch_size = 5
        patch = None
        nonzero = 0
        while patch_size <= max_patch:
            radius = patch_size // 2
            y0 = max(cy_init - radius, 0)
            y1 = min(cy_init + radius, h_d - 1)
            x0 = max(cx_init - radius, 0)
            x1 = min(cx_init + radius, w_d - 1)
            patch = depth_arr[y0:y1+1, x0:x1+1].astype(np.float32)
            flat = patch.flatten()
            nonzero = np.count_nonzero(flat)
            print(f"[DepthHandler] {patch_size}x{patch_size} patch depths (mm): {flat.tolist()}")
            if nonzero >= 3:
                break
            patch_size += 2
        if patch is None or nonzero == 0:
            print(f"[DepthHandler] No non-zero depths found up to {max_patch}x{max_patch} patch.")
            return None

        # 4) Filter non-zero values and cluster by tolerance
        vals = flat[flat > 0]
        vals_m = vals / 1000.0
        med = np.median(vals_m)
        rel_tol = 0.10 * med
        abs_tol = 0.02
        tol = max(rel_tol, abs_tol)
        good = vals_m[np.abs(vals_m - med) <= tol]
        if good.size < (vals_m.size / 2):
            good = vals_m
        z_m = float(np.mean(good))

        # 5) Build front-face mask and bounding-box based on tolerance around median
        depth_m = depth_arr.astype(np.float32) / 1000.0
        front = mask_depth & (np.abs(depth_m - med) <= tol)
        if front.sum() < (mask_depth.sum() * 0.1):
            front = mask_depth
        ys_f, xs_f = np.where(front)
        x_min, x_max = xs_f.min(), xs_f.max()
        y_min, y_max = ys_f.min(), ys_f.max()
        px_w = x_max - x_min + 1
        px_h = y_max - y_min + 1

        # 6) Compute physical size via pinhole model
        if not self.rs_active:
            print("[DepthHandler] RealSense not active")
            return None
        intr = (
            self.rs_pipeline
                .get_active_profile()
                .get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
        )
        width_m = (px_w * z_m) / intr.fx
        height_m = (px_h * z_m) / intr.fy

        # 7) Apply fixed scale factor for calibration (e.g., 0.62)
        fixed_scale = 0.62
        width_m *= fixed_scale
        height_m *= fixed_scale
        print(f"[DepthHandler] Applied fixed scale factor: {fixed_scale}")

        # 8) Deproject centroid to 3D XYZ and compute distance
        center_xyz = rs.rs2_deproject_pixel_to_point(
            intr, [float(cx_init), float(cy_init)], z_m
        )
        distance_m = float(np.linalg.norm(center_xyz))

        return {
            "center_xyz_m": center_xyz,
            "distance_m": distance_m,
            "width_m": float(width_m),
            "height_m": float(height_m),
            "bbox_px": [int(x_min), int(y_min), int(x_max), int(y_max)]
        }
