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
    def __init__(self, device="cuda"):
        """
        1) Set up the 'Depth Anything' pipeline for any RGB input.
        2) Prepare RealSense (D435i) management.
        3) Prepare local camera management using OpenCV.
        """
        self.device = device

        # 1) Depth Anything pipeline
        self.depth_anything = pipeline(
            'depth-estimation',
            model='depth-anything/Depth-Anything-V2-base-hf',
            device=device
        )

        # 2) RealSense placeholders and state.
        self.rs_pipeline = None
        self.rs_config = None
        self.rs_align = None
        self.rs_active = False
        self.frame_timeout = 2000  # ms

        # 3) Local camera placeholders.
        self.local_cap = None
        self.local_cam_index = None
        self.local_cameras_info = self.enumerate_local_cameras()

    @property
    def is_realsense_connected(self):
        """Return True if the RealSense pipeline is currently running."""
        return self.rs_active

    def enumerate_local_cameras(self, max_cams=8):
        """
        Attempt to open camera indices [0, max_cams) using CAP_MSMF (Windows)
        and fallback to CAP_DSHOW. Returns list of indices that opened.
        """
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
        """Return list of connected RealSense devices (if any)."""
        if not RS_OK:
            return []
        try:
            ctx = rs.context()
            devices = []
            for dev in ctx.devices:
                devices.append({
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number)
                })
            print("[DepthHandler] Found RealSense devices:", devices)
            return devices
        except Exception as e:
            print("[DepthHandler] Error listing RealSense devices:", e)
            return []

    ###########################
    # RealSense Management
    ###########################
    def start_realsense(self):
        """Start the RealSense pipeline if not already started."""
        if not RS_OK:
            print("[DepthHandler] pyrealsense2 not installed; cannot start RealSense.")
            return False
        if self.rs_active:
            return True
        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.rs_pipeline.start(self.rs_config)
            self.rs_align = rs.align(rs.stream.color)
            self.rs_active = True
            print("[DepthHandler] RealSense pipeline started.")
            return True
        except Exception as e:
            print("[DepthHandler] Failed to start RealSense:", e)
            self.rs_active = False
            return False

    def stop_realsense(self):
        """Stop the RealSense pipeline if active."""
        if self.rs_active and self.rs_pipeline:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_config = None
            self.rs_align = None
            self.rs_active = False
            print("[DepthHandler] RealSense pipeline stopped.")

    def get_realsense_frames(self):
        """
        Grab color and depth frames from RealSense.
        Returns (color_array, depth_array) as numpy arrays, or (None, None) on failure.
        """
        if not self.rs_active or not self.rs_pipeline:
            print("[DepthHandler] RealSense not active or pipeline missing.")
            return None, None
        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=self.frame_timeout)
            if not frames:
                print("[DepthHandler] RealSense: no frames within timeout.")
                return None, None
            aligned = self.rs_align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                print("[DepthHandler] RealSense: missing color or depth frame.")
                return None, None
            color_arr = np.asanyarray(color_frame.get_data())
            depth_arr = np.asanyarray(depth_frame.get_data())
            return color_arr, depth_arr
        except Exception as e:
            print("[DepthHandler] RealSense error:", e)
            return None, None

    def realsense_color_to_depthanything(self):
        """
        Capture RealSense RGB, convert to PIL, and run Depth Anything.
        Returns resulting PIL image or None.
        """
        color_arr, _ = self.get_realsense_frames()
        if color_arr is None:
            return None
        pil_img = Image.fromarray(color_arr[..., ::-1])  # Convert BGR to RGB
        return self.run_depth_anything(pil_img)

    def realsense_depth_colormap(self):
        """
        Capture RealSense depth, apply a color map, and return a PIL image.
        """
        _, depth_arr = self.get_realsense_frames()
        if depth_arr is None:
            return None
        depth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_arr, alpha=0.03),
            cv2.COLORMAP_JET
        )
        rgb = depth_color[..., ::-1]
        return Image.fromarray(rgb)

    ###########################
    # Local Camera Management
    ###########################
    def start_local_camera(self, index):
        """
        Open a local camera with OpenCV (CAP_MSMF then CAP_DSHOW as fallback).
        """
        if self.local_cap and self.local_cam_index == index:
            return True
        self.stop_local_camera()
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[DepthHandler] Could not open local camera {index}")
            return False
        self.local_cap = cap
        self.local_cam_index = index
        print(f"[DepthHandler] Opened local camera index={index} successfully.")
        return True

    def stop_local_camera(self):
        """Release any open local camera."""
        if self.local_cap:
            self.local_cap.release()
            self.local_cap = None
            self.local_cam_index = None
            print("[DepthHandler] Stopped local camera.")

    def grab_local_frame(self):
        """
        Capture a frame from the local camera.
        Returns a BGR numpy array or None.
        """
        if not self.local_cap:
            return None
        ret, frame = self.local_cap.read()
        if not ret:
            print("[DepthHandler] Failed to read local camera frame.")
            return None
        return frame

    ###########################
    # Depth Anything Processing
    ###########################
    def run_depth_anything(self, pil_img):
        """
        Run the "Depth Anything" pipeline on an RGB PIL image.
        Returns a colorized depth map as a PIL image.
        """
        depth_out = self.depth_anything(pil_img)
        depth_arr = np.array(depth_out["depth"], dtype=np.float32)
        depth_norm = cv2.normalize(depth_arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        rgb = depth_color[..., ::-1]
        return Image.fromarray(rgb)
