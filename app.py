import io
import re
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
from flask_cors import CORS
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import requests

from utils.sam_handler import SamHandler
from utils.depth_handler import DepthHandler, RS_OK
from utils.robopoints_handler import RoboPointsHandler

app = Flask(__name__)
CORS(app)

REMOTE_POINTS_ENDPOINT = "https://5a37-130-225-198-197.ngrok-free.app/predict"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

sam_handler = SamHandler(device=device)
depth_handler = DepthHandler(device=device)
robo_handler = RoboPointsHandler(REMOTE_POINTS_ENDPOINT)

g_state = {
    "mode": "rgb",
    "points_str": "",
    "active_cross_idx": 0,
    "clicked_points": [],
    "last_instruction": "",
    "prev_seg_np_img": None,
    "last_seg_output": None,
    "last_heavy_inference_time": 0.0
}
g_last_raw_frame = None

executor = ThreadPoolExecutor(max_workers=2)
seg_lock = threading.Lock()
depth_lock = threading.Lock()

###########################
# Helper Functions
###########################
def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_remote_for_points(frame_b64, instruction):
    try:
        print("[DEBUG] call_remote_for_points: Sending instruction:", instruction)
        resp = requests.post(
            REMOTE_POINTS_ENDPOINT,
            json={"image": frame_b64, "instruction": instruction},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        print("[DEBUG] Remote endpoint response:", data)
        return data.get("result", "")
    except Exception as e:
        print("[ERROR calling remote for points]:", e)
        return ""

def move_active_cross(direction, dist=1):
    matches = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", g_state["points_str"])
    coords = [(float(a), float(b)) for a, b in matches]
    idx = g_state["active_cross_idx"]
    if not coords or idx < 0 or idx >= len(coords):
        return
    nx, ny = coords[idx]
    shift = dist * 0.01
    if direction == "left":
        nx = max(0.0, nx - shift)
    elif direction == "right":
        nx = min(nx + shift, 1.0)
    elif direction == "up":
        ny = max(0.0, ny - shift)
    elif direction == "down":
        ny = min(ny + shift, 1.0)
    coords[idx] = (nx, ny)
    g_state["points_str"] = "[" + ", ".join(f"({c[0]:.3f}, {c[1]:.3f})" for c in coords) + "]"

def draw_points_on_image(pil_img, points_str, active_idx=-1):
    matches = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", points_str)
    coords = [(float(a), float(b)) for a, b in matches]
    if not coords:
        return pil_img
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape
    for i, (nx, ny) in enumerate(coords):
        px = int(nx * w)
        py = int(ny * h)
        color = (0, 0, 0) if i != active_idx else (255, 0, 0)
        cv2.line(cv_img, (px-10, py-10), (px+10, py+10), color, 2)
        cv2.line(cv_img, (px+10, py-10), (px-10, py+10), color, 2)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

###########################
# Flask Routes
###########################
@app.route("/")
def index():
    vlm_list = ["OpenVLM_1", "OpenVLM_2", "OpenVLM_3"]
    llm_list = ["ChatGPT", "Llama", "Claude"]
    return render_template("dashboard.html", vlm_options=vlm_list, llm_options=llm_list)

@app.route("/camera_info", methods=["GET"])
def camera_info():
    # Get list of connected RealSense devices (if pyrealsense2 is installed)
    rs_devices = depth_handler.list_realsense_devices() if RS_OK else []
    return jsonify({
        "local_cameras": depth_handler.local_cameras_info,
        # Report RealSense as available if we detected any devices
        "realsense_available": True if len(rs_devices) > 0 else False,
        "realsense_devices": rs_devices
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("text", "").strip().lower()
    global g_last_raw_frame
    reply = ""
    if g_state["mode"] == "rgb":
        if text in ["yes", "y"]:
            g_state["mode"] = "depth"
            reply = "Switching to Depth mode."
        else:
            if not g_last_raw_frame:
                reply = "No camera frame available yet. Please wait for the side camera."
            else:
                frame_b64_local = base64.b64encode(g_last_raw_frame).decode("utf-8")
                pts_str = call_remote_for_points(frame_b64_local, text)
                g_state["points_str"] = pts_str
                g_state["active_cross_idx"] = 0
                g_state["mode"] = "depth"
                reply = f"Instruction '{text}' => points={pts_str}, switching to Depth mode."
    elif g_state["mode"] == "depth":
        if text in ["yes", "y"]:
            g_state["mode"] = "sam"
            reply = "Confirmed points => switching to SAM mode."
        elif text in ["no", "n"]:
            reply = "Staying in Depth mode."
        else:
            m = re.search(r"^(\d+)\s*to\s*the\s*(left|right|up|down)$", text)
            if m:
                dist = int(m.group(1))
                direction = m.group(2)
                move_active_cross(direction, dist)
                reply = f"Moved active cross {dist} to the {direction}."
            else:
                reply = "Depth mode: type yes/no or movement commands."
    elif g_state["mode"] == "sam":
        if text in ["yes", "y"]:
            g_state["points_str"] = ""
            g_state["mode"] = "rgb"
            reply = "Segmentation confirmed, returning to RGB mode."
        elif text in ["no", "n"]:
            reply = "Staying in SAM mode."
        else:
            m = re.search(r"^(\d+)\s*to\s*the\s*(left|right|up|down)$", text)
            if m:
                dist = int(m.group(1))
                direction = m.group(2)
                move_active_cross(direction, dist)
                reply = f"Moved cross {dist} to the {direction}."
            else:
                reply = "SAM mode: type yes/no or movement commands."
    else:
        reply = f"Unknown mode: {g_state['mode']}"
    print("[CHAT] user said:", text, "=> reply:", reply)
    return jsonify({"reply": reply})

@app.route("/process_seg", methods=["POST"])
def process_seg_endpoint():
    data = request.json
    frame_b64 = data.get("frame", "")
    if not frame_b64:
        return jsonify({"frame": ""})
    
    # Decode the frame and convert to a PIL image.
    raw_bytes = base64.b64decode(frame_b64)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    clicked_x = data.get("clicked_x")
    clicked_y = data.get("clicked_y")

    if clicked_x is not None and clicked_y is not None:
        # Always run SAM; no toggling
        click_coords = (float(clicked_x), float(clicked_y))
        seg_img = sam_handler.run_sam_overlay(
            pil_img, [click_coords], active_idx=0, max_dim=640
        )
        output_img = seg_img
    else:
        # If no new click, just return the original live frame
        output_img = pil_img

    # Convert output image back to base64 and respond
    out_b64 = pil_to_b64(output_img)
    return jsonify({"frame": out_b64})

@app.route("/process_depth", methods=["POST"])
def process_depth():
    """
    Expects JSON: { "camera_mode": <mode>, "local_idx": <int, optional>, "frame": <base64 string, optional> }
    Supported modes:
      - "off": Stop RealSense & local capture.
      - "default_anything": Use a browser-provided frame (from getUserMedia) with Depth Anything.
      - "sidecam_depth": Same as 'default_anything', but for your UI naming.
      - "local_depth_anything": Use a local camera (via OpenCV) frame with Depth Anything.
      - "realsense_rgb_anything": Use RealSense color frame and run Depth Anything.
      - "realsense_depth": Use RealSense raw depth frame (color mapped).
      - "other": Example mode (invert provided image).
      - "image_depth": Single-call usage for an uploaded image
    """
    data = request.json
    mode = data.get("camera_mode", "off")
    local_idx = data.get("local_idx", -1)
    frame_b64 = data.get("frame", "")

    print(f"[Depth] Received => mode={mode} local_idx={local_idx}")

    if mode == "off":
        # Stop everything
        depth_handler.stop_local_camera()
        depth_handler.stop_realsense()
        return jsonify({"frame": ""})

    elif mode in ("default_anything", "sidecam_depth"):
        # Browser-provided frame => DepthAnything
        if not frame_b64:
            return jsonify({"error": "no local frame"}), 400
        raw = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        out_img = depth_handler.run_depth_anything(pil_img)
        return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "local_depth_anything":
        # Use local camera (OpenCV)
        with depth_lock:
            if not depth_handler.start_local_camera(local_idx):
                return jsonify({"error": f"Failed to open local camera {local_idx}"}), 500
            frame = depth_handler.grab_local_frame()
            if frame is None:
                return jsonify({"error": "no local frame captured"}), 500
            # frame is BGR
            pil_img = Image.fromarray(frame[..., ::-1])  # convert BGR -> RGB
            out_img = depth_handler.run_depth_anything(pil_img)
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "realsense_rgb_anything":
        # RealSense color => DepthAnything
        with depth_lock:
            if not depth_handler.start_realsense():
                return jsonify({"error": "Failed to start RealSense"}), 500
            out_img = depth_handler.realsense_color_to_depthanything()
            if out_img is None:
                return jsonify({"error": "still loading frames"})
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "realsense_depth":
        # RealSense raw depth => color map
        with depth_lock:
            if not depth_handler.start_realsense():
                return jsonify({"error": "Failed to start RealSense"}), 500
            out_img = depth_handler.realsense_depth_colormap()
            if out_img is None:
                return jsonify({"error": "still loading frames"})
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "image_depth":
        # Single call for an uploaded image (like "default_anything" but used once)
        if not frame_b64:
            return jsonify({"error": "no frame provided"}), 400
        raw = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        out_img = depth_handler.run_depth_anything(pil_img)
        return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "other":
        # Example invert
        if frame_b64:
            raw = base64.b64decode(frame_b64)
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.array(pil_img)
            out_img = Image.fromarray(255 - arr)
            return jsonify({"frame": pil_to_b64(out_img)})
        else:
            arr = np.full((240, 320, 3), 127, dtype=np.uint8)
            out_img = Image.fromarray(arr)
            return jsonify({"frame": pil_to_b64(out_img)})

    else:
        # No recognized mode
        return jsonify({"error": f"Unknown camera_mode: {mode}"}), 400

if __name__ == "__main__":
    print("[INFO] Starting on :5000 in multi-threaded mode.")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
