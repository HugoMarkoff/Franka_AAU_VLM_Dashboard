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

REMOTE_POINTS_ENDPOINT = "https://23f4-130-225-198-197.ngrok-free.app/predict"
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
g_frozen_seg = None  # This will store the frozen candidate overlay image (PIL Image or raw bytes)

executor = ThreadPoolExecutor(max_workers=2)
seg_lock = threading.Lock()
depth_lock = threading.Lock()

###########################
# Helper Functions
###########################
def get__current_frame(seg_input, seg_frame, instruction):
    """
    Retrieves the current frame from the segmentation window. If the seg_input dropdown is "off",
    returns an error message; otherwise, if the client has captured a seg_frame (as base64),
    it decodes and updates the global frame (g_last_raw_frame). If no new frame is provided,
    the function falls back on the previously stored frame.
    
    It then encodes the frame and calls the remote endpoint with the instruction.
    
    Parameters:
        seg_input (str): The current dropdown selection ("on" or "off").
        seg_frame (str): The base64-encoded frame from the seg window.
        instruction (str): The instruction (e.g. "cup") for the robot.
        
    Returns:
        str: The result from the remote endpoint (point candidate string) or an error message.
    """
    global g_last_raw_frame
    print(f"[DEBUG] Segmentation input dropdown selected: {seg_input}.")
    if seg_input.lower() == "off":
        msg = "Seg window input is off. Please select an active input to capture its frame."
        print(f"[DEBUG] {msg}")
        return msg

    if seg_frame:
        try:
            frame_bytes = base64.b64decode(seg_frame)
            g_last_raw_frame = frame_bytes
            print("[DEBUG] Received new seg frame from client and updated g_last_raw_frame.")
        except Exception as e:
            error_msg = f"Failed to decode seg_frame: {e}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    else:
        if not g_last_raw_frame:
            error_msg = "No segmentation frame available; g_last_raw_frame is empty."
            print(f"[DEBUG] {error_msg}")
            return error_msg
        else:
            print("[DEBUG] No new seg_frame provided; using stored g_last_raw_frame.")

    try:
        frame_b64 = base64.b64encode(g_last_raw_frame).decode("utf-8")
    except Exception as e:
        error_msg = f"Error encoding seg frame: {e}"
        print(f"[ERROR] {error_msg}")
        return error_msg

    print(f"[DEBUG] Prepared seg frame in base64. Sending to remote endpoint with instruction: {instruction}")
    result = call_remote_for_points(frame_b64, instruction)
    print(f"[DEBUG] Remote endpoint returned: {result}")
    return result


def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_remote_for_points(frame_b64, instruction):
    try:
        print(f"[DEBUG] User input '{instruction}' sent to robot point.")
        resp = requests.post(
            REMOTE_POINTS_ENDPOINT,
            json={"image": frame_b64, "instruction": instruction},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", "")
        print(f"[DEBUG] Coordinates returned from robo point: {result}")
        return result
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

def draw_points_on_image(pil_img, points_str, active_idx=0, only_active=False):
    """
    Draws crosses on pil_img at the normalized points in points_str.
    If only_active=True, only the active_idx point is drawn (in blue).
    """
    matches = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", points_str)
    coords = [(float(a), float(b)) for a, b in matches]
    if not coords:
        print("[DEBUG] No coordinates found in points_str. Returning original image.")
        return pil_img

    print(f"[DEBUG] Drawing crosses overlay (only_active={only_active}) at idx={active_idx} over coords: {coords}")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape

    for i, (nx, ny) in enumerate(coords):
        if only_active and i != active_idx:
            continue
        px = int(nx * w)
        py = int(ny * h)
        # blue cross in BGR
        color = (255, 0, 0)
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
    """
    Stage 1 (“rgb” mode)   : talk to remote AI to get candidate points.
    Stage 2 (“confirm” mode): user types ‘yes’ or something else.
                              If ‘yes’ we FINALISE and also return object_info.
    """
    data  = request.json or {}
    text  = data.get("text", "").strip().lower()
    global g_last_raw_frame

    # ------------------------------------------------------------------
    #  STAGE 1  Get candidate points from remote AI
    # ------------------------------------------------------------------
    if g_state["mode"] == "rgb":
        seg_input  = data.get("seg_input", "on")
        seg_frame  = data.get("seg_frame", None)
        candidate  = get__current_frame(seg_input, seg_frame, text)   # calls remote AI

        # Error handling
        lowered = candidate.lower()
        if ("no segmentation frame available" in lowered or
            "failed to decode" in lowered or
            "seg window input is off" in lowered):
            return jsonify({"reply": candidate})

        # Draw overlay with ONLY the first (blue) cross
        pil_img = Image.open(io.BytesIO(g_last_raw_frame)).convert("RGB")
        raw_b64 = pil_to_b64(pil_img)
        overlay = (
            draw_points_on_image(pil_img, candidate, active_idx=0, only_active=True)
            if candidate else pil_img
        )
        overlay_b64 = pil_to_b64(overlay)

        # Switch to confirm mode
        g_state.update({
            "mode"       : "confirm",
            "points_str" : candidate
        })

        return jsonify({
            "reply"        : candidate,
            "raw_frame"    : raw_b64,
            "overlay_frame": overlay_b64
        })

    # ------------------------------------------------------------------
    #  STAGE 2  User replies while in “confirm” mode
    # ------------------------------------------------------------------
    elif g_state["mode"] == "confirm":
        if text in ("yes", "y"):
            # --- a) run SAM at best point --------------------------------
            coords = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", g_state["points_str"])
            bx, by = map(float, coords[0]) if coords else (0.5, 0.5)

            pil_img   = Image.open(io.BytesIO(g_last_raw_frame)).convert("RGB")
            final_seg = sam_handler.run_sam_overlay(
                pil_img, [(bx, by)], active_idx=0, max_dim=640
            )
            final_b64 = pil_to_b64(final_seg)

            # --- b) depth geometry --------------------------------------
            object_info = None
            if depth_handler.start_realsense():
                _, depth_arr = depth_handler.get_realsense_frames()
                mask_bool    = sam_handler.get_last_mask()
                object_info  = depth_handler.calculate_object_info(mask_bool, depth_arr)

            # --- c) reset state & return --------------------------------
            g_state.update({"mode":"rgb", "points_str":""})
            return jsonify({
                "reply"       : "Segmentation confirmed.",
                "seg_frame"   : final_b64,
                "object_info" : object_info
            })

        else:
            # User rejected – reset and allow new question
            g_state.update({"mode":"rgb", "points_str":""})
            return jsonify({"reply": "Candidate not confirmed. Returning to RGB mode."})

    # ------------------------------------------------------------------
    #  Unknown mode fallback
    # ------------------------------------------------------------------
    return jsonify({"reply": f"Unknown mode: {g_state['mode']}"})

@app.route("/process_seg", methods=["POST"])
def process_seg_endpoint():
    """
    FIRST CLICK  (freeze):
        • Runs SAM at the click → green mask
        • Computes 3-D centre / distance / W×H immediately (RealSense)
        • Returns overlay image + object_info

    SECOND CLICK (unfreeze):
        • Resets segmentation state (handled client-side with /reset_seg)
    """
    global g_last_raw_frame

    # ---------- 0  decode incoming frame ----------
    data      = request.json or {}
    frame_b64 = data.get("frame", "")
    if not frame_b64:
        return jsonify({"frame": ""})

    raw_bytes        = base64.b64decode(frame_b64)
    g_last_raw_frame = raw_bytes
    pil_img          = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    clicked_x = data.get("clicked_x")
    clicked_y = data.get("clicked_y")

    # ------------------------------------------------------------------
    # 1  SEGMENTATION PATH  (user clicked inside the image)
    # ------------------------------------------------------------------
    if clicked_x is not None and clicked_y is not None:
        click_coords = (float(clicked_x), float(clicked_y))

        # 1-a  SAM mask & overlay
        sam_img = sam_handler.run_sam_overlay(
            pil_img, [click_coords], active_idx=0, max_dim=640
        )

        # 1-b  Depth-based geometry (if RealSense available)
        object_info = None
        if depth_handler.start_realsense():                     # no-op if already started
            _, depth_arr = depth_handler.get_realsense_frames()
            mask_bool    = sam_handler.get_last_mask()
            object_info  = depth_handler.calculate_object_info(mask_bool, depth_arr)

        # 1-c  Optional blue crosses (robot candidate points)
        output_img = (
            draw_points_on_image(
                sam_img, g_state["points_str"], g_state["active_cross_idx"]
            )
            if g_state["points_str"]
            else sam_img
        )

    # ------------------------------------------------------------------
    # 2  DEPTH-ANYTHING PATH  (no click, e.g. RealSense RGB stream)
    # ------------------------------------------------------------------
    else:
        object_info     = None
        depth_processed = depth_handler.run_depth_anything(pil_img)
        output_img      = (
            draw_points_on_image(
                depth_processed, g_state["points_str"], g_state["active_cross_idx"]
            )
            if g_state["points_str"]
            else depth_processed
        )

    # ---------- 3  send back frame (+ geometry) ----------
    out_b64 = pil_to_b64(output_img)
    return jsonify({
        "frame"       : out_b64,
        "object_info" : object_info       # may be None
    })

@app.route("/process_realsense_seg", methods=["POST"])
def process_realsense_seg():
    with depth_lock:
        if not depth_handler.start_realsense():
            return jsonify({"error": "Failed to start RealSense"}), 500
        frames = depth_handler.get_realsense_frames()
        if frames is None:
            return jsonify({"error": "still loading frames"}), 500
        color_arr, _ = frames  # get color frame; ignore depth frame
        if color_arr is None:
            return jsonify({"error": "No color frame available"}), 500
        # Convert from BGR to RGB.
        rgb_arr = cv2.cvtColor(color_arr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_arr)
        return jsonify({"frame": pil_to_b64(pil_img)})

@app.route("/process_depth", methods=["POST"])
def process_depth():
    """
    Processes an image for depth information using Depth Anything logic and returns a clean output.
    """
    data = request.json
    mode = data.get("camera_mode", "off")
    local_idx = data.get("local_idx", -1)
    frame_b64 = data.get("frame", "")

    print(f"[Depth] Received => mode={mode} local_idx={local_idx}")

    if mode == "off":
        depth_handler.stop_local_camera()
        depth_handler.stop_realsense()
        return jsonify({"frame": ""})

    elif mode in ("default_anything", "sidecam_depth"):
        if not frame_b64:
            return jsonify({"error": "no local frame"}), 400
        raw = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        out_img = depth_handler.run_depth_anything(pil_img)
        return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "local_depth_anything":
        with depth_lock:
            if not depth_handler.start_local_camera(local_idx):
                return jsonify({"error": f"Failed to open local camera {local_idx}"}), 500
            frame = depth_handler.grab_local_frame()
            if frame is None:
                return jsonify({"error": "no local frame captured"}), 500
            pil_img = Image.fromarray(frame[..., ::-1])  # Convert BGR -> RGB
            out_img = depth_handler.run_depth_anything(pil_img)
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "realsense_rgb_anything":
        with depth_lock:
            if not depth_handler.start_realsense():
                return jsonify({"error": "Failed to start RealSense"}), 500
            out_img = depth_handler.realsense_color_to_depthanything()
            if out_img is None:
                return jsonify({"error": "still loading frames"})
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "realsense_depth":
        with depth_lock:
            if not depth_handler.start_realsense():
                return jsonify({"error": "Failed to start RealSense"}), 500
            out_img = depth_handler.realsense_depth_colormap()
            if out_img is None:
                return jsonify({"error": "still loading frames"})
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "image_depth":
        if not frame_b64:
            return jsonify({"error": "no frame provided"}), 400
        raw = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        out_img = depth_handler.run_depth_anything(pil_img)
        return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "other":
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
        return jsonify({"error": f"Unknown camera_mode: {mode}"}), 400
    
@app.route("/reset_seg", methods=["POST"])
def reset_seg():
    """
    Reset segmentation state (abort any pending confirm),
    clear all points, and return to RGB mode.
    """
    g_state["mode"] = "rgb"
    g_state["points_str"] = ""
    g_state["active_cross_idx"] = 0
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    print("[INFO] Starting on :5000 in multi-threaded mode.")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
