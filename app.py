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

REMOTE_POINTS_ENDPOINT = "https://95be-130-225-198-197.ngrok-free.app/predict"
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

def draw_points_on_image(pil_img, points_str, active_idx=-1):
    matches = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", points_str)
    coords = [(float(a), float(b)) for a, b in matches]
    if not coords:
        print("[DEBUG] No coordinates found in points_str. Returning original image.")
        return pil_img
    print(f"[DEBUG] Drawing crosses overlay on image with coordinates: {coords}")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape
    for i, (nx, ny) in enumerate(coords):
        px = int(nx * w)
        py = int(ny * h)
        # Draw cross: black for normal points, red for the active one.
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
    # Stage 1: (RGB mode) Capture seg frame, get candidate points, and overlay them on the raw RGB frame.
    if g_state["mode"] == "rgb":
        # Expect these fields from the client:
        seg_input = data.get("seg_input", "on")   # (e.g., "on" or "off" from the seg dropdown)
        seg_frame = data.get("seg_frame", None)     # Base64 captured seg window frame
        
        # Call helper to get candidate points from the remote endpoint.
        candidate_points = get__current_frame(seg_input, seg_frame, text)
        if ("no segmentation frame available" in candidate_points.lower() or
            "failed to decode" in candidate_points.lower() or
            "seg window input is off" in candidate_points.lower()):
            reply = candidate_points
            return jsonify({"reply": reply})
        
        # Now, instead of processing through Depth Anything, simply use the raw RGB image.
        try:
            pil_img = Image.open(io.BytesIO(g_last_raw_frame)).convert("RGB")
            # Overlay the candidate points onto the raw RGB image.
            candidate_overlay = (
                draw_points_on_image(pil_img, candidate_points, 0)
                if candidate_points else pil_img
            )
            candidate_overlay_b64 = pil_to_b64(candidate_overlay)
            print("[DEBUG] Candidate overlay image prepared from raw RGB frame with candidate points frozen.")
        except Exception as e:
            error_msg = f"Error processing candidate overlay image: {e}"
            print("[ERROR]", error_msg)
            return jsonify({"reply": error_msg})
        
        # Save candidate points and switch to confirm mode.
        g_state["points_str"] = candidate_points
        g_state["active_cross_idx"] = 0
        g_state["mode"] = "confirm"
        reply = (f"Found points: {candidate_points}\n"
                 "Displaying best candidate (from RGB image) on SEG window – confirm?")
        return jsonify({"reply": reply, "seg_frame": candidate_overlay_b64})
    
    # Stage 2: (Confirm mode) Wait for user confirmation to run SAM.
    elif g_state["mode"] == "confirm":
        if text in ["yes", "y"]:
            # Extract candidate point coordinates from the points string.
            candidate_coords = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", g_state["points_str"])
            if candidate_coords:
                best_x, best_y = map(float, candidate_coords[0])
            else:
                best_x, best_y = 0.5, 0.5  # Default to center if none found.
            try:
                pil_img = Image.open(io.BytesIO(g_last_raw_frame)).convert("RGB")
                # Run SAM as if the user had clicked at the best candidate point.
                sam_output = sam_handler.run_sam_overlay(pil_img, [(best_x, best_y)], active_idx=0, max_dim=640)
                final_overlay = (
                    draw_points_on_image(sam_output, g_state["points_str"], g_state["active_cross_idx"])
                    if g_state["points_str"] else sam_output
                )
                final_seg_b64 = pil_to_b64(final_overlay)
            except Exception as e:
                return jsonify({"reply": f"Error running SAM: {e}"})
            g_state["mode"] = "rgb"  # Return to RGB mode for the next command.
            reply = "Segmentation confirmed. Returning to RGB mode."
            return jsonify({"reply": reply, "seg_frame": final_seg_b64})
        else:
            reply = "Candidate not confirmed. Please adjust your command and try again."
            return jsonify({"reply": reply})
    
    # (Legacy branches for depth and sam can remain unchanged.)
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
        return jsonify({"reply": reply})
    
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
        return jsonify({"reply": reply})
    else:
        reply = f"Unknown mode: {g_state['mode']}"
        return jsonify({"reply": reply})

@app.route("/process_seg", methods=["POST"])
def process_seg_endpoint():
    global g_last_raw_frame  # Declare the global variable
    data = request.json
    frame_b64 = data.get("frame", "")
    if not frame_b64:
        return jsonify({"frame": ""})
    
    # Decode the incoming frame and update the global seg frame.
    raw_bytes = base64.b64decode(frame_b64)
    g_last_raw_frame = raw_bytes  
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    
    clicked_x = data.get("clicked_x")
    clicked_y = data.get("clicked_y")
    
    if clicked_x is not None and clicked_y is not None:
        # When a click is provided, run SAM on the original seg frame.
        click_coords = (float(clicked_x), float(clicked_y))
        sam_output = sam_handler.run_sam_overlay(pil_img, [click_coords], active_idx=0, max_dim=640)
        # Overlay robot points on the SAM output (if any) and use that as final output.
        output_img = draw_points_on_image(sam_output, g_state["points_str"], g_state["active_cross_idx"]) if g_state["points_str"] else sam_output
    else:
        # Otherwise, process the raw seg frame through Depth Anything;
        # This gives you a static depth-processed image (with best selected point in blue).
        depth_processed = depth_handler.run_depth_anything(pil_img)
        # Now overlay the robot-selected crosses onto this processed image.
        output_img = draw_points_on_image(depth_processed, g_state["points_str"], g_state["active_cross_idx"]) if g_state["points_str"] else depth_processed

    out_b64 = pil_to_b64(output_img)
    return jsonify({"frame": out_b64})

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

if __name__ == "__main__":
    print("[INFO] Starting on :5000 in multi-threaded mode.")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
