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
import flask
from flask import Response, stream_with_context, request
import httpx, asyncio
from utils.sam_handler import SamHandler
from utils.depth_handler import DepthHandler, RS_OK
from utils.robopoints_handler import RoboPointsHandler
import os

app = Flask(__name__)
CORS(app)

ROBOPOINT_IMAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOPOINT_IMAGE_COUNTER_FILE = os.path.join(ROBOPOINT_IMAGE_DIR, "robopoint_image_counter.txt")

# in settings.py or top of app.py
NGROK_BASE = "https://c2b4-130-225-198-197.ngrok-free.app"

REMOTE_POINTS_ENDPOINT = f"{NGROK_BASE}/predict"      # RoboPoint
LLM_STREAM             = f"{NGROK_BASE}/chat-stream"  # token stream
LLM_FULL               = f"{NGROK_BASE}/chat"         # full reply

# top-of-file imports
from utils.llm_handler import LLMHandler

llm = LLMHandler(stream_url=LLM_STREAM, full_url=LLM_FULL)

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

POINT_RE = re.compile(r"\(([0-9.+-eE]+)\s*,\s*([0-9.+-eE]+)\)")

def _points_from_string(raw: str):
    """
    Turn RoboPoint's 'result' string into a list of (x,y) floats in [0,1].

    Accepts:
        '[(0.37, 0.61)]'
        '(0.37,0.61)'
        '0.37,0.61'
        'x=0.37 y=0.61'
        '[(0.37,0.61), (0.12,0.88)]'
    Returns [] if **no** valid pair is found.
    """
    raw = raw.strip()
    # 1) quick normalisation of the simplest '0.37,0.61' case
    if raw.count(',') == 1 and raw.count('(') == 0:
        try:
            x, y = map(float, raw.split(','))
            return [(x, y)]
        except Exception:
            pass
    # 2) general regex search – tolerant of any wrapper text
    matches = POINT_RE.findall(raw)
    return [tuple(map(float, xy)) for xy in matches]

def get_next_image_index():
    """Get and increment the persistent image index."""
    if not os.path.exists(ROBOPOINT_IMAGE_COUNTER_FILE):
        idx = 1
    else:
        with open(ROBOPOINT_IMAGE_COUNTER_FILE, "r") as f:
            idx = int(f.read().strip() or "1")
    with open(ROBOPOINT_IMAGE_COUNTER_FILE, "w") as f:
        f.write(str(idx + 1))
    return idx


# ────────────────────────────────────────────────────────────────────
#  /exec_action  – returns placement + full geometry string
# ────────────────────────────────────────────────────────────────────
@app.route("/exec_action", methods=["POST"])
def exec_action():
    """
    Enhanced version that handles single action from multi-step sequence
    """
    global g_last_raw_frame

    data        = request.json or {}
    block_txt   = data.get("action_block", "")
    frame_b64   = data.get("seg_frame", "")

    # 0) Decode frozen frame
    try:
        g_last_raw_frame = base64.b64decode(frame_b64)
    except Exception as e:
        return jsonify({"error": f"Invalid seg_frame: {e}"}), 400
    pil_img = Image.open(io.BytesIO(g_last_raw_frame)).convert("RGB")
    # --- SAVE THE IMAGE ---
    img_idx = get_next_image_index()
    img_path = os.path.join(ROBOPOINT_IMAGE_DIR, f"{img_idx}.jpg")
    pil_img.save(img_path, "JPEG")
    print(f"[INFO] Saved RoboPoint image as {img_path}")

    # 1) Parse action block
    request_match = re.search(r"RoboPoint\s*Request:\s*([^;\n|]+)", block_txt, flags=re.I)
    action_match = re.search(r"Action\s*Request:\s*([^;\n|]+)", block_txt, flags=re.I)
    
    if not request_match or not action_match:
        return jsonify({"error": "Malformed [ACTION] block"}), 400
    
    obj_phrase = request_match.group(1).strip()
    action_type = action_match.group(1).strip().lower()
    
    # Extract object and location
    if " at " in obj_phrase:
        object_name, location = obj_phrase.split(" at ", 1)
        instruction = f"{action_type} the {object_name.strip()} at {location.strip()}"
    else:
        object_name = obj_phrase
        instruction = f"{action_type} the {object_name.strip()}"

    # 2) Call RoboPoint
    rp_result = robo_handler.call_remote_for_points(frame_b64, instruction)
    print(f"[RoboPoint result] {rp_result}")
    pts = _points_from_string(rp_result)

    if not pts:
        err = f"RoboPoint returned no usable coordinates for '{instruction}'."
        return jsonify({"error": err, "rp_result": rp_result}), 502

    # Use the first candidate point
    x0, y0 = pts[0]

    # 3) Determine placement description
    def coarse_placement(nx: float, ny: float) -> str:
        col = "left" if nx < 0.33 else "right" if nx > 0.66 else "centre"
        row = "top"  if ny < 0.33 else "bottom" if ny > 0.66 else "middle"
        return f"{row}-{col}" if (row != "middle" or col != "centre") else "centre"

    placement = coarse_placement(x0, y0)

    # 4) SAM overlay
    seg_img = sam_handler.run_sam_overlay(pil_img, pts, active_idx=0)
    seg_img = draw_points_on_image(seg_img, str(pts), active_idx=0, only_active=True)
    seg_b64 = pil_to_b64(seg_img)

    # 5) Depth geometry (only for pick operations or when available)
    object_info = None
    geometry_msg = "(No geometry available)"
    
    if depth_handler.start_realsense():
        _, depth_arr = depth_handler.get_realsense_frames()
        mask_bool    = sam_handler.get_last_mask()
        object_info  = depth_handler.calculate_object_info(mask_bool, depth_arr)
        
        if object_info:
            cx, cy, cz = object_info["center_xyz_m"]
            if "pick" in action_type:
                # Full geometry for pick operations
                geometry_msg = (
                    f"Center: {cx:.3f}, {cy:.3f}, {cz:.3f} m; "
                    f"distance {object_info['distance_m']:.3f} m; "
                    f"W×H {object_info['width_m']*100:.1f}×"
                    f"{object_info['height_m']*100:.1f} cm"
                )
            else:
                # Just center point and height for place operations
                geometry_msg = f"Center: {cx:.3f}, {cy:.3f}, {cz:.3f} m"

    # 6) Human-readable found message
    found_msg = f"{instruction} located at {placement} ({x0:.2f}, {y0:.2f})"

    # 7) Respond
    return jsonify({
        "seg_frame"   : seg_b64,
        "found_msg"   : found_msg,
        "geometry_msg": geometry_msg,
        "placement"   : placement,
        "object_info" : object_info,
        "rp_result"   : rp_result,
        "action_type" : action_type
    })


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

def broadcast_to_llm_console(token):
    # TEMP: Just print. You should route to a WebSocket or SSE for true real-time in the browser.
    print(f"[LLM Reasoning] {token}")


# ---------------------------------------------------------------------------
#  STREAMING CHAT ENDPOINT  (drop-in replacement)
# ---------------------------------------------------------------------------
@app.route("/chat-stream", methods=["POST"])
def chat_stream():
    """
    Streams two kinds of Server-Sent Events (SSE):

        event: reasoning   data: <incremental text inside <think>...</think>>
        event: answer      data: <final answer (after </think>)>

    Nothing else is ever sent – no “[DONE]”.
    """
    import httpx, asyncio
    text = (request.json or {}).get("text", "").strip()

    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                llm.stream_url,                # remote FastAPI streaming endpoint
                json={"prompt": text},
            ) as r:

                in_think     = False           # True ⇢ we are inside <think>...</think>
                answer_parts = []              # collect answer after </think>

                async for raw in r.aiter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue

                    tok = raw[5:].lstrip()

                    # --- handle open/close tags ---------------------------------
                    while tok:
                        if not in_think:
                            # look for <think>
                            start = tok.find("<think>")
                            if start == -1:
                                # no reasoning begins here – treat as answer text
                                answer_parts.append(tok)
                                tok = ""
                            else:
                                # jump past "<think>"
                                tok   = tok[start + 7 :]
                                in_think = True
                        else:
                            # we are inside reasoning – look for </think>
                            end = tok.find("</think>")
                            if end == -1:
                                # no end yet → whole chunk is reasoning
                                yield f"event: reasoning\ndata: {tok}\n\n"
                                tok = ""
                            else:
                                # stream up to end, then switch to answer mode
                                reasoning_chunk = tok[:end]
                                if reasoning_chunk:
                                    yield f"event: reasoning\ndata: {reasoning_chunk}\n\n"
                                tok       = tok[end + 8 :]  # jump past "</think>"
                                in_think  = False
                    # ----------------------------------------------------------------

                # after upstream closes: flush whatever answer we buffered
                final_answer = "".join(answer_parts).strip()
                if final_answer:
                    yield f"event: answer\ndata: {final_answer}\n\n"

    # ---------- bridge async generator → Flask streaming Response ---------------
    def event_stream():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agen = relay().__aiter__()
        try:
            while True:
                chunk = loop.run_until_complete(agen.__anext__())
                yield chunk
        except StopAsyncIteration:
            pass

    return Response(stream_with_context(event_stream()),
                    mimetype="text/event-stream")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    text = data.get("text", "").strip()
    frame_b64 = data.get("seg_frame", "")

    # Step 1: Call LLM handler
    thinking_tokens, answer, intent = llm.process(text)

    # Step 2: If not action (just chat), reply as usual
    if (
        intent.get("type") != "action" or
        not intent.get("object") or
        not intent.get("action")
    ):
        return jsonify({"reply": answer, "answer": answer})

    # Step 3: If action is ready, run RoboPoint pipeline!
    # Compose instruction for RoboPoint (e.g., "pick the white cup in the middle of the screen")
    object_phrase = intent["object"]
    placement_str = intent["placement"] or ""
    action_phrase = intent["action"]
    instruction = f"{action_phrase} the {object_phrase} {placement_str}".strip()

    # 3a: Send frame + instruction to RoboPoint
    pts_str = robo_handler.call_remote_for_points(frame_b64, instruction)

    # 3b: Decode returned points, run segmentation & geometry
    pil_img = Image.open(io.BytesIO(base64.b64decode(frame_b64))).convert("RGB")
    coords = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", pts_str)
    bx, by = map(float, coords[0]) if coords else (0.5, 0.5)

    # Segment the best candidate
    final_seg = sam_handler.run_sam_overlay(
        pil_img, [(bx, by)], active_idx=0, max_dim=640
    )
    final_b64 = pil_to_b64(final_seg)

    object_info = None
    if depth_handler.start_realsense():
        _, depth_arr = depth_handler.get_realsense_frames()
        mask_bool = sam_handler.get_last_mask()
        object_info = depth_handler.calculate_object_info(mask_bool, depth_arr)

    # Format geometry for user
    if object_info:
        geometry_msg = (
            f"Center: {', '.join(f'{v:.3f}' for v in object_info['center_xyz_m'])} m; "
            f"distance {object_info['distance_m']:.3f} m; "
            f"W×H {object_info['width_m']*100:.1f}×{object_info['height_m']*100:.1f} cm"
        )
    else:
        geometry_msg = "(No geometry available)"

    # 4: Respond to the user — no ugly [ACTION_READY]!
    return jsonify({
        "reply": "Finding object(s) and distance(s), please wait a few seconds",
        "seg_frame": final_b64,
        "object_info": object_info,
        "geometry_msg": geometry_msg
    })


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
