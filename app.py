import os
import io
import re
import cv2
import base64
import numpy as np
import torch
import requests
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS
from transformers import pipeline

from sam2.sam2_image_predictor import SAM2ImagePredictor  

app = Flask(__name__)
CORS(app)

# Enter the actual remote access point here. We get a print from the AI server once booted up - paste the address here.
REMOTE_POINTS_ENDPOINT = "https://ef4e-130-225-198-197.ngrok-free.app/predict"  # Where we get initial points

#Tring to initialize cuda - if it uses cpu the framerate will be extremely slow
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", device)

#Initializing the depth anything and segment anything models.
depth_pipeline = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-base-hf', device=device)
sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
print("[INFO] SAM2 and Depth Pipeline loaded.")

# Global state for remebering which steps in the "pipeline" we are at to change the view in teh camera "window"
g_state = {
    "mode": "rgb",      # "rgb" => normal camera, "depth" => depth mode, "sam" => SAM overlay
    "points_str": "",   # Original points from server in string format: "[(0.1,0.2),(...)]"
    "active_cross_idx": 0,  # Which cross is "active" (blue)
    "last_instruction": "",
    "pil_image": None,  # Store the last captured PIL image from the user (for reference)
}

# For real-time streaming, westore the last raw frame from the browser
g_last_raw_frame = None  


# The HTML and JS code for the localhost UI (Mostly written by my friend GTP :D)
INDEX_HTML = r"""
<!DOCTYPE html>
<html>
<head>
  <title>Depth & SAM on the Client</title>
  <style>
    body {
      font-family: Arial, sans-serif; background: #f0f2f5;
      display: flex; justify-content: center; align-items: center;
      height: 100vh; margin: 0;
    }
    #container {
      background: #fff; border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      width: 900px; max-width: 95%;
      padding: 20px;
      position: relative;
    }
    #videoElement {
      width: 100%; background: #000; border-radius: 6px; max-height: 540px;
    }
    #chatContainer {
      margin-top: 20px;
      border: 1px solid #ddd;
      border-radius: 10px;
      height: 300px;
      display: flex; flex-direction: column;
    }
    #chatMessages {
      flex: 1; padding: 10px; overflow-y: auto; border-bottom: 1px solid #ddd;
    }
    #chatInputContainer { display: flex; }
    #chatInput {
      flex: 1; padding: 10px; border: none; outline: none;
      border-radius: 0 0 0 10px; font-size: 16px;
    }
    #sendButton {
      padding: 10px 20px; background-color: #007BFF; color: #fff;
      border: none; cursor: pointer; border-radius: 0 0 10px 0; font-size: 16px;
    }
    #sendButton:hover { background-color: #0056b3; }
    #cameraSelect {
      position: absolute; top: 10px; left: 10px; z-index: 999;
      padding: 8px; border-radius: 4px;
    }
  </style>
</head>
<body>
  <div id="container">
    <select id="cameraSelect"></select><br/>
    <!-- We'll show either a normal <video> for the raw webcam or an <img> for the processed frames -->
    <img id="videoElement" />

    <div id="chatContainer">
      <div id="chatMessages"></div>
      <div id="chatInputContainer">
        <input type="text" id="chatInput" placeholder="Type your instruction..." />
        <button id="sendButton">Send</button>
      </div>
    </div>
  </div>

<script>
// We'll use a "capture loop" to constantly get frames from the user's webcam, 
// send them to the Flask server, and get back either the same frame or a processed one.

let localStream = null;
let selectedDeviceId = null;
let isCapturing = false;

// On load, populate cameras
window.addEventListener('DOMContentLoaded', async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter(d => d.kind === 'videoinput');
  const cameraSelect = document.getElementById('cameraSelect');
  cameraSelect.innerHTML = '';
  videoDevices.forEach((dev, idx) => {
    const opt = document.createElement('option');
    opt.value = dev.deviceId;
    opt.text = dev.label || 'Camera ' + (idx + 1);
    cameraSelect.appendChild(opt);
  });
  cameraSelect.addEventListener('change', async () => {
    stopCapture();
    selectedDeviceId = cameraSelect.value;
    await startCapture(selectedDeviceId);
  });
  if (videoDevices.length > 0) {
    selectedDeviceId = videoDevices[0].deviceId;
    cameraSelect.value = selectedDeviceId;
    await startCapture(selectedDeviceId);
  }
});

// Start capturing frames
async function startCapture(deviceId) {
  if (isCapturing) return;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ 
      video: { deviceId: { exact: deviceId } }
    });
  } catch (err) {
    console.error("Camera error:", err);
    return;
  }
  isCapturing = true;
  captureLoop();
}

// Stop capturing frames
function stopCapture() {
  isCapturing = false;
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
}

// The main loop that sends frames to the server
async function captureLoop() {
  if (!isCapturing) return;
  try {
    // 1) Grab a frame from localStream
    const frameB64 = await grabFrameFromStream(localStream);

    // 2) Send to /process_frame to get either the raw or the processed frame back
    const resp = await fetch("/process_frame", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ frame: frameB64 })
    });
    if (!resp.ok) throw new Error("Frame processing failed, status=" + resp.status);
    const data = await resp.json();

    // data.frame is a base64-encoded JPEG
    // We'll display it in the <img id="videoElement">
    const videoElement = document.getElementById('videoElement');
    videoElement.src = "data:image/jpeg;base64," + data.frame;
  } catch (err) {
    console.error(err);
  }
  // Next iteration
  if (isCapturing) {
    // 10 FPS? -> wait 100ms
    setTimeout(captureLoop, 100);
  }
}

// Convert the next webcam frame to base64
async function grabFrameFromStream(stream) {
  return new Promise((resolve, reject) => {
    // We use a <video> + <canvas> approach
    const videoTemp = document.createElement('video');
    videoTemp.autoplay = true;
    videoTemp.srcObject = stream;
    videoTemp.addEventListener('loadeddata', () => {
      const canvas = document.createElement('canvas');
      canvas.width = videoTemp.videoWidth;
      canvas.height = videoTemp.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoTemp, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL('image/jpeg').split(',')[1]; // remove prefix
      resolve(b64);
    });
    videoTemp.addEventListener('error', (e) => {
      reject(e);
    });
  });
}

// ----------- Chat & user instructions --------------
function addChatMessage(sender, msg) {
  const chatMessages = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.style.marginBottom = "10px";
  div.innerHTML = `<strong>${sender}:</strong> ${msg}`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
  const chatInput = document.getElementById('chatInput');
  const userText = chatInput.value.trim();
  if (!userText) return;
  chatInput.value = '';
  addChatMessage('You', userText);

  // Send to /chat
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ text: userText })
    });
    if (!resp.ok) throw new Error("Chat request failed, status=" + resp.status);
    const data = await resp.json();
    if (data.reply) {
      addChatMessage("AAU Agent", data.reply);
    }
  } catch(err) {
    console.error(err);
    addChatMessage("AAU Agent", "Error: " + err.toString());
  }
}

document.getElementById('sendButton').addEventListener('click', sendMessage);
document.getElementById('chatInput').addEventListener('keyup', (ev) => {
  if (ev.key === 'Enter' || ev.keyCode === 13) {
    sendMessage();
  }
});
</script>
</body>
</html>
"""

####Flask routes, how we basically both communicate internally and externally (server) to do different actions###

@app.route('/')
def index():
    return INDEX_HTML

#Extremely primitive "Chatbot" which looks for commands in a specific order like yes/no 
@app.route('/chat', methods=['POST'])
def chat():
    """
    User typed something in the chat -> interpret it.
    We'll handle "yes/no" or "2 to the left" or normal instructions, etc.
    """
    global g_state
    data = request.json
    text = data.get("text", "").strip().lower()
    reply = ""
    if g_state["mode"] == "rgb":
        if text == "yes" or text == "y":
            g_state["mode"] = "depth"
            reply = "Great. Switching to Depth mode... (Look at the depth feed now.)"
        elif text == "no" or text == "n":
            reply = "Okay, please provide a new instruction or refine your request."
        elif re.match(r"^.*(left|right|up|down).*$", text):
            reply = "We haven't got any crosses yet or you haven't confirmed them. Try giving an instruction first."
        else:
            g_state["last_instruction"] = text
            reply = "Instruction noted. We'll detect points on the next frame."
    elif g_state["mode"] == "depth":
        if text == "yes" or text == "y":
            g_state["mode"] = "sam"
            reply = "Switching to SAM mode. We'll do segmentation on the live feed now."
        elif text == "no" or text == "n":
            reply = "Okay, we remain in Depth mode. Please specify how to move the cross or give a new instruction."
        else:
            m = re.search(r"(\d+)\s*to\s*the\s*(left|right|up|down)", text)
            if m:
                dist = int(m.group(1))
                direction = m.group(2)
                move_active_cross(direction, dist)
                reply = f"Moved the active cross {dist} to the {direction}. See updated Depth feed."
            else:
                reply = "Not sure how to interpret. Try '2 to the left' or 'yes/no'."
    elif g_state["mode"] == "sam":
        if text == "yes" or text == "y":
            reply = "Great! The object is now segmented with SAM."
        elif text == "no" or text == "n":
            reply = "Feel free to refine or move the cross. (But full dynamic re-segmentation is not implemented in this demo.)"
        else:
            m = re.search(r"(\d+)\s*to\s*the\s*(left|right|up|down)", text)
            if m:
                dist = int(m.group(1))
                direction = m.group(2)
                move_active_cross(direction, dist)
                reply = f"Moved cross {dist} to the {direction}. We'll re-run SAM on next frames."
            else:
                reply = "Ok. Try 'yes/no' or '2 to the left'."
    else:
        reply = "Unknown mode."

    return jsonify({"reply": reply})


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    The browser constantly sends frames here.
    We'll check g_state["mode"]:
      - 'rgb': Possibly run remote detection if new instruction arrived, draw crosses in red.
      - 'depth': Run local Depth pipeline, draw crosses in red except the active one in blue.
      - 'sam': Run local SAM pipeline, show segmented overlay.

    Return a base64-encoded JPEG of the processed frame.
    """
    global g_state, g_last_raw_frame

    data = request.json
    frame_b64 = data.get("frame", "")
    if not frame_b64:
        return jsonify({"frame": ""})

    # Decode to PIL
    raw_bytes = base64.b64decode(frame_b64)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    g_last_raw_frame = raw_bytes

    # Depending on mode, do different things - super primitive logic
    if g_state["mode"] == "rgb":
        if g_state["last_instruction"]:
            instr = g_state["last_instruction"]
            g_state["last_instruction"] = ""
            points_str = call_remote_for_points(frame_b64, instr)
            g_state["points_str"] = points_str
            g_state["active_cross_idx"] = 0  # reset
        result_img = draw_points_on_image(pil_img, g_state["points_str"], active_idx=-1)
        return jsonify({"frame": pil_to_base64(result_img)})
    elif g_state["mode"] == "depth":
        depth_map = run_depth_anything(pil_img)
        result_img = draw_points_on_image(depth_map, g_state["points_str"], active_idx=g_state["active_cross_idx"])
        return jsonify({"frame": pil_to_base64(result_img)})
    elif g_state["mode"] == "sam":
        result_img = run_sam_overlay(pil_img, g_state["points_str"], g_state["active_cross_idx"])
        return jsonify({"frame": pil_to_base64(result_img)})
    return jsonify({"frame": frame_b64})



### HELPER FUNCTIONS ###

def call_remote_for_points(frame_b64, instruction):
    """Posts {image=..., instruction=...} to REMOTE_POINTS_ENDPOINT, returns the server's 'result' string."""
    try:
        resp = requests.post(
            REMOTE_POINTS_ENDPOINT,
            json={"image": frame_b64, "instruction": instruction},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        points_str = data.get("result", "")
        return points_str
    except Exception as e:
        print("[ERROR] remote points:", e)
        return ""

def draw_points_on_image(pil_img, points_str, active_idx=-1):
    """
    points_str is something like "[(0.1,0.2),(0.3,0.4)]".
    We draw them on the PIL image. If active_idx >= 0, that cross is blue, others red.
    We'll just convert to OpenCV for convenience, then back to PIL.
    """
    coords = parse_points(points_str)
    if not coords:
        return pil_img

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape

    for i, (nx, ny) in enumerate(coords):
        px = int(nx * w)
        py = int(ny * h)
        color = (0, 0, 255)  # red
        if i == active_idx:
            color = (255, 0, 0)  # blue
        draw_cross(cv_img, px, py, color=color, size=10, thickness=2)

    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def parse_points(points_str):
    """Parse '[(0.1,0.2),(0.3,0.4)]' into list of (nx, ny)."""
    matches = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", points_str)
    coords = []
    for m in matches:
        x = float(m[0])
        y = float(m[1])
        coords.append((x, y))
    return coords

def draw_cross(cv_img, x, y, color=(0,0,255), size=10, thickness=2):
    cv2.line(cv_img, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(cv_img, (x + size, y - size), (x - size, y + size), color, thickness)

def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def run_depth_anything(pil_img):
    """Return a PIL Image that is the depth colormap of pil_img."""
    depth_out = depth_pipeline(pil_img)
    depth_arr = np.array(depth_out['depth'], dtype=np.float32)
    depth_norm = cv2.normalize(depth_arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return Image.fromarray(depth_color[..., ::-1])  # BGR->RGB

def run_sam_overlay(pil_img, points_str, active_idx):
    """Run SAM using the 'active_idx' cross as the main prompt, overlay on the original image."""
    coords = parse_points(points_str)
    if not coords or active_idx < 0 or active_idx >= len(coords):
        return pil_img

    w, h = pil_img.size
    (nx, ny) = coords[active_idx]
    px = int(nx * w)
    py = int(ny * h)

    # Actually run SAM2 here 
    sam2_predictor.set_image(np.array(pil_img))  # big overhead each frame, but for demo
    point_coords = np.array([[px, py]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)
    with torch.no_grad():
        masks, scores, logits = sam2_predictor.predict(point_coords=point_coords, point_labels=point_labels)
    if masks is None or len(masks) == 0:
        return pil_img

    mask = masks[0]
    # Overly simplistic overlay: blend foreground in green, background darker 
    cv_img = np.array(pil_img).astype(np.uint8)
    for r in range(cv_img.shape[0]):
        for c in range(cv_img.shape[1]):
            if mask[r, c]:
                cv_img[r, c] = (0.5 * cv_img[r, c] + 0.5 * np.array([0,255,0])).astype(np.uint8)
            else:
                cv_img[r, c] = (0.8 * cv_img[r, c]).astype(np.uint8)

    # Also draw the cross - remove if we just want the mask later
    draw_cross(cv_img, px, py, color=(255,0,0), size=10, thickness=2)
    return Image.fromarray(cv_img)

# We do a small shift in the normalized coordinate maybe better to actually change to another marked point instead in the future
def move_active_cross(direction, dist):
    """Move the currently active cross by 'dist' in the specified direction (left/right/up/down)."""
    coords = parse_points(g_state["points_str"])
    idx = g_state["active_cross_idx"]
    if not coords or idx < 0 or idx >= len(coords):
        return
    nx, ny = coords[idx]
    if direction == "left":
        shift = dist * 0.01
        nx -= shift
        nx = max(0.0, nx)
    elif direction == "right":
        shift = dist * 0.01
        nx += shift
        nx = min(nx, 1.0)
    elif direction == "up":
        shift = dist * 0.01
        ny -= shift
        ny = max(0.0, ny)
    elif direction == "down":
        shift = dist * 0.01
        ny += shift
        ny = min(ny, 1.0)

    coords[idx] = (nx, ny)
    new_str = "[" + ", ".join(f"({c[0]:.3f}, {c[1]:.3f})" for c in coords) + "]"
    g_state["points_str"] = new_str

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
