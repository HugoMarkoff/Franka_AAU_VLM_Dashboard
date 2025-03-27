# Franka_AAU_VLM_Dashboard

A localhost dashboard for communicating with an AI endpoint and displaying a live video feed. This project demonstrates a pipeline combining remote point (RoboPoint) detection with local depth estimation and segmentation (using SAM2 and Depth Anything). The built-in chat interface is currently basic and primarily designed to handle simple yes/no responses for navigating the pipeline.
The chatbot could be replaced by a LLM/agent/resoning model with instructions of how to activate different python commands depending on resoning from users input

---

## Features

- **Live Video Feed:** Displays real-time video from your camera(s).
- **Remote Point Detection:** Sends a captured frame and instruction to a remote endpoint to detect object points.
- **Depth Estimation:** Runs a depth-estimation pipeline locally and overlays detected points.
- **Segmentation & Tracking:** Uses SAM2 to segment an object based on a selected point.
- **Interactive Chat:** A primitive chatbot interface to switch between modes (RGB, Depth, Tracking) via simple text commands (e.g., "yes", "no", "next", "prev").

---

## Test Setup - may work flawlessly on other systems too

- **Operating System:** Windows 11 (tested on a system with an NVIDIA 4070 laptop GPU)
- **Python Version:** Python 3.11

---

## Installation

### 1. Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/HugoMarkoff/Franka_AAU_VLM_Dashboard.git
cd Franka_AAU_VLM_Dashboard
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment. For example:

- **On Windows:**

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **On Linux:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Requirements

Inside the root folder of the repository, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Install Additional Packages

Some packages are not available on PyPI and must be installed manually:

- **SAM2 (Segment Anything Model 2):**

  ```bash
  python -m pip install git+https://github.com/facebookresearch/sam2.git
  ```

- **Depth Anything:**

  Clone and install the Depth Anything V2 repository:

  ```bash
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git
  cd Depth-Anything-V2
  pip install -r requirements.txt
  cd ..
  ```

---

## Running the Application

### 1. Set Up the AI Server (Remote Endpoint)

On the AI server, navigate to the project directory (`/data/Hugo/RoboPointDemo/`) and activate its virtual environment:

```bash
cd /data/Hugo/RoboPointDemo/
source RoboPointVenv/bin/activate
```

Then run the server code  `main.py`). The server will load the necessary models and create a tunnel. You will be provided with a tunnel address (e.g., `https://ef4e-130-225-198-197.ngrok-free.app`).  
Make sure that the `REMOTE_POINTS_ENDPOINT` in your client code (in `app.py`) is updated with this address appended by `/predict`. For example:

```
REMOTE_POINTS_ENDPOINT = "https://ef4e-130-225-198-197.ngrok-free.app/predict"
```

### 2. Run the Client Application

From the repository root (ensure you’re in the virtual environment):

```bash
python app.py
```

You should see a message similar to:

```
 * Running on http://127.0.0.1:5000
```

Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to view the dashboard.

---

## Usage Notes

- **Interactive Chat:**  
  The built-in chat is primitive. It primarily listens for yes/no responses to transition between pipeline stages (RGB → Depth → SAM & Tracking).  
  - In **RGB mode**, type an instruction (e.g., "mug"). The dashboard sends a frame to the remote server and displays detected red crosses.
  - Confirm with “yes” to switch to **Depth mode**, where a single depth map is generated and reused.
  - Confirm again with “yes” in Depth mode to run SAM on the active point.

---

## Troubleshooting

- **Remote Endpoint:**  
  Ensure that the tunnel URL provided by the AI server is correctly set in `REMOTE_POINTS_ENDPOINT` in `app.py`.

---

## License

[MIT License](LICENSE)  

---

