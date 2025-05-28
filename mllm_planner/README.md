# 🤖 Enhanced Robot Action Planner

A sophisticated 4-step pipeline system that combines local Vision Language Model (VLM) analysis with online Large Language Model (LLM) reasoning to generate executable robot action plans from visual scenes and task descriptions.

## ✨ Key Features

- **🧠 4-Step Pipeline Architecture**: Local vision analysis + online reasoning for optimal performance
- **👁️ Advanced Scene Understanding**: Qwen2.5-VL-7B for precise object detection and spatial relationships
- **🌐 Online LLM Integration**: DeepSeek API through Monica for enhanced logical reasoning
- **📦 Python Module Interface**: Easy import and integration with other systems
- **⚡ Intelligent Caching**: Vision analysis results cached for faster subsequent runs
- **📺 Flexible Output Options**: Terminal display + output capture for monitoring and logging
- **🛡️ Robust Error Handling**: Comprehensive exception handling with clear error messages
- **🔄 Multi-step Task Support**: Handles complex sequential robot tasks

## 🏗️ Architecture Overview

### 4-Step Processing Pipeline

```
📸 Image Input → 👁️ Vision Analysis → ✅ Validation → 🧠 Reasoning → 🤖 Action Plan
     ↓              (Qwen2.5-VL)      (Syntax)     (DeepSeek)    (Executable)
```

1. **STEP 1: Vision Analysis** 🔍
   - **Model**: Qwen2.5-VL-7B-Instruct (Local GPU)
   - **Function**: Object detection, spatial relationship analysis
   - **Output**: Structured scene description with objects and relationships

2. **STEP 2: Syntax Validation** ✅
   - **Function**: Data validation, consistency checking
   - **Process**: Object name standardization, spatial logic verification
   - **Output**: Cleaned and validated scene understanding

3. **STEP 3: Reasoning Refinement** 🧠
   - **Model**: DeepSeek API (Online)
   - **Function**: Logical consistency enhancement, ambiguity resolution
   - **Output**: Refined spatial understanding optimized for planning

4. **STEP 4: Action Planning** 🚀
   - **Model**: DeepSeek API (Online)
   - **Function**: Generate executable robot action sequences
   - **Output**: Formatted action plan with spatial constraint handling

## 📋 Requirements

### System Requirements
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system RAM (32GB+ recommended)
- **Python**: 3.8 or higher
- **OS**: Linux (tested on Ubuntu 20.04+)

### Dependencies Installation

```bash
# Core dependencies
pip install torch torchvision transformers
pip install qwen-vl-utils pillow requests
pip install accelerate bitsandbytes

# Optional: For conda environment
conda create -n robot_planner python=3.9
conda activate robot_planner
```

### API Configuration

You need a Monica API key for DeepSeek access:

```python
# In online_mllm_planner.py
MONICA_API_KEY = "your-monica-api-key-here"
```

Get your API key from [Monica.im](https://monica.im)

## 🚀 Quick Start

### Basic Usage

```python
from online_mllm_planner import robot_action_planner

# Simple task
result = robot_action_planner(
    task_description="pick up the pen",
    image_path="/path/to/scene.jpg"
)

if result["success"]:
    print(f"Objects detected: {result['detected_objects']}")
    print(f"Action plan: {result['action_plan']}")
else:
    print("Planning failed")
```

### Advanced Usage with Output Capture

```python
from online_mllm_planner import robot_action_planner

# Multi-step task with full logging
result = robot_action_planner(
    task_description="first pick up the paper, then pick up the banana",
    image_path="./images/complex_scene.png",
    capture_output=True,  # Capture all terminal output
    quiet=False,          # Show progress on terminal
    force_reanalyze=True  # Force fresh analysis
)

# Access results
print(f"Success: {result['success']}")
print(f"Objects: {result['detected_objects']}")
print(f"Plan: {result['action_plan']}")

# Access captured logs for debugging/audit
if 'terminal_output' in result:
    print(f"Captured {len(result['terminal_output'])} characters of logs")
    # Save logs to file for analysis
    with open("planning_log.txt", "w") as f:
        f.write(result['terminal_output'])
```

## 📊 Function Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_description` | `str` | **Required** | Task for robot to perform (e.g., "pick up the banana") |
| `image_path` | `str` | **Required** | Path to scene image file (JPG, PNG supported) |
| `output_format` | `str` | `"json"` | Output format (legacy parameter, always returns dict) |
| `force_reanalyze` | `bool` | `False` | Force re-analysis even if image is cached |
| `quiet` | `bool` | `False` | Suppress terminal progress output |
| `capture_output` | `bool` | `False` | Capture terminal output as string in result |

### Output Modes

| `quiet` | `capture_output` | Behavior |
|---------|------------------|----------|
| `False` | `False` | Show output on terminal only |
| `False` | `True` | Show output on terminal **AND** capture as string |
| `True` | `False` | No terminal output |
| `True` | `True` | No terminal output, capture as string |

## 📤 Return Format

```python
{
    "task_description": "first pick up the paper, then pick up the banana",
    "image_path": "/path/to/image.jpg",
    "detected_objects": ["Paper", "Banana", "Pen", "Controller"],
    "spatial_relationships": "The banana is on top of the paper. The pen is to the right...",
    "action_plan": "plan: (Paper, (0.0, 0.0), MovetoObject) -> (Paper, (0.0, 0.0), PickupObject) -> ...",
    "success": true,
    "terminal_output": "Full captured pipeline logs..."  # Only if capture_output=True
}
```

### Return Fields

| Field | Type | Description |
|-------|------|-------------|
| `task_description` | `str` | Input task description |
| `image_path` | `str` | Input image file path |
| `detected_objects` | `list[str]` | Objects found in scene |
| `spatial_relationships` | `str` | Spatial analysis summary |
| `action_plan` | `str` | Executable robot action sequence |
| `success` | `bool` | Whether planning succeeded |
| `terminal_output` | `str` | Captured pipeline logs (if `capture_output=True`) |

## 🤖 Robot Actions

The system generates plans using these standardized robot actions:

| Action | Format | Description |
|--------|--------|-------------|
| `MovetoObject` | `(Object, (0.0, 0.0), MovetoObject)` | Navigate to specified object |
| `PickupObject` | `(Object, (0.0, 0.0), PickupObject)` | Grasp and lift object |
| `PutDownObject` | `(Object, (0.0, 0.0), PutDownObject)` | Release object at current position |
| `OpenGripper` | `(Object, (0.0, 0.0), OpenGripper)` | Open robot gripper |
| `CloseGripper` | `(Object, (0.0, 0.0), CloseGripper)` | Close robot gripper |
| `Home` | `(Object, (0.0, 0.0), Home)` | Return to home position |

### Example Action Plan
```
plan: (Banana, (0.0, 0.0), MovetoObject) -> (Banana, (0.0, 0.0), PickupObject) -> (Paper, (0.0, 0.0), MovetoObject) -> (Paper, (0.0, 0.0), PickupObject)
```

## 🧠 Intelligent Planning Logic

### Spatial Constraint Handling
- **Blocking Detection**: Identifies objects on top of target objects
- **Dependency Resolution**: Automatically moves blocking objects first
- **Accessibility Analysis**: Ensures all target objects can be reached

### Multi-Step Task Processing
```python
# Single-step task
"pick up the banana" → Direct pickup plan

# Multi-step task  
"first pick up the paper, then pick up the banana" → Sequential plan with proper ordering
```

### Spatial Relationship Understanding
- **"X is on top of Y"** → Must move X before accessing Y
- **"X is under Y"** → Y blocks access to X
- **"X is to the left/right of Y"** → Spatial positioning for navigation

## 🔧 Configuration & Setup

### Model Configuration
```python
# Vision Model (Local)
QWEN2_5_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Online Reasoning Model
ONLINE_MODEL = "deepseek-chat"
MONICA_API_URL = "https://openapi.monica.im/v1/chat/completions"
```

### Performance Optimization Settings
- **GPU Memory**: Uses `torch.bfloat16` for efficiency
- **Caching**: Vision analysis results cached by image path
- **Generation**: KV caching enabled for faster inference
- **Memory Management**: Automatic cleanup after processing

## 🛠️ Advanced Integration Examples

### 1. Simple Robot Controller Integration
```python
from online_mllm_planner import robot_action_planner

class RobotController:
    def execute_task(self, task, image_path):
        result = robot_action_planner(task, image_path, quiet=True)
        
        if result["success"]:
            action_plan = result["action_plan"]
            return self.execute_plan(action_plan)
        else:
            raise Exception("Planning failed")
    
    def execute_plan(self, plan_string):
        # Parse and execute individual actions
        # Implementation depends on your robot interface
        pass
```

### 2. Monitoring and Logging System
```python
import datetime
from online_mllm_planner import robot_action_planner

def monitored_planning(task, image_path, log_dir="logs"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result = robot_action_planner(
        task_description=task,
        image_path=image_path,
        capture_output=True,  # Capture full logs
        quiet=False           # Show progress
    )
    
    # Save detailed logs
    log_file = f"{log_dir}/planning_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Success: {result['success']}\n")
        f.write(f"Objects: {result['detected_objects']}\n")
        f.write(f"Plan: {result['action_plan']}\n\n")
        f.write("=== PIPELINE LOGS ===\n")
        f.write(result.get('terminal_output', 'No logs captured'))
    
    return result, log_file
```

### 3. Batch Processing with Progress Tracking
```python
from online_mllm_planner import robot_action_planner
import os

def batch_process_scenes(scene_dir, tasks):
    results = []
    
    for i, (task, image_file) in enumerate(tasks):
        print(f"Processing {i+1}/{len(tasks)}: {task}")
        
        image_path = os.path.join(scene_dir, image_file)
        
        try:
            result = robot_action_planner(
                task_description=task,
                image_path=image_path,
                quiet=True  # Suppress individual logs for batch
            )
            results.append({
                "task": task,
                "image": image_file,
                "result": result
            })
        except Exception as e:
            results.append({
                "task": task,
                "image": image_file,
                "error": str(e)
            })
    
    return results

# Usage
tasks = [
    ("pick up the red ball", "scene1.jpg"),
    ("move the box to the left", "scene2.jpg"),
    ("first clear the table, then organize items", "scene3.jpg")
]

batch_results = batch_process_scenes("./scenes/", tasks)
```

### 4. Web API Service
```python
from flask import Flask, request, jsonify, send_file
from online_mllm_planner import robot_action_planner
import tempfile
import os

app = Flask(__name__)

@app.route('/plan', methods=['POST'])
def plan_action():
    try:
        data = request.json
        
        result = robot_action_planner(
            task_description=data['task'],
            image_path=data['image_path'],
            capture_output=data.get('include_logs', False),
            quiet=not data.get('verbose', False)
        )
        
        return jsonify(result)
        
    except FileNotFoundError:
        return jsonify({"error": "Image file not found"}), 404
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Planning failed: {e}"}), 500

@app.route('/plan_with_upload', methods=['POST'])
def plan_with_upload():
    try:
        # Handle file upload
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        task = request.form.get('task')
        
        if not task:
            return jsonify({"error": "No task description provided"}), 400
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            
            result = robot_action_planner(
                task_description=task,
                image_path=tmp_file.name,
                quiet=True
            )
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return jsonify(result)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📊 Performance Benchmarks

### Timing Breakdown
- **Model Loading**: ~30-60 seconds (first time only)
- **Vision Analysis**: ~5-15 seconds per image
- **Online API Calls**: ~2-8 seconds per step
- **Total Pipeline**: ~15-45 seconds per task
- **Cached Runs**: ~5-15 seconds (vision analysis skipped)

### Memory Usage
- **GPU VRAM**: ~6-8GB for Qwen2.5-VL-7B
- **System RAM**: ~4-8GB during processing
- **Model Caching**: Keeps vision model loaded between calls

### Accuracy Features
- **Multi-step Validation**: Cross-validation between pipeline steps
- **Spatial Consistency**: Logic checking for spatial relationships
- **Fallback Planning**: Robust action plan generation with multiple strategies
- **Error Recovery**: Graceful handling of API failures and edge cases

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. GPU Memory Errors
```python
# Error: CUDA out of memory
# Solution: Close other GPU processes or restart Python
import torch
torch.cuda.empty_cache()  # Clear GPU cache
```

#### 2. API Connection Issues
```python
# Error: Monica API connection failed
# Solutions:
# 1. Check API key in online_mllm_planner.py
# 2. Verify internet connection
# 3. Check Monica API status
```

#### 3. No Objects Detected
```python
# Error: Empty detected_objects list
# Solutions:
result = robot_action_planner(
    task="your task",
    image_path="your_image.jpg",
    force_reanalyze=True,  # Force fresh analysis
    quiet=False            # See detailed logs
)
```

#### 4. Model Loading Failures
```python
# Error: Failed to load vision model
# Solutions:
# 1. Ensure CUDA is available: torch.cuda.is_available()
# 2. Check GPU memory: nvidia-smi
# 3. Install missing dependencies: pip install qwen-vl-utils
```

### Debug Mode
Enable detailed logging to diagnose issues:

```python
result = robot_action_planner(
    task_description="pick up banana",
    image_path="scene.jpg",
    capture_output=True,  # Capture all logs
    quiet=False,          # Show progress
    force_reanalyze=True  # Fresh analysis
)

# Check detailed logs
print("=== PIPELINE LOGS ===")
print(result.get('terminal_output', 'No logs captured'))
```

## 🔐 Security & Best Practices

### API Key Management
```python
# Option 1: Environment variables
import os
MONICA_API_KEY = os.getenv('MONICA_API_KEY')

# Option 2: Config file
import json
with open('config.json') as f:
    config = json.load(f)
    MONICA_API_KEY = config['monica_api_key']
```

### Input Validation
```python
import os
from pathlib import Path

def safe_planning(task, image_path):
    # Validate inputs
    if not task or not task.strip():
        raise ValueError("Task description cannot be empty")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    if Path(image_path).suffix.lower() not in valid_extensions:
        raise ValueError("Unsupported image format")
    
    return robot_action_planner(task, image_path)
```

## 📝 License & Attribution

- **Qwen2.5-VL**: Apache 2.0 License (Alibaba DAMO Academy)
- **Transformers**: Apache 2.0 License (Hugging Face)
- **Monica API**: Commercial service (Monica.im)
- **This Project**: MIT License

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Add** comprehensive tests for new functionality
4. **Update** documentation and README
5. **Submit** pull request with detailed description

### Development Setup
```bash
git clone your-fork-url
cd robot-action-planner
conda create -n dev_planner python=3.9
conda activate dev_planner
pip install -r requirements.txt
```
---

**🚀 Ready to start planning robot actions? Import the module and begin with a simple task!** 