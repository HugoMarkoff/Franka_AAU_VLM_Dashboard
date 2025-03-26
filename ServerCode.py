# server.py
import io
import base64
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn
from pyngrok import ngrok

# RoboPoint imports
from robopoint.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from robopoint.conversation import conv_templates
from robopoint.model.builder import load_pretrained_model
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

###############################################################################
# CONFIG
###############################################################################

# Path to your local folder with the RoboPoint model (with config.json, etc.)
MODEL_PATH = "./RoboPointModels/robopoint-v1-vicuna-v1.5-13b"
# If it's a LoRA checkpoint, specify the base model folder. Otherwise keep None.
MODEL_BASE_PATH = None

CONV_MODE = "llava_v1"

DEFAULT_INSTRUCTIONS = (
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
    "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
    "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
)

TEMPERATURE = 0.2
TOP_P = 0.9
NUM_BEAMS = 1
MAX_NEW_TOKENS = 1024

TUNNEL_PORT = 8000

###############################################################################
# FASTAPI APP + CORS
###############################################################################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    instruction: str
    image: str  # base64

###############################################################################
# LOAD THE ROBOPOINT MODEL
###############################################################################
print("[INFO] Loading RoboPoint model from:", MODEL_PATH)

model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=MODEL_BASE_PATH,
    model_name=model_name
)
model.cuda().eval()
print(f"[INFO] Loaded model '{model_name}' successfully.")

vision_tower = getattr(model.model, "vision_tower", None)
if vision_tower is None or (isinstance(vision_tower, list) and all(v is None for v in vision_tower)):
    print("[WARN] vision_tower is None (not fully multimodal?).")
else:
    print("[INFO] vision_tower present.")

###############################################################################
# /predict
###############################################################################
@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    """
    - Decodes base64 image
    - Appends user instruction + DEFAULT_INSTRUCTIONS
    - If <IMAGE_TOKEN> not in prompt, insert it
    - Builds conversation from conv_templates
    - Calls model.generate(images=..., image_sizes=...)
    - Returns text
    """
    # 1) Base64 -> PIL
    try:
        image_bytes = base64.b64decode(request.image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # 2) Combine user + default instructions
    user_input = request.instruction.strip() + "\n" + DEFAULT_INSTRUCTIONS

    # 3) Insert <IMAGE_TOKEN> if missing
    if DEFAULT_IMAGE_TOKEN not in user_input:
        if getattr(model.config, 'mm_use_im_start_end', False):
            user_input = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN +
                "\n" + user_input
            )
        else:
            user_input = DEFAULT_IMAGE_TOKEN + "\n" + user_input

    # 4) Build conversation
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # 5) Tokenize
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).cuda()

    # 6) Image -> tensor
    image_tensor = process_images([pil_image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    # 7) Generate
    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[pil_image.size],
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_beams=NUM_BEAMS,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

    # 8) Decode
    result_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return {"result": result_text}

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    public_url = ngrok.connect(TUNNEL_PORT)
    print("[INFO] ngrok tunnel available at:", public_url.public_url)
    uvicorn.run(app, host="0.0.0.0", port=TUNNEL_PORT)
