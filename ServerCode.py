###############################################################################
# main.py – conversational action extractor for RoboPoint
###############################################################################
import io
import base64
import os
import threading
import torch
import re
import json
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn
from pyngrok import ngrok

# ------------------ Transformers imports ------------------
try:
    # transformers >= 4.37
    from transformers.streamers import TextIteratorStreamer
except ImportError:
    from transformers import TextIteratorStreamer                     # type: ignore

from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Default LLM used for the /chat* endpoints (small & fast)
DEFAULT_LLM_NAME  = "Qwen/Qwen3-8B"
LLM_KWARGS        = {
    "torch_dtype": "auto",
    "device_map": "auto",
}

###############################################################################
# HIGH-LEVEL SYSTEM PROMPT
###############################################################################
LLM_SYSTEM_PROMPT = r"""
You are RoboPoint-Chat — an advanced AI assistant wired to a robot arm.

───────────────────────────  YOUR JOB  ───────────────────────────
1. Friendly conversation is allowed, but your *primary* task is to
   convert user instructions into ROBOT ACTIONS.

2. Every robot instruction is built from three *parameters*  
      • ACTION      → **Pick** or **Place**
                     synonyms (multi-lingual):
                       Pick  ≈ pick | grab | grasp | fetch | take | get | lift
                               recoger | prendre | nehmen | agarrar …
                       Place ≈ place | put | drop | set | position | insert
                               poner | mettre | colocar | legen …
      • OBJECT      → physical item(s) (e.g. “red cube”, “taza”, “Schlüssel”)
      • LOCATION    → where the object is or should be placed  
                     (optional; may be missing)

3. While chatting, keep a short-term memory of any ACTION, OBJECT or
   LOCATION already given in this session.  
   If something is missing, ask a **concise clarifying question** and
   store what you already know.

4. **Translate every parameter to English before you output it.**  
   The user may speak any language; the final `[ACTION]` block must
   contain *only English words* for ACTION, OBJECT and LOCATION.

5. Only when you hold at least ACTION + OBJECT for each step, reply
   using the exact FINAL BLOCK below.  Otherwise, keep chatting.

────────────────────────  FINAL BLOCK FORMAT  ────────────────────────
When (and only when) you are ready, reply **only** with:

[ACTION]
RoboPoint Request: <req-1>; <req-2>; …; <req-N>
Action Request:    <act-1>; <act-2>; …; <act-N>

where   <req-k> = <object-k>                 (if no location)  or
                  <object-k> at <location-k> (if a location exists)

Both lists **must align in order and length**.

──────────────────────────── EXAMPLES ────────────────────────────────
**User speaks Spanish**

User : « Recoge el cubo rojo en la esquina superior derecha
        y ponlo dentro de la taza en medio ».
You  :
[ACTION]
RoboPoint Request: red cube at top-right corner; cup at middle
Action Request:    Pick; Place

**User mixes French & Danish**

User : « Prends les clés dans le plateau ; læg dem i koppen ».
You  :
[ACTION]
RoboPoint Request: keys at tray; cup
Action Request:    Pick; Place

Never guess.  Ask if unsure.  Never output anything except the block
above when you use the [ACTION] format.
"""


###############################################################################
# FASTAPI APP + CORS
###############################################################################
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins      = ["*"],
    allow_credentials  = True,
    allow_methods      = ["*"],
    allow_headers      = ["*"],
)

###############################################################################
# REQUEST SCHEMAS
###############################################################################
class PredictionRequest(BaseModel):
    instruction: str
    image: str

class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    session_id: Optional[str] = "default"            # memory key

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
# LLM MODEL CACHE
###############################################################################
_llm_cache: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}
def get_llm(model_name: str = DEFAULT_LLM_NAME) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    if model_name in _llm_cache:
        return _llm_cache[model_name]
    print(f"[INFO] Loading LLM '{model_name}' …")
    try:
        tok  = AutoTokenizer.from_pretrained(model_name)
        mdl  = AutoModelForCausalLM.from_pretrained(model_name, **LLM_KWARGS).eval()
        _llm_cache[model_name] = (tok, mdl)
        return tok, mdl
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot load model '{model_name}': {e}")

###############################################################################
# CONVERSATION-MEMORY (RAM only, keyed by session_id)
###############################################################################
_session_history : Dict[str, List[Dict[str,str]]] = {}   # list of chat msgs

def get_history(session_id:str) -> List[Dict[str,str]]:
    return _session_history.setdefault(session_id, [])

# ────────────────────────────────────────────────────────────────────────────
#  Replace the old regex + function with this complete version
# ────────────────────────────────────────────────────────────────────────────
ACTION_BLOCK_RE = re.compile(
    r"\[ACTION\]\s*"                       # literal tag
    r"RoboPoint\s*Request:\s*(?P<reqs>.*?)\s*"   # non-greedy up to “Action…”
    r"Action\s*Request:\s*(?P<acts>.*)",          # rest of the text
    flags=re.IGNORECASE | re.DOTALL,
)

def parse_action_block(text: str) -> Optional[Dict[str, List[str]]]:
    """
    Extract the 2 aligned lists from an [ACTION] block, whether or not there
    is a newline between them.  Returns:

        { "requests": ["obj at loc", …],
          "actions" : ["Pick", …] }
    or None if parsing fails.
    """
    m = ACTION_BLOCK_RE.search(text)
    if not m:
        return None

    # split on ';' or '|' that the model often uses (line-breaks are already
    # eaten by DOTALL, so they appear as raw '\n' inside the strings)
    reqs = [r.strip() for r in re.split(r"[;|]", m["reqs"]) if r.strip()]
    acts = [a.strip() for a in re.split(r"[;|]", m["acts"]) if a.strip()]

    if len(reqs) != len(acts) or not reqs:
        return None
    return {"requests": reqs, "actions": acts}


###############################################################################
# PRE-LOAD THE DEFAULT CHAT LLM ON APP STARTUP
###############################################################################
@app.on_event("startup")
def preload_default_llm() -> None:
    """
    Load the default LLM into memory when the server starts so the
    first /chat or /chat-stream call doesn’t pay the cold-start cost.
    """
    try:
        print(f"[INFO] Pre-loading default LLM '{DEFAULT_LLM_NAME}' …")
        get_llm(DEFAULT_LLM_NAME)            # caches the model
        print("[INFO] Default LLM ready.")
    except HTTPException as e:
        # If the model cannot be loaded, we still start the API so you
        # can diagnose the problem via /chat later.
        print(f"[WARN] Could not preload LLM: {e.detail}")

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

# ────────────────────────────────────────────────────────────────────────────
#  Full new body of /chat (synchronous) – unchanged except for the memory wipe
# ────────────────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    tok, mdl = get_llm(request.model or DEFAULT_LLM_NAME)
    history  = get_history(request.session_id)

    history.append({"role": "user", "content": request.prompt})
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}] + history

    prompt_text = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)

    inputs = tok([prompt_text], return_tensors="pt").to(mdl.device)
    with torch.inference_mode():
        gen_ids = mdl.generate(**inputs, max_new_tokens=1024,
                               temperature=0.7, top_p=0.95, do_sample=True)

    resp = tok.decode(gen_ids[0][len(inputs.input_ids[0]):],
                      skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": resp})

    block = parse_action_block(resp)
    if block:
        _session_history[request.session_id] = []       # ← wipe memory
        return {"result": resp, "parsed": block}

    return {"result": resp}


# ────────────────────────────────────────────────────────────────────────────
#  Full new body of /chat-stream – memory wipe added inside event_stream()
# ────────────────────────────────────────────────────────────────────────────
@app.post("/chat-stream")
def chat_stream_endpoint(request: ChatRequest):
    tok, mdl = get_llm(request.model or DEFAULT_LLM_NAME)
    history  = get_history(request.session_id)

    history.append({"role": "user", "content": request.prompt})
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}] + history
    prompt_text = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)

    model_inputs = tok([prompt_text], return_tensors="pt").to(mdl.device)
    streamer     = TextIteratorStreamer(tok, skip_prompt=True,
                                        skip_special_tokens=True)

    def _worker():
        with torch.inference_mode():
            mdl.generate(**model_inputs, streamer=streamer,
                         max_new_tokens=1024, temperature=0.7,
                         top_p=0.95, do_sample=True)

    threading.Thread(target=_worker, daemon=True).start()

    collected   = ""
    action_sent = False

    def event_stream():
        nonlocal collected, action_sent
        for token in streamer:
            collected += token
            yield f"data: {token}\n\n"

            if not action_sent and "[ACTION]" in collected:
                blk = parse_action_block(collected)
                if blk:
                    yield f"data: \n{json.dumps(blk)}\n\n"
                    _session_history[request.session_id] = []   # ← wipe memory
                    action_sent = True

        yield "data: [DONE]\n\n"

    def finish_chat():
        history.append({"role": "assistant", "content": collected.strip()})

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream",
                             background=finish_chat)


###############################################################################
# MAIN (dev-mode only)
###############################################################################
if __name__ == "__main__":
    public_url = ngrok.connect(TUNNEL_PORT)
    print("[INFO] Public URL:", public_url.public_url)
    uvicorn.run(app, host="0.0.0.0", port=TUNNEL_PORT)
