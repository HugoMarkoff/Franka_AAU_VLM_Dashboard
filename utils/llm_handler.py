"""utils/llm_handler.py
A thin helper around the remote /chat and /chat‑stream endpoints that:
  • Streams the *thinking* tokens so the dashboard can show them live
    (Window 7 “LLM Reasoning”).
  • Collects the final answer so the normal chat pane (Window 4) gets one
    clean reply when generation ends.
  • Runs a crude NLP pass to decide whether the user’s message describes an
    **ACTION** (pick‑and‑place) or ordinary chit‑chat, and—if an action—tries
    to extract ``{object}`` and (optional) ``{placement}`` phrases that we can
    forward to RoboPoint.

Typical usage from *app.py*::

    from utils.llm_handler import LLMHandler

    llm = LLMHandler(
        stream_url="https://<ngrok>/chat-stream",   # SSE endpoint
        full_url  ="https://<ngrok>/chat"           # full‑text fallback
    )

    thinking, answer, intent = llm.process(text)
    # → thinking  = list[str] (token fragments)
    #   answer    = str       (full detokenised reply)
    #   intent    = {
    #        "type"     : "action"|"chat",
    #        "object"   : str|None,
    #        "placement": str|None
    #     }
"""
from __future__ import annotations

import asyncio
import httpx
import re
from collections import deque
from typing import List, Dict, Tuple, Optional, Generator


# -----------------------------------------------------------------------------
#  Intent / slot‑filling helpers
# -----------------------------------------------------------------------------
ACTION_VERBS = [
    "pick", "pickup", "grab", "take", "lift",
    "place", "put", "drop", "release",
]

PLACEMENT_PREPS = [
    "in", "into", "on", "onto", "at", "to", "inside", "within"
]

OBJ_REGEX = re.compile(r"pick(?:\s+up)?\s+the\s+([a-zA-Z0-9 _-]+?)\b", re.I)
PLACE_REGEX = re.compile(r"(?:" + "|".join(PLACEMENT_PREPS) + r")\s+the\s+([a-zA-Z0-9 _-]+?)\b", re.I)


def _extract_action_slots(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (object, placement) or (None, None) if not found."""
    obj = None
    placement = None

    # Object—look after the verb
    m = OBJ_REGEX.search(text)
    if m:
        obj = m.group(1).strip()

    # Placement—look after a preposition
    p = PLACE_REGEX.search(text)
    if p:
        placement = p.group(1).strip()

    return obj, placement


def _looks_like_action(text: str) -> bool:
    t = text.lower()
    return any(v in t for v in ACTION_VERBS)


# -----------------------------------------------------------------------------
#  LLM Handler class
# -----------------------------------------------------------------------------
class LLMHandler:
    """Convenience wrapper for the remote LLM API (stream + full modes)."""

    def __init__(self, stream_url: str, full_url: str, model: str | None = None):
        self.stream_url = stream_url.rstrip("/")
        self.full_url   = full_url.rstrip("/")
        self.model      = model  # default name or None

    # .................................
    #  public high‑level entry point
    # .................................
    def process(self, prompt: str) -> Tuple[List[str], str, Dict[str, Optional[str]]]:
        """Return (thinking_tokens, final_answer, intent_dict).

        *thinking_tokens* — list of chunks to append to Window 7 as they arrive.
        *final_answer*   — the whole model reply, de‑tokenised.
        *intent_dict*    — {type:"chat"|"action", object:?str, placement:?str}
        """
        # Run event‑loop to stream tokens (blocks in Flask thread)
        thinking, answer = asyncio.run(self._stream_and_collect(prompt))

        # Intent detection
        if _looks_like_action(prompt):
            obj, place = _extract_action_slots(prompt)
            intent = {"type": "action", "object": obj, "placement": place}
        else:
            intent = {"type": "chat", "object": None, "placement": None}

        return thinking, answer, intent

    # .................................
    #  internal streaming coroutine
    # .................................
    async def _stream_and_collect(self, prompt: str) -> Tuple[List[str], str]:
        """Open /chat‑stream and gather all tokens (with fallback to /chat)."""
        thinking: List[str] = []
        answer_parts: List[str] = []

        payload = {"prompt": prompt}
        if self.model:
            payload["model"] = self.model

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", self.stream_url, json=payload) as r:
                    async for raw in r.aiter_lines():
                        if not raw or not raw.startswith("data:"):
                            continue  # keep‑alive or malformed
                        tok = raw[5:].lstrip()
                        if tok == "[DONE]":
                            break
                        thinking.append(tok)
                        answer_parts.append(tok)
        except Exception:
            # Fallback: one‑shot /chat so we still answer
            try:
                resp = httpx.post(self.full_url, json=payload, timeout=60)
                resp.raise_for_status()
                answer = resp.json()["result"]
                return [answer], answer
            except Exception as e:
                err = f"<LLM error: {e}>"
                return [err], err

        answer = "".join(answer_parts)
        return thinking, answer
