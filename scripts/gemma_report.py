"""Ask Gemma 4 to write the call report from facts already counted in Qdrant.

The model does not see embeddings. It only sees the people, calls, and role
marks produced by face_report.
"""

from __future__ import annotations

import json
import re
import threading

MODEL_ID = "google/gemma-4-E2B-it"

_lock = threading.Lock()
_model = None
_processor = None


def facts_for_model(report: dict) -> dict:
    """Drop vectors and anything the model could treat as a measurement."""
    keep = (
        "gap_s",
        "threshold",
        "recordings",
        "calls",
        "people",
        "common_faces",
        "visitor_other_inmates",
        "inmate_visitor_calls",
        "inmate_calls",
        "visitor_calls",
        "roles_set",
    )
    facts = {key: report[key] for key in keep if key in report}
    if "people" in facts:
        facts["people"] = [
            {key: value for key, value in person.items() if key not in ("sample", "side_x", "role_source")}
            for person in facts["people"]
        ]
    return facts


def prompt_for(report: dict) -> str:
    facts = facts_for_model(report)
    return (
        "Write a prison-call face report. Use only the JSON. "
        "Do not add a person, a call, or a role that is not in the named list.\n"
        "A recording is one video. A call is one stretch of that video.\n"
        "Inmate and visitor exist only when role is inmate or visitor. "
        "An empty role is unmarked, not an inmate and not a visitor.\n\n"
        "Reply with exactly five lines of prose. No JSON. No code fence.\n"
        "1. Faces in more than one recording. Use only common_faces. "
        "If common_faces is [], the line is: Faces in more than one recording: None.\n"
        "2. Visitors who also share a call with more than one inmate. "
        "Use only visitor_other_inmates. If that list is [] or every "
        "also_with_other_inmates is false, say none of them do, and name "
        "the inmate each visitor did share a call with.\n"
        "3. How many calls each inmate shared with each visitor. "
        "Use only inmate_visitor_calls. If that list is [], write None.\n"
        "4. How many calls each inmate was in. Use only inmate_calls. "
        "If inmate_calls is [], write None.\n"
        "5. How many calls each visitor was in. Use only visitor_calls. "
        "If visitor_calls is [], write None.\n\n"
        f"JSON:\n{json.dumps(facts, indent=2)}"
    )


def _json_object(reply: str) -> dict | None:
    text = reply.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _label(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("label") or item.get("inmate") or item.get("id") or item)
    return str(item)


def _none(value) -> bool:
    if value is None or value == [] or value == {}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"none", "none."}:
        return True
    return False


def _first(data: dict, names: tuple[str, ...]):
    for name in names:
        if name in data:
            return data[name]
    return None


def present(reply: str) -> str:
    """Turn a JSON dump from the small model into the five report lines."""
    data = _json_object(reply)
    if data is None:
        return reply.strip()
    common = _first(data, ("common_faces", "faces_in_more_than_one_recording", "common_faces_report"))
    others = _first(data, ("visitor_other_inmates", "visitors_who_share_a_call_with_more_than_one_inmate", "visitor_other_inmates_report"))
    pairs = _first(data, ("inmate_visitor_calls", "inmate_visitor_calls_report"))
    inmates = _first(data, ("inmate_calls", "inmate_calls_report"))
    visitors = _first(data, ("visitor_calls", "visitor_calls_report"))
    if all(item is None for item in (common, others, pairs, inmates, visitors)):
        return reply.strip()

    lines = [_faces_line(common), _visitors_line(others), _pairs_line(pairs), _counts_line("Inmate calls", inmates), _counts_line("Visitor calls", visitors)]
    return "\n".join(lines)


def _faces_line(value) -> str:
    if _none(value):
        return "Faces in more than one recording: None."
    if isinstance(value, list):
        names = ", ".join(_label(item) for item in value)
        return f"Faces in more than one recording: {names}."
    return f"Faces in more than one recording: {value}."


def _visitors_line(value) -> str:
    if _none(value):
        return "Visitors who also have calls with other inmates: None."
    if not isinstance(value, list):
        return f"Visitors who also have calls with other inmates: {value}."
    repeated = []
    only = []
    for item in value:
        if not isinstance(item, dict):
            only.append(_label(item))
            continue
        mates = item.get("inmates") or []
        mate_names = ", ".join(_label(mate) for mate in mates) or "nobody"
        if item.get("also_with_other_inmates"):
            repeated.append(f"{_label(item)} does, with {mate_names}")
        else:
            only.append(f"{_label(item)} does not; that face shares a call with {mate_names}")
    parts = repeated + only
    return "Visitors who also have calls with other inmates: " + "; ".join(parts) + "."


def _pairs_line(value) -> str:
    if _none(value):
        return "Calls shared by an inmate and a visitor: None."
    if not isinstance(value, list):
        return f"Calls shared by an inmate and a visitor: {value}."
    parts = []
    for item in value:
        if not isinstance(item, dict):
            parts.append(_label(item))
            continue
        count = item.get("calls", 0)
        word = "call" if count == 1 else "calls"
        parts.append(f"{item.get('inmate', _label(item))} and {item.get('visitor', 'a visitor')} shared {count} {word}")
    return "Calls shared by an inmate and a visitor: " + "; ".join(parts) + "."


def _counts_line(title: str, value) -> str:
    if _none(value):
        return f"{title}: None."
    if not isinstance(value, list):
        return f"{title}: {value}."
    parts = []
    for item in value:
        if isinstance(item, dict):
            count = item.get("calls", 0)
            word = "call" if count == 1 else "calls"
            parts.append(f"{_label(item)} was in {count} {word}")
        else:
            parts.append(_label(item))
    return f"{title}: " + "; ".join(parts) + "."


def inmate_side(reply: str) -> str:
    """The side Gemma named as the inmate: left, right, or unknown."""
    text = reply.lower()
    named = re.findall(r"\b(left|right|unknown)\b", text)
    if not named:
        return "unknown"
    if "left" in named and "right" in named:
        return "unknown"
    return named[0]


def judge_inmate_side(image, max_new_tokens: int = 40) -> dict:
    """Ask Gemma which side of this frame is the inmate."""
    import torch

    prompt = (
        "This is one frame from a prison visit. "
        "LEFT is the person on the left side of the image. "
        "RIGHT is the person on the right side of the image. "
        "The inmate is the person being held. The visitor came to see the inmate. "
        "Which side is the inmate? Reply with one word: left, right, or unknown."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    with _lock:
        _load()
        inputs = _processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(_model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = _model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1]
        reply = _processor.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    return {"reply": reply, "inmate_side": inmate_side(reply)}


def _load():
    global _model, _processor
    if _model is not None:
        return
    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype)
    _model.to(device)
    _model.eval()


def write_report(report: dict, max_new_tokens: int = 700) -> str:
    """Return Gemma's report for these Qdrant facts."""
    import torch

    text = prompt_for(report)
    with _lock:
        _load()
        messages = [{"role": "user", "content": text}]
        inputs = _processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(_model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = _model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1]
        reply = _processor.decode(output[0][prompt_len:], skip_special_tokens=True)
    return present(reply)
