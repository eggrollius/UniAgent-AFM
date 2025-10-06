from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..afm_schema import AFMStep
from .tools import run_tool

# -------------------- Valid actions & final tag --------------------
VALID = {"retrieve", "quote", "summarize", "compare", "stop"}

FINAL_TAG = "final"
FINAL_RE = re.compile(r"<\s*final\s*>(.*?)</\s*final\s*>", re.I | re.S)


def extract_final(text: str) -> str:
    if not text:
        raise ValueError(f"Final answer missing or not wrapped in <{FINAL_TAG}> tags.")
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("` \n\r\t")
    m = FINAL_RE.search(t)
    if not m:
        raise ValueError(f"Final answer missing or not wrapped in <{FINAL_TAG}> tags.")
    return m.group(1).strip()


# -------------------- Prompt loading via Jinja --------------------
def _render_system_prompt() -> str:
    prompts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
    if not os.path.isdir(prompts_dir):
        raise FileNotFoundError(
            f"Legal prompt directory not found: {prompts_dir} "
            f"(expected legal_agent/prompts/ with legal_agent_system.j2)"
        )
    env = Environment(
        loader=FileSystemLoader(prompts_dir),
        autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("legal_agent_system.j2")
    return tmpl.render()


# -------------------- OpenAI client (lazy, cached) --------------------
_CLIENT_INSTANCE: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _CLIENT_INSTANCE = OpenAI(api_key=key)
    return _CLIENT_INSTANCE


# -------------------- JSON schema for planning --------------------
def _planner_schema() -> Dict[str, Any]:
    return {
        "name": "PlannerStep",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["thought", "action"],
            "properties": {
                "thought": {"type": "string"},
                "action": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "args"],
                    "properties": {
                        "name": {"type": "string", "enum": sorted(list(VALID))},
                        "args": {"type": "object"},
                    },
                },
            },
        },
    }


def plan_step(history: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    r = _get_client().chat.completions.create(
        model=model,
        temperature=0,
        messages=history,
        response_format={"type": "json_schema", "json_schema": _planner_schema()},
    )
    txt = (r.choices[0].message.content or "").strip()
    try:
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            raise ValueError("Planner output is not a JSON object")
        act = obj.get("action")
        if not isinstance(act, dict):
            raise ValueError("Missing or non-object 'action'")
        # Normalize args: coerce None/missing to {}
        if "args" not in act or act["args"] is None:
            act["args"] = {}
        obj["action"] = act
        return obj
    except Exception as e:
        # Surface a clear error with the raw content to aid debugging
        raise ValueError(f"Failed to parse planner step JSON: {e}; content={txt!r}") from e


# -------------------- Finalization --------------------
def _finalize_answer(history: List[Dict[str, str]], model: str, final_tag: str = FINAL_TAG) -> str:
    client = _get_client()
    # First attempt
    prompt = (
        f"Return ONLY the final answer wrapped exactly as <{final_tag}>ANSWER||doc:ID1,doc:ID2</{final_tag}> "
        f"or <{final_tag}>REFUSE</{final_tag}>. No extra text."
    )
    r = client.chat.completions.create(
        model=model, temperature=0, messages=history + [{"role": "user", "content": prompt}]
    )
    txt = (r.choices[0].message.content or "").strip()
    try:
        return extract_final(txt)
    except ValueError:
        # Retry once, with explicit correction
        retry = (
            f"Your previous reply did not follow the required format.\n"
            f"Return ONLY one of:\n"
            f"  <{final_tag}>ANSWER||doc:ID1,doc:ID2</{final_tag}>\n"
            f"  <{final_tag}>REFUSE</{final_tag}>\n"
            f"No prose or markdown."
        )
        r2 = client.chat.completions.create(
            model=model, temperature=0, messages=history + [{"role": "user", "content": retry}]
        )
        txt2 = (r2.choices[0].message.content or "").strip()
        return extract_final(txt2)


# -------------------- Solve loop --------------------
def solve(question: str, model: str = "gpt-4o-mini", max_steps: int = 6) -> Tuple[List[AFMStep], str]:
    sys_prompt = _render_system_prompt()
    history: List[Dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]
    steps: List[AFMStep] = []

    for _ in range(max_steps):
        plan = plan_step(history, model=model)

        thought = plan.get("thought", "")
        action = plan.get("action", {})
        name = str(action.get("name", "")).strip().lower()
        if name not in VALID:
            raise ValueError(f"Unsupported action: {name!r}")

        if name == "stop":
            steps.append(
                AFMStep(function="stop", thought=thought, tool_call={"name": "stop", "args": {}}, tool_result=None)
            )
            break

        raw_args = action.get("args", {}) or {}
        call, result = run_tool(name, **raw_args)

        steps.append(AFMStep(function=name, thought=thought, tool_call=call, tool_result=result))

        # Log the plan as JSON and the observation deterministically
        history.append({"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)})
        obs_text = json.dumps(result, ensure_ascii=False)
        history.append({"role": "system", "content": f"Observation: {obs_text[:2000]}"} )

    final = _finalize_answer(history, model=model)
    return steps, final
