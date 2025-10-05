from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI  # >=1.40
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..afm_schema import AFMStep
from .tools import run_tool

# ---------- Valid tools & docs ----------
VALID = {
    "add", "subtract", "multiply", "divide", "square", "power", "sqrt", "round", "stop"
}

TOOLS_DOC = """\
add(a: number, b: number) -> number
subtract(a: number, b: number) -> number
multiply(a: number, b: number) -> number
divide(a: number, b: number) -> number
square(x: number) -> number
power(x: number, y: number) -> number
sqrt(x: number) -> number
stop() -> null
"""

# ---------- Final tag handling ----------
FINAL_TAG = "final"
FINAL_RE = re.compile(r"<\s*final\s*>(.*?)</\s*final\s*>", re.I | re.S)

def extract_final_answer(text: str) -> str:
    """
    Tolerant extractor: handles case/whitespace and accidental code fences.
    Raises ValueError if the <final>...</final> tag is missing.
    """
    if not text:
        raise ValueError(f"Final answer missing or not wrapped in <{FINAL_TAG}> tags.")
    t = text.strip()
    # Strip accidental code fences if any
    if t.startswith("```"):
        # remove leading and trailing backticks/newlines/spaces
        t = t.strip("` \n\r\t")
    m = FINAL_RE.search(t)
    if not m:
        raise ValueError(f"Final answer missing or not wrapped in <{FINAL_TAG}> tags.")
    return m.group(1).strip()

# ---------- OpenAI client (lazy) ----------
_client_singleton: Optional[OpenAI] = None
def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client_singleton = OpenAI(api_key=key)
    return _client_singleton

# ---------- Prompt loading via Jinja ----------
def _render_system_prompt() -> str:
    prompts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
    env = Environment(
        loader=FileSystemLoader(prompts_dir),
        autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("math_agent_system.j2")
    return tmpl.render(
        tool_list=TOOLS_DOC,
        tool_names=sorted(list(VALID)),
    )

# ---------- JSON-schema for planning ----------
def _planner_schema() -> Dict[str, Any]:
    """
    Enforce that each step is:
      {
        "thought": str,
        "action": {"name": one_of(VALID), "args": object}
      }
    """
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
                        "args": {"type": "object"}  # validated by tool when run
                    }
                }
            }
        }
    }

def _plan_step(history: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    r = _client().chat.completions.create(
        model=model,
        temperature=0,
        messages=history,
        response_format={"type": "json_schema", "json_schema": _planner_schema()},
    )
    content = (r.choices[0].message.content or "").strip()
    return json.loads(content)

# ---------- Arg resolution (numbers & carry variable 'r') ----------
def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        xs = x.strip()
        # Allow leading +/-, decimals
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", xs):
            return float(xs)
    raise ValueError(f"Non-numeric literal: {x!r}")

def resolve_arg(val: Any, env: Dict[str, float]) -> float:
    """
    Allowed inputs:
      - numeric literals (int/float/str-of-number)
      - symbol 'r' => last result (carry)
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        v = val.strip()
        if v.lower() == "r":
            if "r" not in env:
                raise ValueError("Carry variable 'r' not set yet")
            return float(env["r"])
        # numeric string?
        return _to_float(v)
    # Reject any complex structure (lists, dicts, etc.)
    return _to_float(val)

def normalize_args(name: str, raw_args: Dict[str, Any], env: Dict[str, float]) -> Dict[str, float]:
    n = name.lower()
    if n in {"add", "subtract", "multiply", "divide"}:
        a = resolve_arg(raw_args.get("a"), env)
        b = resolve_arg(raw_args.get("b"), env)
        return {"a": a, "b": b}
    elif n == "square":
        x = resolve_arg(raw_args.get("x"), env)
        return {"x": x}
    elif n == "power":
        x = resolve_arg(raw_args.get("x"), env)
        y = resolve_arg(raw_args.get("y"), env)
        return {"x": x, "y": y}
    elif n == "sqrt":
        x = resolve_arg(raw_args.get("x"), env)
        return {"x": x}
    elif n == "round":                                           # ← NEW
        x = resolve_arg(raw_args.get("x"), env)                  # ← NEW
        nd = raw_args.get("ndigits", 0)                          # ← NEW
        nd = int(resolve_arg(nd, env) if isinstance(nd, str) else nd)
        return {"x": x, "ndigits": nd}                           # ← NEW
    elif n == "stop":
        return {}
    else:
        raise ValueError(f"Unsupported action name: {name}")

# ---------- Finalization with one retry ----------
def _finalize_answer(history: List[Dict[str, str]], model: str, final_tag: str = FINAL_TAG) -> str:
    """
    Ask for the final answer; if tag missing, retry once with a stricter instruction.
    Deterministic: temperature=0, fixed wording.
    """
    client = _client()
    # Attempt 1
    prompt = (
        f"Return ONLY the final numeric answer wrapped exactly as "
        f"<{final_tag}>NUMBER</{final_tag}>. No prose or extra text."
    )
    r = client.chat.completions.create(
        model=model, temperature=0, messages=history + [{"role": "user", "content": prompt}]
    )
    txt = (r.choices[0].message.content or "").strip()
    try:
        return extract_final_answer(txt)
    except ValueError:
        # Attempt 2 (one retry)
        retry = (
            f"Your previous reply did not follow the required format.\n"
            f"Return ONLY the final numeric answer wrapped exactly as:\n"
            f"<{final_tag}>NUMBER</{final_tag}>\n"
            f"No prose, no markdown, no extra text."
        )
        r2 = client.chat.completions.create(
            model=model, temperature=0, messages=history + [{"role": "user", "content": retry}]
        )
        txt2 = (r2.choices[0].message.content or "").strip()
        return extract_final_answer(txt2)

# ---------- Solve loop ----------
def solve(question: str, model: str = "gpt-4o-mini", max_steps: int = 8) -> Tuple[List[AFMStep], str]:
    """
    Runs the deterministic planner-executor loop and returns:
      - steps: list[AFMStep]
      - final: string with <final>NUMBER</final>
    """
    system_prompt = _render_system_prompt()
    history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]
    steps: List[AFMStep] = []
    env: Dict[str, float] = {}  # variable environment; 'r' holds last numeric result

    for _ in range(max_steps):
        plan = _plan_step(history, model=model)
        thought = plan["thought"]
        action = plan["action"]
        name: str = str(action["name"]).strip().lower()
        if name not in VALID:
            raise ValueError(f"Unsupported action: {name}")

        if name == "stop":
            steps.append(
                AFMStep(function="stop", thought=thought, tool_call={"name": "stop", "args": {}}, tool_result=None)
            )
            break

        # Normalize args & run tool deterministically
        raw_args = action.get("args", {})
        args = normalize_args(name, raw_args, env)
        call, result = run_tool(name, **args)

        # AFM step log
        steps.append(
            AFMStep(function=name, thought=thought, tool_call=call, tool_result=result)
        )

        # Update history with the plan JSON and observation (truncate observation for safety)
        history.append({"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)})
        obs_text = json.dumps(result, ensure_ascii=False)
        history.append({"role": "system", "content": f"Observation: {obs_text[:2000]}"} )

        # Update carry
        if isinstance(result, (int, float)):
            env["r"] = float(result)

    # Finalization (answer only, strictly tagged)
    final_answer = _finalize_answer(history, model=model, final_tag=FINAL_TAG)
    return steps, final_answer
