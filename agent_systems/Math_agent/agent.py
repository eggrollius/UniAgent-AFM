# agent_systems/Math_agent/agent.py

from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Optional, Tuple

from .tools import run_tool
from .afm_schema import AFMStep

MOCK_MODE = os.getenv("MOCK_AGENT", "0") == "1"

VALID: set[str] = {"add", "multiply", "square", "stop"}
ALIASES: Dict[str, str] = {
    "addition":"add","plus":"add","sum":"add",
    "mul":"multiply","times":"multiply","product":"multiply",
    "sqr":"square","pow2":"square",
    "stop":"stop","finalize":"stop","finish":"stop","done":"stop",
    "none":"stop","no-op":"stop","noop":"stop"
}

SYS_PROMPT = (
    "You are a careful math planner.\n"
    "At each step, choose ONE action from this exact set:\n"
    "  add(a,b) | multiply(a,b) | square(x) | stop()\n"
    "Arguments MUST be numeric literals or previously defined variables from the question "
    "(e.g., if the question says x=4 then you may use x).\n"
    "Return STRICT JSON ONLY (no prose):\n"
    '{"thought":"<short>","action":{"name":"add|multiply|square|stop","args":{...}}}\n'
    "If the task is complete, use name='stop' (args can be {})."
)

def extract_json_block(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def normalize_action_name(name: str) -> str:
    n = (name or "").strip().lower()
    return ALIASES.get(n, n)

# --- NEW: parse variable assignments like "x=4", "y = -3.5"
ASSIGN_RE = re.compile(r"\b([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\b")
def extract_env_vars(question: str) -> Dict[str, float]:
    env: Dict[str, float] = {}
    for var, val in ASSIGN_RE.findall(question):
        try:
            env[var] = float(val)
        except Exception:
            pass
    return env

def resolve_arg(val: Any, env: Dict[str, float]) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        # numeric string?
        try:
            return float(s)
        except Exception:
            pass
        # variable reference?
        if s in env:
            return float(env[s])
    raise ValueError(f"Unresolvable arg: {val}")

# ---- LLM adapter (OpenAI) ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

_client = None
def _get_client():
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError("OpenAI client not available. Install openai>=1.40 or replace the adapter.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and not MOCK_MODE:
            raise RuntimeError("OPENAI_API_KEY not set. Export it or enable MOCK_AGENT=1.")
        _client = OpenAI(api_key=api_key)
    return _client

def llm_plan_step(history: List[Dict[str,str]], model: str) -> Dict[str, Any]:
    if MOCK_MODE:
        return {"thought":"Add 3 and 5", "action":{"name":"add","args":{"a":3,"b":5}}}
    client = _get_client()
    msgs = [{"role":"system","content":SYS_PROMPT}] + history
    r = client.chat.completions.create(model=model, messages=msgs, temperature=0)
    txt = (r.choices[0].message.content or "").strip()
    block = extract_json_block(txt)
    if not block:
        return {"thought":"fallback add","action":{"name":"add","args":{"a":0,"b":0}}}
    try:
        return json.loads(block)
    except Exception:
        return {"thought":"parse error -> fallback add","action":{"name":"add","args":{"a":0,"b":0}}}

def solve(question: str, model: str = "gpt-4o-mini", max_steps: int = 8) -> Tuple[List[AFMStep], str]:
    if MOCK_MODE:
        steps = [
            AFMStep(function="add", thought="Add 3 and 5", tool_call={"name":"add","args":{"a":3,"b":5}}, tool_result=8.0),
            AFMStep(function="multiply", thought="Multiply by 2", tool_call={"name":"multiply","args":{"a":8,"b":2}}, tool_result=16.0),
        ]
        return steps, "16"

    steps: List[AFMStep] = []
    env = extract_env_vars(question)   # NEW: seed variable bindings from question
    history: List[Dict[str,str]] = [{"role":"user","content":f"Question: {question}"}]
    last_value: Any = None

    for _ in range(max_steps):
        plan = llm_plan_step(history, model=model)
        thought = plan.get("thought","")
        action = plan.get("action",{}) or {}
        raw_name = action.get("name","add")
        raw_args = action.get("args",{}) or {}

        name = normalize_action_name(str(raw_name))
        if name == "stop":
            steps.append(AFMStep(function="stop", thought=thought or "Stopping.", tool_call={"name":"stop","args":{}}, tool_result=None))
            break

        if name not in VALID:
            steps.append(AFMStep(function="plan",
                                 thought=f"Unsupported action '{raw_name}', normalized '{name}'. Skipping.",
                                 tool_call={"name":str(raw_name),"args":raw_args},
                                 tool_result=None))
            history.append({"role":"assistant","content":json.dumps(plan)})
            history.append({"role":"system","content":f"Observation: unsupported action '{name}'."})
            continue

        # NEW: resolve/validate numeric args using env
        try:
            args: Dict[str, float] = {k: resolve_arg(v, env) for k, v in raw_args.items()}
        except Exception as e:
            steps.append(AFMStep(function="plan",
                                 thought=f"Bad arguments for '{name}': {e}. Skipping.",
                                 tool_call={"name":name,"args":raw_args},
                                 tool_result=None))
            history.append({"role":"assistant","content":json.dumps(plan)})
            history.append({"role":"system","content":f"Observation: invalid args for {name}: {e}."})
            continue

        call, result = run_tool(name, **args)
        steps.append(AFMStep(function=name, thought=thought, tool_call=call, tool_result=result))

        # Feedback loop
        history.append({"role":"assistant","content":json.dumps(plan)})
        history.append({"role":"system","content":f"Observation: {result}."})
        last_value = result

    # Finalize answer
    client = _get_client()
    history.append({"role":"user","content":"Given the observations, return only the final numeric answer."})
    r = client.chat.completions.create(model=model, messages=history, temperature=0)
    final_answer = (r.choices[0].message.content or "").strip()
    return steps, final_answer