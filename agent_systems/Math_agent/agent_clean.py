# agent_systems/Math_agent/agent_clean.py
from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Tuple, Optional

from .tools import run_tool
from .afm_schema import AFMStep

# ---------- Strict config ----------
VALID: set[str] = {"add", "multiply", "square", "stop"}

SYS_PROMPT = (
    "You are a careful math planner.\n"
    "At each step, you must choose ONE action from this exact set:\n"
    "  add(a,b) | multiply(a,b) | square(x) | stop()\n"
    "Arguments MUST be numeric literals or previously defined variables from the question "
    "(e.g., if the question says x=4 then you may use x).\n"
    "Return STRICT JSON ONLY (no prose, no markdown), exactly this schema:\n"
    '{"thought":"<short>","action":{"name":"add|multiply|square|stop","args":{...}}}\n'
    "If the task is complete, use name='stop' and args={}. "
    "Do NOT return anything except the JSON object."
)

# ---------- Utilities ----------
def extract_json_block(text: str) -> str:
    """Return the first top-level {...} JSON object or raise."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise ValueError("Unclosed JSON object in model output.")

ASSIGN_RE = re.compile(r"\b([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\b")

def extract_env_vars(question: str) -> Dict[str, float]:
    env: Dict[str, float] = {}
    for var, val in ASSIGN_RE.findall(question):
        env[var] = float(val)
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
    raise ValueError(f"Unresolvable arg: {val!r}")

def require_args_for_action(name: str, args: Dict[str, Any]) -> None:
    if name == "add" or name == "multiply":
        if not {"a", "b"} <= set(args.keys()):
            raise ValueError(f"{name} requires args {{a,b}}; got {list(args.keys())}")
    elif name == "square":
        if "x" not in args:
            raise ValueError("square requires arg {x}")
    elif name == "stop":
        # allow empty or {}
        pass

# ---------- OpenAI adapter (strict) ----------
from openai import OpenAI  # pip install openai>=1.40

_client: Optional[OpenAI] = None
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=api_key)
    return _client

def llm_plan_step(history: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    client = _get_client()
    msgs = [{"role": "system", "content": SYS_PROMPT}] + history
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0)
    txt = (resp.choices[0].message.content or "").strip()
    block = extract_json_block(txt)
    plan = json.loads(block)

    # Strict schema checks
    if not isinstance(plan, dict):
        raise ValueError("Planner output must be a JSON object.")
    if "thought" not in plan or "action" not in plan:
        raise ValueError("Planner output must contain 'thought' and 'action'.")
    action = plan["action"]
    if not isinstance(action, dict) or "name" not in action or "args" not in action:
        raise ValueError("Planner 'action' must have 'name' and 'args' fields.")
    name = str(action["name"]).strip().lower()
    if name not in VALID:
        raise ValueError(f"Unsupported action name: {name}")
    if not isinstance(action["args"], dict):
        raise ValueError("Planner 'args' must be an object.")
    return plan

# ---------- Public solve ----------
def solve(question: str, model: str = "gpt-4o-mini", max_steps: int = 8) -> Tuple[List[AFMStep], str]:
    steps: List[AFMStep] = []
    env = extract_env_vars(question)
    history: List[Dict[str, str]] = [{"role": "user", "content": f"Question: {question}"}]
    last_value: Any = None

    for _ in range(max_steps):
        plan = llm_plan_step(history, model=model)
        thought = plan["thought"]
        action = plan["action"]
        name = str(action["name"]).strip().lower()
        raw_args = action["args"]

        require_args_for_action(name, raw_args)

        if name == "stop":
            steps.append(AFMStep(function="stop", thought=thought, tool_call={"name": "stop", "args": {}}, tool_result=None))
            break

        # resolve args strictly
        args: Dict[str, float] = {k: resolve_arg(v, env) for k, v in raw_args.items()}

        call, result = run_tool(name, **args)
        steps.append(AFMStep(function=name, thought=thought, tool_call=call, tool_result=result))

        # feedback
        history.append({"role": "assistant", "content": json.dumps(plan)})
        history.append({"role": "system", "content": f"Observation: {result}."})
        last_value = result

    # final numeric answer
    client = _get_client()
    history.append({"role": "user", "content": "Given the observations, return only the final numeric answer."})
    r = client.chat.completions.create(model=model, messages=history, temperature=0)
    final_answer = (r.choices[0].message.content or "").strip()
    return steps, final_answer
