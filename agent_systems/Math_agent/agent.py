from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Tuple, Optional
from .tools import run_tool
from .afm_schema import AFMStep
from openai import OpenAI  # pip install openai>=1.40

# ---------- Strict config ----------
VALID: set[str] = {
    "add", "subtract", "multiply", "divide", "square", "power", "sqrt", "stop"
}

SYS_PROMPT = (
    "You are a careful math planner.\n"
    "At each step, you must choose ONE action from this exact set:\n"
    "  add(a,b) | subtract(a,b) | multiply(a,b) | divide(a,b) | square(x) | power(a,b) | sqrt(x) | stop()\n"
    "Arguments MUST be numeric literals or variables explicitly defined in the QUESTION (e.g., x=4),\n"
    "or the carry variable r which always refers to the MOST RECENT Observation value.\n"
    "You MUST NOT invent new variable names. Only use variables from the question or r.\n"
    "If the question uses words like 'X apples' or 'Y dollars' without assigning them numeric values,\n"
    "treat them as ordinary text, NOT as variables. Do not attempt to use them in tool calls.\n"
    "Return STRICT JSON ONLY (no prose, no markdown), exactly this schema:\n"
    '{"thought":"<short>","action":{"name":"add|subtract|multiply|divide|square|power|sqrt|stop","args":{...}}}\n'
    "If the task is complete, use name='stop' and args={}. "
    "Do NOT return anything except the JSON object. NEVER divide by zero. "
    "For negative bases, only use power(a,b) with integer b."
)

# ---------- Utilities ----------
ASSIGN_RE = re.compile(r"\b([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\b")

def extract_env_vars(question: str) -> Dict[str, float]:
    env: Dict[str, float] = {}
    for var, val in ASSIGN_RE.findall(question):
        env[var] = float(val)
    return env

def resolve_arg(val: Any, env: Dict[str, float], last_value: Optional[float]) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        # special carry variable
        if s == "r":
            if last_value is None:
                raise ValueError("r is not available before the first step")
            return float(last_value)
        # numeric string?
        try:
            return float(s)
        except Exception:
            pass
        # variable reference from question
        if s in env:
            return float(env[s])
    raise ValueError(f"Unresolvable arg: {val!r}")

def require_args_for_action(name: str, args: Dict[str, Any]) -> None:
    if name in {"add", "subtract", "multiply", "divide", "power"}:
        if not {"a", "b"} <= set(args.keys()):
            raise ValueError(f"{name} requires args {{a,b}}; got {list(args.keys())}")
    elif name in {"square", "sqrt"}:
        if "x" not in args:
            raise ValueError(f"{name} requires arg {{x}}")
    elif name == "stop":
        pass

# ---------- OpenAI adapter (strict) ----------
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

    schema = {
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
                        "name": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide", "square", "power", "sqrt", "stop"]
                        },
                        "args": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "a": {"type": ["number", "string"]},
                                "b": {"type": ["number", "string"]},
                                "x": {"type": ["number", "string"]}
                            }
                        }
                    }
                }
            }
        }
    }

    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    txt = (resp.choices[0].message.content or "").strip()
    plan = json.loads(txt)

    # strict checks
    if not isinstance(plan, dict):
        raise ValueError("Planner output must be a JSON object.")
    if "thought" not in plan or "action" not in plan:
        raise ValueError("Planner output must contain 'thought' and 'action'.")
    action = plan["action"]
    if not isinstance(action, dict) or "name" not in action or "args" not in action:
        raise ValueError("Planner 'action' must have 'name' and 'args'.")
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
            steps.append(AFMStep(function="stop", thought=thought,
                                 tool_call={"name": "stop", "args": {}}, tool_result=None))
            break

        # resolve args strictly
        try:
            args: Dict[str, float] = {k: resolve_arg(v, env, last_value) for k, v in raw_args.items()}
        except ValueError as err:
            # Feed the error back to the planner so it can correct course.
            history.append({"role": "assistant", "content": json.dumps(plan)})
            history.append({"role": "system", "content": f"Error: {err}. Use only numeric literals, question-defined variables, or r."})
            continue

        call, result = run_tool(name, **args)
        steps.append(AFMStep(function=name, thought=thought, tool_call=call, tool_result=result))

        # feedback
        history.append({"role": "assistant", "content": json.dumps(plan)})
        history.append({"role": "system", "content": f"Observation: {result}."})
        last_value = result  # <-- keep result in sync

    # final numeric answer
    client = _get_client()
    history.append({"role": "user", "content": "Given the observations, return only the final numeric answer."})
    r = client.chat.completions.create(model=model, messages=history, temperature=0)
    final_answer = (r.choices[0].message.content or "").strip()
    return steps, final_answer
