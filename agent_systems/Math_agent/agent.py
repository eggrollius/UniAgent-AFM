from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI  # openai>=1.40
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .tools import run_tool
from .afm_schema import AFMStep

# ---------- Config ----------
VALID: set[str] = {
    "add", "subtract", "multiply", "divide", "square", "power", "sqrt", "stop"
}
TOOLS_DOC = "add(a,b) | subtract(a,b) | multiply(a,b) | divide(a,b) | square(x) | power(a,b) | sqrt(x) | stop()"
FINAL_TAG = "final"  # used for <final>…</final>
ASSIGN_RE = re.compile(r"\b([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\b")
FINAL_RE = re.compile(rf"<{FINAL_TAG}>\s*([-+]?\d+(?:\.\d+)?)\s*</{FINAL_TAG}>")

# ---------- Prompt loading via Jinja ----------
def _render_system_prompt() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tmpl_dir = os.path.join(repo_root, "prompts")
    env = Environment(
        loader=FileSystemLoader(tmpl_dir),
        autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("math_agent_system.j2")
    return tmpl.render(
        tool_list=TOOLS_DOC,
        tool_names=sorted(list(VALID)),
    )

# ---------- Utilities ----------
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
        if s == "r":
            if last_value is None:
                raise ValueError("r is not available before the first step")
            return float(last_value)
        try:
            return float(s)
        except Exception:
            pass
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

# ---------- OpenAI client ----------
_client: Optional[OpenAI] = None
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=api_key)
    return _client

# ---------- Planner step (schema-enforced) ----------
def llm_plan_step(history: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    client = _get_client()
    msgs = history  # system prompt is inserted by caller

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
                            "enum": sorted(list(VALID))
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

# ---------- Final answer extraction ----------
def extract_final_answer(text: str) -> str:
    m = FINAL_RE.search(text)
    if not m:
        raise ValueError(f"Final answer missing or not wrapped in <{FINAL_TAG}> tags.")
    return m.group(1).strip()

# ---------- Public solve ----------
def solve(question: str, model: str = "gpt-4o-mini", max_steps: int = 8) -> Tuple[List[AFMStep], str]:
    client = _get_client()
    system_prompt = _render_system_prompt()

    steps: List[AFMStep] = []
    env = extract_env_vars(question)
    history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"}
    ]
    last_value: Optional[float] = None

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

        # resolve args strictly (aware of r)
        args: Dict[str, float] = {k: resolve_arg(v, env, last_value) for k, v in raw_args.items()}
        call, result = run_tool(name, **args)
        steps.append(AFMStep(function=name, thought=thought, tool_call=call, tool_result=result))

        # feedback
        history.append({"role": "assistant", "content": json.dumps(plan)})
        history.append({"role": "system", "content": f"Observation: {result}."})
        last_value = result

    # ask for final numeric answer inside tags
    history.append({"role": "user", "content": f"Given the observations, return only the final numeric answer as <{FINAL_TAG}>N</{FINAL_TAG}>."})
    r = client.chat.completions.create(model=model, messages=history, temperature=0)
    final_raw = (r.choices[0].message.content or "").strip()
    final_answer = extract_final_answer(final_raw)
    return steps, final_answer