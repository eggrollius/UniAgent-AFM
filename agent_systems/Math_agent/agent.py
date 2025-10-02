import os, re, json
from typing import List, Dict, Any
from .tools import run_tool
from .afm_schema import AFMStep

# Replace with your preferred client if needed
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYS = ("You are a careful math planner. At each step, decide ONE action among "
       "add(a,b), multiply(a,b), square(x). Provide a short thought. "
       "Return strict JSON: {\"thought\": str, \"action\": {\"name\": str, \"args\": {...}}} .")
JSON = r"\{(?:[^{}]|(?R))*\}"

def llm_plan_step(history: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    msgs = [{"role":"system","content":SYS}] + history
    r = client.chat.completions.create(model=model, messages=msgs, temperature=0)
    txt = r.choices[0].message.content
    m = re.search(JSON, txt, flags=re.S)
    if not m:
        return {"thought":"fallback add", "action":{"name":"add","args":{"a":0,"b":0}}}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"thought":"parse error -> fallback add", "action":{"name":"add","args":{"a":0,"b":0}}}

def solve(question: str, model: str = "gpt-4o-mini", max_steps: int = 8):
    steps: List[AFMStep] = []
    history: List[Dict[str,str]] = [{"role":"user","content":f"Question: {question}"}]
    last_value = None

    for _ in range(max_steps):
        plan = llm_plan_step(history, model)
        thought = plan.get("thought","")
        action = plan.get("action", {})
        name = action.get("name","add")
        args = action.get("args", {})

        call, result = run_tool(name, **args)
        steps.append(AFMStep(function=name, thought=thought, tool_call=call, tool_result=result))

        history.append({"role":"assistant", "content":json.dumps(plan)})
        history.append({"role":"system", "content":f"Observation: {result}."})
        last_value = result

        # simple stop rule: if model asks to stop (not used here), or heuristic by steps
        if name == "stop":
            break

    # Final answer prompt
    history.append({"role":"user","content":"Given the observations, return only the final numeric answer."})
    r = client.chat.completions.create(model=model, messages=history, temperature=0)
    final = r.choices[0].message.content.strip()
    return steps, final