from typing import Any, Dict

class ToolError(Exception):
    pass

def add(a: float, b: float) -> float:
    return float(a) + float(b)

def multiply(a: float, b: float) -> float:
    return float(a) * float(b)

def square(x: float) -> float:
    return float(x) * float(x)

def run_tool(name: str, **kwargs) -> (Dict[str, Any], Any):
    if name == "add":
        res = add(**kwargs)
    elif name == "multiply":
        res = multiply(**kwargs)
    elif name == "square":
        res = square(**kwargs)
    else:
        raise ToolError(f"Unknown tool: {name}")
    return {"name": name, "args": kwargs}, res