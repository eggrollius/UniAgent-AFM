from dataclasses import dataclass
from typing import Dict, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]

def add(a: float, b: float) -> float:
    return float(a) + float(b)

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return float(a) - float(b)

def multiply(a: float, b: float) -> float:
    return float(a) * float(b)

def divide(a: float, b: float) -> float:
    """Divide a by b (strict: b != 0)."""
    a = float(a)
    b = float(b)
    if b == 0.0:
        raise ValueError("divide: division by zero")
    return a / b

def square(x: float) -> float:
    return float(x) * float(x)

def power(a: float, b: float) -> float:
    """a ** b with strict domain rules:
       - if a < 0, b must be an integer (to avoid complex results).
    """
    a = float(a); b = float(b)
    # integer check for b
    if a < 0 and abs(b - round(b)) > 1e-12:
        raise ValueError("power: negative base requires integer exponent")
    return a ** b

def sqrt(x: float) -> float:
    """sqrt(x) with x >= 0."""
    x = float(x)
    if x < 0:
        raise ValueError("sqrt: domain error (x < 0)")
    return x ** 0.5

def _round_half_up(x: float, ndigits: int = 0) -> float:
    q = Decimal(str(x))
    if ndigits >= 0:
        exp = Decimal("1").scaleb(-ndigits)  # 10^(-ndigits)
    else:
        # e.g., ndigits = -1 → round to tens
        exp = Decimal("1").scaleb(-ndigits)  # still works with quantize
    return float(q.quantize(exp, rounding=ROUND_HALF_UP))

def round(x: float, ndigits: int = 0) -> float:  # noqa: A001 (intentional name)
    """Deterministic grade-school rounding: HALF_UP."""
    return _round_half_up(float(x), int(ndigits))

class ToolError(Exception):
    pass

def run_tool(name: str, **kwargs):
    call = {"name": name, "args": kwargs}
    if name == "add":
        res = add(**kwargs)
    elif name == "subtract":
        res = subtract(**kwargs)
    elif name == "multiply":
        res = multiply(**kwargs)
    elif name == "divide":
        res = divide(**kwargs)
    elif name == "square":
        res = square(**kwargs)
    elif name == "power":
        res = power(**kwargs)
    elif name == "sqrt":
        res = sqrt(**kwargs)
    elif name == "round":                       # ← NEW
        res = round(**kwargs)                   # ← NEW
    elif name == "stop":
        res = None
    else:
        raise ToolError(f"Unknown tool: {name}")
    return call, res

