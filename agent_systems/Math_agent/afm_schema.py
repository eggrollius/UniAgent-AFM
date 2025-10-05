from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import time, json

AFM_VERSION = "0.1"

@dataclass
class AFMStep:
    function: str            # e.g., add, multiply, square
    thought: str             # brief rationale from the model
    tool_call: Dict[str, Any]
    tool_result: Any

@dataclass
class AFMTrajectory:
    id: str
    task: str                # "MathQA"
    question: str
    context: str
    steps: List[AFMStep]
    answer: str
    meta: Dict[str, Any]

    def to_json(self) -> str:
        rec = asdict(self)
        rec["meta"]["afm_version"] = AFM_VERSION
        rec["meta"]["timestamp"] = int(time.time())
        return json.dumps(rec, ensure_ascii=False)