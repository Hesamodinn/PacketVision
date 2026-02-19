# ui_bus.py
from dataclasses import dataclass
from typing import Any, Dict
import queue

@dataclass
class UIEvent:
    type: str
    data: Dict[str, Any]

class EventBus:
    def __init__(self):
        self.q: "queue.Queue[UIEvent]" = queue.Queue()

    def emit(self, type: str, **data):
        self.q.put(UIEvent(type=type, data=data))
