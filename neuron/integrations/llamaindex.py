from typing import Any, List, Optional
from llama_index.core.memory import BaseMemory
from neuron.memory import Memory

class NEURONMemoryStore(BaseMemory):
    def __init__(self, user_id: str, NEURON_instance: Optional[Memory] = None):
        self.user_id = user_id
        self.NEURON = NEURON_instance or Memory()

    def get(self, input_str: Optional[str] = None, **kwargs: Any) -> List[Any]:
        if not input_str:
            return []
        
        context = self.NEURON.retrieve(input_str, self.user_id)
        # Format for LlamaIndex
        return [f"Long-term belief: {b['label']}" for b in context.get("direct_beliefs", [])]

    def put(self, message: str, role: str = "user") -> None:
        if role == "user":
            self.NEURON.process(message, self.user_id)

    def reset(self) -> None:
        pass
        
    @classmethod
    def from_defaults(cls, user_id: str) -> "NEURONMemoryStore":
        return cls(user_id=user_id)
