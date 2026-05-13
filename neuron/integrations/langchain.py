from typing import List, Any, Dict, Optional
from langchain.schema import BaseMemory
from pydantic import Field
from neuron.memory import Memory

class NEURONMemory(BaseMemory):
    user_id: str
    memory_key: str = "NEURON_history"
    NEURON: Memory = Field(default_factory=Memory)

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Use the query/input to retrieve relevant context from neuron
        query = inputs.get("input", "") or inputs.get("query", "")
        context = self.NEURON.retrieve(query, self.user_id)
        
        # Format the context for the LLM
        formatted_context = "NEURON Long-term Memory Context:\n"
        for b in context.get("direct_beliefs", []):
            formatted_context += f"- {b['label']}\n"
        
        if context.get("unresolved_tensions"):
            formatted_context += "\nUnresolved Tensions:\n"
            for t in context["unresolved_tensions"]:
                formatted_context += f"- Tension: {t['belief_a']} vs {t['belief_b']}\n"
                
        return {self.memory_key: formatted_context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        # Process the new interaction into NEURON
        input_str = inputs.get("input", "") or inputs.get("query", "")
        output_str = outputs.get("output", "") or outputs.get("response", "")
        
        # We only process the input to abstract the user's intent/facts
        if input_str:
            self.NEURON.process(input_str, self.user_id)

    def clear(self) -> None:
        # NEURON memory is persistent, so 'clear' is usually a no-op or scoped
        pass
