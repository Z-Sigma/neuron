from typing import Any, Dict, Optional
from neuron.memory import Memory

class NEURONSharedMemory:
    """
    A shared memory layer for CrewAI agents.
    Allows multiple agents in a crew to share a common belief graph.
    """
    def __init__(self, crew_id: str, NEURON_instance: Optional[Memory] = None):
        self.crew_id = crew_id
        self.NEURON = NEURON_instance or Memory()

    def save(self, task_output: str, metadata: Dict[str, Any] = None):
        # Treat task output as a new event to abstract
        self.NEURON.process(task_output, self.crew_id)

    def search(self, query: str) -> str:
        # Retrieve context for an agent's current task
        context = self.NEURON.retrieve(query, self.crew_id)
        
        beliefs = [b['label'] for b in context.get('direct_beliefs', [])]
        if not beliefs:
            return "No relevant long-term memory found."
            
        return "Relevant Shared Beliefs:\n" + "\n".join(beliefs)
