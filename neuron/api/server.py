from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict
from neuron.memory import Memory
from neuron.daemon.coherence import CoherenceDaemon
from neuron.models import Node

app = FastAPI(title="NEURON API", description="Biologically Inspired Memory Layer for AI Agents")
memory = Memory()
daemon = CoherenceDaemon(memory.store)

class ProcessRequest(BaseModel):
    text: str
    user_id: str

class RetrieveRequest(BaseModel):
    query: str
    user_id: str

@app.post("/v1/memory/process")
async def process_memory(req: ProcessRequest):
    try:
        node = memory.process(req.text, req.user_id)
        return {"status": "success", "node": node}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/memory/retrieve")
async def retrieve_memory(req: RetrieveRequest):
    try:
        context = memory.retrieve(req.query, req.user_id)
        return {"status": "success", "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/memory/consolidate/{user_id}")
async def consolidate_memory(user_id: str):
    try:
        daemon.run_cycle(user_id)
        return {"status": "success", "message": f"Consolidation cycle completed for {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
