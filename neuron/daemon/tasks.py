from celery import Celery
from neuron.daemon.coherence import CoherenceDaemon
from neuron.graph.postgres_store import PostgresGraphStore
from neuron.config import settings

# Initialize Celery
app = Celery('neuron_daemon', broker=f"redis://{settings.redis_host}:6379/0")

# Lazy init store and daemon
_store = None
_daemon = None

def get_daemon():
    global _store, _daemon
    if _daemon is None:
        _store = PostgresGraphStore()
        _daemon = CoherenceDaemon(_store)
    return _daemon

@app.task
def run_coherence_cycle(user_id: str):
    daemon = get_daemon()
    daemon.run_cycle(user_id)

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Calls run_coherence_cycle for all users every 6 hours
    # This would need a way to get all active user IDs
    pass
