import logging
import pytest
from neuron.adaptive.adaptive_memory import AdaptiveMemory
from neuron.config import settings

logger = logging.getLogger(__name__)


def test_adaptive_flow():
    prev_store = settings.graph_store_type
    prev_adaptive = settings.enable_adaptive_memory
    prev_mode = settings.adaptive_mode
    settings.enable_adaptive_memory = True
    settings.graph_store_type = "in_memory"
    settings.adaptive_mode = "manual"

    brain = AdaptiveMemory()
    user_id = "test_adaptive_user"

    brain.process("The capital of France is Paris.", user_id)
    brain.process("The Eiffel Tower is in Paris.", user_id)

    activities = brain.store.get_users_needing_maintenance(window_hours=1)
    assert user_id in activities, "WRITE activity should be logged"

    result = brain.retrieve("Where is the Eiffel Tower?", user_id, score_feedback=0.9)
    assert "event_id" in result

    events = brain.store.get_retrieval_events(user_id, limit=5)
    assert events, "Retrieval event should be persisted"
    assert events[0].strategy_id

    report_before = brain.strategy_report()
    for _ in range(50):
        brain.store.log_retrieval_event(events[0])

    brain.force_sleep(user_id)
    report_after = brain.strategy_report()
    assert len(report_after) >= len(report_before)

    health = brain.graph_health(user_id)
    assert health["total_nodes"] >= 2
    assert health["status"] in ("healthy", "noisy", "empty")

    settings.graph_store_type = prev_store
    settings.enable_adaptive_memory = prev_adaptive
    settings.adaptive_mode = prev_mode
