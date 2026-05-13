import threading
import time
import logging
from neuron.config import settings

logger = logging.getLogger(__name__)

class SleepScheduler(threading.Thread):
    """
    Background daemon thread that periodically runs the SleepConsolidator
    on all users who have had recent activity.
    """
    def __init__(self, adaptive_memory):
        super().__init__()
        self.adaptive_memory = adaptive_memory
        self.daemon = True # Dies when main process exits
        self.stop_event = threading.Event()
        self.interval = settings.maintenance_interval_hours * 3600

    def stop(self):
        self.stop_event.set()

    def run(self):
        logger.info(f"SleepScheduler started. Interval: {settings.maintenance_interval_hours} hours.")
        
        while not self.stop_event.is_set():
            try:
                # 1. Identify users needing maintenance
                # We look back over the same interval as our sleep cycle
                users = self.adaptive_memory.store.get_users_needing_maintenance(
                    window_hours=settings.maintenance_interval_hours
                )
                
                if users:
                    logger.info(f"SleepScheduler: Found {len(users)} users needing maintenance.")
                    for user_id in users:
                        if self.stop_event.is_set(): break
                        self.adaptive_memory.force_sleep(user_id)
                else:
                    logger.debug("SleepScheduler: No active users found in window.")
                
            except Exception as e:
                logger.error(f"SleepScheduler error: {e}")
            
            # Wait for next interval, but check stop_event frequently
            # We sleep in 60s increments to remain responsive to shutdown
            elapsed = 0
            while elapsed < self.interval and not self.stop_event.is_set():
                time.sleep(min(60, self.interval - elapsed))
                elapsed += 60
        
        logger.info("SleepScheduler stopped.")
