import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional
import uuid


class JobManager:
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def submit(self, func: Callable[..., Any], *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {"status": "pending", "result": None, "error": None}
        self.executor.submit(self._run_job, job_id, func, args, kwargs)
        return job_id

    def _run_job(self, job_id: str, func: Callable[..., Any], args: tuple, kwargs: dict):
        self._update_job(job_id, status="running")
        try:
            result = func(*args, **kwargs)
            self._update_job(job_id, status="completed", result=result)
        except Exception as exc:  # pragma: no cover
            self._update_job(job_id, status="failed", error=str(exc))

    def _update_job(self, job_id: str, **updates):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.jobs.get(job_id)





