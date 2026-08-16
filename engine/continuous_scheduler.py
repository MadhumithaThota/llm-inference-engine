from __future__ import annotations

import time
from collections.abc import Callable


class ContinuousBatchScheduler:
    def __init__(
        self,
        *,
        max_batch_size: int = 4,
        max_wait_time: float = 0.05,
        session_factory: Callable[[list], object],
        clock=time.monotonic,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._session_factory = session_factory
        self._clock = clock
        self._pending_requests: list = []
        self._active_sessions: list = []
        self._pending_since: float | None = None
        self._next_session_id = 1

    def status(self) -> dict[str, int]:
        return {
            "pending_requests": len(self._pending_requests),
            "active_sessions": len(self._active_sessions),
        }

    def submit(self, request):
        self._pending_requests.append(request)

        if self._pending_since is None:
            self._pending_since = self._clock()

        print(
            "[scheduler] queued request",
            f"pending={len(self._pending_requests)}",
            f"active_sessions={len(self._active_sessions)}",
        )

    def has_work(self) -> bool:
        return bool(self._pending_requests or self._active_sessions)

    def _start_session(self, batch: list):
        session = self._session_factory(batch)
        session.session_id = self._next_session_id
        self._next_session_id += 1
        self._active_sessions.append(session)
        print(
            f"[scheduler] started session {session.session_id}",
            f"batch_size={len(batch)}",
            f"active_sessions={len(self._active_sessions)}",
            f"pending={len(self._pending_requests)}",
        )
        return session

    def _should_start_session(self, now: float) -> bool:
        if not self._pending_requests:
            return False

        if not self._active_sessions:
            return True

        if len(self._pending_requests) >= self.max_batch_size:
            return True

        if self._pending_since is None:
            return False

        return (now - self._pending_since) >= self.max_wait_time

    def _maybe_start_sessions(self):
        now = self._clock()
        started_sessions = []

        while self._should_start_session(now):
            batch = self._pending_requests[: self.max_batch_size]
            del self._pending_requests[: len(batch)]

            started_sessions.append(self._start_session(batch))

            if self._pending_requests:
                self._pending_since = now
            else:
                self._pending_since = None

        return started_sessions

    def step(self):
        """
        Advance active sessions by one decode step.

        Returns a tuple of (started_sessions, completed_sessions).
        """
        started_sessions = self._maybe_start_sessions()
        completed_sessions = []

        for session in list(self._active_sessions):
            session.step()

            if session.is_finished:
                self._active_sessions.remove(session)
                completed_sessions.append(session)
                print(
                    f"[scheduler] completed session {session.session_id}",
                    f"active_sessions={len(self._active_sessions)}",
                    f"pending={len(self._pending_requests)}",
                )

        if self._pending_requests:
            self._maybe_start_sessions()

        return started_sessions, completed_sessions
