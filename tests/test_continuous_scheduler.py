from engine.continuous_scheduler import ContinuousBatchScheduler


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeSession:
    def __init__(self, requests, steps_to_finish=2):
        self.requests = list(requests)
        self.steps_to_finish = steps_to_finish
        self.steps = 0
        self._finished = False

    def step(self):
        self.steps += 1
        if self.steps >= self.steps_to_finish:
            self._finished = True

    @property
    def is_finished(self):
        return self._finished


class FakeRequest:
    def __init__(self, prompt):
        self.prompt = prompt


def test_continuous_scheduler_starts_new_session_while_old_one_is_active():
    clock = FakeClock()
    scheduler = ContinuousBatchScheduler(
        max_batch_size=2,
        max_wait_time=1.0,
        session_factory=lambda batch: FakeSession(batch, steps_to_finish=4),
        clock=clock,
    )

    r1 = FakeRequest("one")
    r2 = FakeRequest("two")

    scheduler.submit(r1)

    started_sessions, completed_sessions = scheduler.step()

    assert len(started_sessions) == 1
    assert len(completed_sessions) == 0
    assert len(scheduler._active_sessions) == 1

    scheduler.submit(r2)
    started_sessions, completed_sessions = scheduler.step()

    assert len(started_sessions) == 0
    assert len(completed_sessions) == 0
    assert len(scheduler._active_sessions) == 1

    clock.advance(1.1)
    started_sessions, completed_sessions = scheduler.step()

    assert len(started_sessions) == 1
    assert len(completed_sessions) == 0
    assert len(scheduler._active_sessions) == 2
