from queue import Empty
from threading import Thread

from engine.generator import generate
from engine.continuous_scheduler import ContinuousBatchScheduler
from engine.batching.batched_generator import BatchedGenerationSession
from engine.request_queue import request_queue


batch_scheduler = ContinuousBatchScheduler(
    max_batch_size=4,
    max_wait_time=0.25,
    session_factory=BatchedGenerationSession,
)


def _drain_queue():
    drained = []

    while True:
        try:
            drained.append(request_queue.get_nowait())
        except Empty:
            break

    return drained


def worker_loop():

    while True:
        drained_requests = []

        try:
            request = request_queue.get(timeout=0.05)
            drained_requests.append(request)
            drained_requests.extend(_drain_queue())
        except Empty:
            pass

        if not drained_requests and not batch_scheduler.has_work():
            continue

        if drained_requests:
            status = batch_scheduler.status()
            print(
                "[worker] received requests",
                f"count={len(drained_requests)}",
                f"pending={status['pending_requests']}",
                f"active_sessions={status['active_sessions']}",
            )

        try:
            for request in drained_requests:
                if request.stream:
                    result = generate(
                        request.prompt,
                        request.max_new_tokens,
                        request.max_context_length,
                        request.output_handler,
                        request.kv_cache,
                        request.temperature,
                        request.top_k,
                        request.top_p,
                        request.repetition_penalty,
                        request.stop_sequences,
                    )

                    if request.future is not None and not request.future.done():
                        request.future.set_result(result)

                    request_queue.task_done()
                else:
                    batch_scheduler.submit(request)

            _started_sessions, completed_sessions = batch_scheduler.step()

            for session in completed_sessions:
                responses = session.results or []

                for request, response in zip(session.requests, responses):
                    print("\nDEBUG RESULT:")
                    print(response)

                    if request.future is not None and not request.future.done():
                        request.future.set_result(response)

                    request_queue.task_done()

        except Exception as e:

            print("Batch inference exception:", e)

            for session in list(batch_scheduler._active_sessions):
                for request in session.requests:
                    if request.future is not None and not request.future.done():
                        request.future.set_exception(e)
                    request_queue.task_done()

            for request in batch_scheduler._pending_requests:
                if request.future is not None and not request.future.done():
                    request.future.set_exception(e)
                request_queue.task_done()

            batch_scheduler._active_sessions.clear()
            batch_scheduler._pending_requests.clear()
            batch_scheduler._pending_since = None


worker = Thread(
    target=worker_loop,
    daemon=True,
)

worker.start()
