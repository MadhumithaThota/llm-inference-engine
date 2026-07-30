from threading import Thread

from engine.generation_engine import generation_engine
from engine.request_queue import request_queue


def worker_loop():

    while True:

        request = request_queue.get()

        try:

            result = generation_engine.generate(request)

            request.future.set_result(result)

        except Exception as e:

            request.future.set_exception(e)

        finally:

            request_queue.task_done()


worker = Thread(
    target=worker_loop,
    daemon=True,
)

worker.start()