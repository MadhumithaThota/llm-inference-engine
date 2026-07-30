from concurrent.futures import Future

from engine.request import GenerationRequest
from engine.request_queue import request_queue


class Scheduler:

    def submit(self, request: GenerationRequest):
        request.future = Future()

        request_queue.put(request)

        return request.future


scheduler = Scheduler()