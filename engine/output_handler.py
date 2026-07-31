from abc import ABC, abstractmethod
from queue import Queue


class OutputHandler(ABC):

    @abstractmethod
    def on_text(self, text: str):
        pass

    @abstractmethod
    def finish(self):
        pass


class BufferedOutputHandler(OutputHandler):

    def __init__(self):
        self.parts = []

    def on_text(self, text):
        self.parts.append(text)

    def finish(self):
        return "".join(self.parts)


class StreamingOutputHandler(OutputHandler):

    def __init__(self):
        self.queue = Queue()

    def on_text(self, text):
        self.queue.put(text)

    def finish(self):
        # Signal that streaming is complete
        self.queue.put(None)

    def generator(self):
        while True:
            text = self.queue.get()

            if text is None:
                break

            yield text