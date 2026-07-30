from abc import ABC, abstractmethod

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
        self.stream = []

    def on_text(self, text):
        self.stream.append(text)

    def finish(self):
        def generator():
            for text in self.stream:
                yield text

        return generator()