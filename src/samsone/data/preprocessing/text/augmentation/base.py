from abc import ABC, abstractmethod
import re


class TextPreprocessor(ABC):
    @abstractmethod
    def __call__(self, text):
        pass


class ToLowerCase(TextPreprocessor):
    def __call__(self, text):
        return text.lower()


class RemoveExcessiveWhitespace(TextPreprocessor):
    def __call__(self, text: str) -> str:
        # Replaces 4 or more whitespaces with an empty string
        return re.sub(r"\s{4,}", " ", text)


class RemoveNonLatin(TextPreprocessor):
    def __call__(self, text: str) -> str:
        # Replaces any character outside the basic ASCII range with an empty string
        return re.sub(r"[^\x00-\x7F]", "", text)


class RemoveRepeatedSpecialChars(TextPreprocessor):
    def __call__(self, text: str) -> str:
        # Replaces --- or ### (3+) with a single instance of that char
        text = re.sub(r"-{3,}", "-", text)
        text = re.sub(r"#{3,}", "#", text)
        return text


class TextPreprocessorPipeline(TextPreprocessor):
    def __init__(self, preprocessors: list[TextPreprocessor]):
        self.preprocessors = preprocessors

    def __call__(self, text: str) -> str:
        for proc in self.preprocessors:
            text = proc(text)
        return text
