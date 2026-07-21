from abc import ABC,abstractmethod
from ..enums import Provider
from .parser import Parser
class ProviderParser(Parser,ABC):
    @property
    @abstractmethod
    def provider(self)->Provider:raise NotImplementedError
