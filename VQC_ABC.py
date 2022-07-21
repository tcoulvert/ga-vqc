from abc import ABC, abstractmethod

class VQC(ABC):
    @abstractmethod
    def circuit():
        pass
    
    @abstractmethod
    def train():
        pass
    
    