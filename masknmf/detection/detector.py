from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    
    @abstractmethod
    def detect_instances(self, data):
        pass
