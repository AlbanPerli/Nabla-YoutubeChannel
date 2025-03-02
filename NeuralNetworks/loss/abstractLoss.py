from abc import ABC, abstractmethod

class LossFunction(ABC):
    
    def __init__(self):
        self.grads = None
    
    @abstractmethod
    def forward(self, *args):
        pass
    
    @abstractmethod
    def gradients(self, *args):
        pass
    
    def __call__(self, *args, **kwds):    
        return self.forward(*args, **kwds)