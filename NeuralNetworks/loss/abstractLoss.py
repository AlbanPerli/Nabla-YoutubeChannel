from abc import ABC, abstractmethod

class LossFunction(ABC):
    
    EPSILON = 1e-10
    
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