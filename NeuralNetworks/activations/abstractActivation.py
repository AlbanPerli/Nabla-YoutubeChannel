from abc import ABC, abstractmethod

class Activation(ABC):
    
    def __init__(self):
       self.output = None
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, x):
        pass
    
    def __call__(self, *args, **kwds):
        if len(kwds) == 0:
            return self.forward(*args)
        if "b" in kwds:
            return self.backward(kwds["b"])