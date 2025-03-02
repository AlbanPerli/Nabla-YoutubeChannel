from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.inputs = None
        self.outputs = None

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, output_grad):
        pass

    def __call__(self, *args, **kwds):
        if len(kwds) == 0 and len(args) == 1:
            return self.forward(*args)
        if "b" in kwds:
            return self.backward(kwds["b"])
        