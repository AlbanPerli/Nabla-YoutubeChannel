class Activation:
    
    def __init__(self):
       self.output = None
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError
    
    def __call__(self, *args, **kwds):
        if len(kwds) == 0:
            return self.forward(*args)
        if "b" in kwds:
            return self.backward(kwds["b"])