class Activation:
    
    def __init__(self):
       self.output = None
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError
    
    def __call__(self, *args, **kwds):
        # if arg name is f then call forward else call backward
        if "f" in kwds:
            # extract the value of f
            return self.forward(kwds["f"])
        return self.backward(kwds["b"])