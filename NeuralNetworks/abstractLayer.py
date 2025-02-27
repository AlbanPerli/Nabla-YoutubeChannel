class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        if "f" in kwds:
            return self.forward(kwds["f"])
        return self.backward(kwds["b"])
    