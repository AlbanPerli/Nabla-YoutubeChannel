from layers.abstractLayer import Layer


class FlattenLayer(Layer):
    
    def __init__(self):
        self.outputs = []
    
    def forward(self, x):
        if type(x) is not list:
            raise ValueError("x: {}, is not a list".format(x))
        if len(x) == 0:
            raise ValueError("x: {}, is empty".format(x))
        self.outputs = []
        for i in x:
            if type(i) is list:
                self.outputs.extend(i)
            else:
                self.outputs.append(i)
        return self.outputs
    
    def backward(self, output_grad):
        return output_grad