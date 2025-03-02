from layers.abstractLayer import Layer

class Module(Layer):
    
    def __init__(self, *args):
        self.layers = args
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer(b=output_grad)
        return output_grad