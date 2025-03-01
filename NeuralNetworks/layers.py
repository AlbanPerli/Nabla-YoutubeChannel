import math
from lossFunctions import MSE
from activationLayers import *
from denseLayer import Dense
from abstractLayer import Layer
from activations import *
from perceptron import Perceptron

class CNN1D(Layer):
    
    def __init__(self, n_filters, filter_size, learning_rate=0.01):
        self.filter_size = filter_size
        self.filters = [Perceptron(filter_size, learning_rate) for _ in range(n_filters)]
        self.outputs = [0.0 for _ in range(n_filters)]
        self.chunked_input = []
        
    def forward(self, x):
        self.chunked_input = [x[i:i+self.filter_size] for i in range(len(x)-self.filter_size+1)]
        for chunk in self.chunked_input:
            for i, f in enumerate(self.filters):
                self.outputs[i] = f(chunk)
        
        self.outputs = [f(x) for f in self.filters]
        return self.outputs
    
    def backward(self, output_grad):
        out = [f(b=g) for f,g in zip(self.filters, output_grad)]
        out_grad = []
        for i in range(len(self.filters[0].weights)):
            out_grad.append(sum([f.grads[i] for f in self.filters]))
        return out_grad

class Parallel(Layer):
    
    def __init__(self, *args):
        self.layers = args
        self.struct_outputs = []
        self.flatten_outputs = []

    def forward(self, x):
        self.struct_outputs = []
        self.flatten_outputs = []
        for layer in self.layers:
            self.struct_outputs.append(layer(x))
        for o in self.struct_outputs:
            if isinstance(o, list):
                self.flatten_outputs.extend(o)
            else:
                self.flatten_outputs.append(o)
        return self.flatten_outputs
    
    def backward(self, output_grad):
        if len(output_grad) != len(self.flatten_outputs):
            raise ValueError("output_grad: {}, is not the same size as the number of outputs:{}".format(len(output_grad), len(self.flatten_outputs)))
        struct_output_sizes = [len(o) if isinstance(o, list) else 1 for o in self.struct_outputs]
        output_grads = []
        for size in struct_output_sizes:
            output_grads.append(output_grad[:size])
            output_grad = output_grad[size:]
        o = [layer(b=g) for layer, g in zip(self.layers, output_grad)]
        return o

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
    
module = Module( Dense(4, 8, learning_rate=0.1),
                     SigmoidLayer(8),
                     Dense(8, 3, learning_rate=0.1),
    )

loss = MSE()

if __name__ == "__main__":
    for i in range(100):
        x = [0.4, -1.0, 0.5, 0.2]
        y = module(x)
        errors = loss(y, [0.2,0.7,0.1], grad=True)
        out_grad = module(b=loss.grads)
        