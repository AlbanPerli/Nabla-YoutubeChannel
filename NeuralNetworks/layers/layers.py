import math
from .activationLayers import *
from .abstractLayer import Layer
from .perceptron import Perceptron

class CNN1D(Layer):
    
    def __init__(self, n_filters, filter_size, learning_rate=0.01):
        super().__init__()
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