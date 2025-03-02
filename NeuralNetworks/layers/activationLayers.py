
from .abstractLayer import Layer
from activations.activations import *

class SoftmaxLayer(Layer):
    
    # def __init__(self, n_inputs):
    #     self.n_inputs = n_inputs
        
    def forward(self, x):
        exps = [math.exp(i) for i in x]
        self.outputs = [i/sum(exps) for i in exps]
        return self.outputs
    
    def backward(self, output_grad):
        out_grad = [o * (1.0 - o) * g for o, g in zip(self.outputs, output_grad)]
        return out_grad

class ActivationLayer(Layer):
    
    def __init__(self, n_neurons, activation):
        self.activations = [activation() for _ in range(n_neurons)]
        
    def forward(self, x):
        self.outputs = [a(i) for a,i in zip(self.activations, x)]
        return self.outputs
    
    def backward(self, output_grad):
        return [a(b=g) for a,g in zip(self.activations, output_grad)]

class SigmoidLayer(ActivationLayer):
    
    def __init__(self, n_neurons):
        super().__init__(n_neurons, Sigmoid)

class TanhLayer(ActivationLayer):
    
    def __init__(self, n_neurons):
        super().__init__(n_neurons, Tanh)
        
class ReLULayer(ActivationLayer):
    
    def __init__(self, n_neurons):
        super().__init__(n_neurons, ReLU)
        
class LeakyReLULayer(ActivationLayer):
    
    def __init__(self, n_neurons):
        super().__init__(n_neurons, LeakyReLU)
        