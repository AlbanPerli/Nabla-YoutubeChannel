import random
from layers.abstractLayer import Layer
from activations.activations import *

class Perceptron(Layer):
    
    def __init__(self, n_inputs, 
                 weights = None, 
                 use_bias = False, 
                 learning_rate=0.01, 
                 capWeights=3.0):
        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
            
        self.use_bias = use_bias
        if use_bias:
            self.bias = random.uniform(-1, 1)
        else:
            self.bias = None
        
        self.learning_rate = learning_rate
        self.capWeights = capWeights
        
        self.output = 0.0
        self.inputs = []
        self.grads = [0.0 for _ in range(n_inputs)]
         
    def forward(self, inputs):
        assert type(inputs) is list
        assert type(inputs[0]) is not list
        assert len(inputs) > 0
        assert len(inputs) == len(self.weights)
        
        self.inputs.append(inputs)
        if self.use_bias:
            self.output = sum([i*w for i,w in zip(inputs, self.weights)]) + self.bias
        else:
            self.output = sum([i*w for i,w in zip(inputs, self.weights)])
        return self.output
    
    def backward(self, error):
        for inputs in self.inputs:
            for i in range(len(self.weights)):
                self.grads[i] = error * inputs[i]
                self.weights[i] += self.learning_rate * self.grads[i]
            if self.use_bias:
                self.bias += self.learning_rate * error
        self.inputs = []
        return error