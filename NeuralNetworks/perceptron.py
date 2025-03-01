import random
import math
from loss import MeanSquareError, d_MeanSquareError
from abstractLayer import Layer
from activations import *

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
        
    def __call__(self, *args, **kwds):
         # if arg name is f then call forward else call backward
        if "f" in kwds:
            # extract the value of f
            return self.forward(kwds["f"])
        return self.backward(kwds["b"])
        
if __name__ == "__main__":
    p = Perceptron(3,learning_rate=0.1)
    a = Sigmoid()
    loss = MeanSquareError
    
    for i in range(1000):
        
        yLin = p(f=[0.4, -1.0, 0.5])
        yNLin = a(f=yLin)
        
        error = loss([yNLin], [0.9])
        
        d_loss = d_MeanSquareError([yNLin], [0.9])
        d = a(b=d_loss[0])
        dP = p(b=d)
        
        print(yNLin)
        print(p.weights, p.bias)
        print()

