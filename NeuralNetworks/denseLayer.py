from abstractLayer import Layer
from perceptron import Perceptron

class Dense(Layer):
    
    def __init__(self, n_inputs, n_neurons, lr=0.01):
        self.neurons = [Perceptron(n_inputs, learning_rate=lr) for _ in range(n_neurons)]
        self.outputs = [0.0 for _ in range(n_neurons)]
    
    def forward(self, x):
        self.outputs = [n(x) for n in self.neurons]
        return self.outputs
    
    def backward(self, output_grad):
        out = [n(b=g) for n,g in zip(self.neurons, output_grad)]
        out_grad = []
        for i in range(len(self.neurons[0].weights)):
            out_grad.append(sum([n.grads[i] for n in self.neurons]))
        return out_grad

