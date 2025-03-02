import random
import math

from loss.binary_cross_entropy import BinaryCrossEntropy
from loss.categorical_cross_entropy import CategoricalCrossEntropy
from activations.activations import Sigmoid
from layers.perceptron import Perceptron
from layers.activationLayers import SigmoidLayer, SoftmaxLayer
from layers.denseLayer import Dense
from layers.layers import Module
from loss.mae import MAE
from loss.mse import MSE

if __name__ == "__main__":
    p = Perceptron(3, use_bias=True, learning_rate=0.1)
    a = Sigmoid()
    loss = MAE()
    
    for i in range(1000):
        
        yLin = p([0.4, -1.0, 0.5])
        yNLin = a(yLin)
        
        error = loss(yNLin, 0.9, grad=True)
        d_loss = loss.grads[0]
        
        d = a(b=d_loss)
        dP = p(b=d)
        
        print(yNLin)
        print(p.weights, p.bias)
        print()

        
        
    module = Module( Dense(4, 8, lr=0.1),
                     SigmoidLayer(8),
                     Dense(8, 3, lr=0.1),
                     SoftmaxLayer()
    )

    loss = CategoricalCrossEntropy()
    for i in range(200):
        x = [0.4, -1.0, 0.5, 0.2]
        y = module(x)
        print("Output: ", y)
        errors = loss(y, [1.0,0.0,0.0], grad=True)
        out_grad = module(b=loss.grads)
        print("Error: ", errors)



    module = Module( Dense(4, 8, lr=0.1),
                     SigmoidLayer(8),
                     Dense(8, 1, lr=0.1),
                     SigmoidLayer(1)
    )

    loss = BinaryCrossEntropy()
    for i in range(1000):
        x = [0.4, -1.0, 0.5, 0.2]
        y = module(x)
        print("Output: ", y)
        errors = loss(y[0], 1.0, grad=True)
        out_grad = module(b=loss.grads)
        print("Error: ", errors)