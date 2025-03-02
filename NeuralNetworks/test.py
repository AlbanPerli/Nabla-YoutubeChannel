import random
import math

from activations.activations import Sigmoid
from layers.perceptron import Perceptron
from layers.activationLayers import SigmoidLayer
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
    )

    loss = MSE()
    for i in range(100):
        x = [0.4, -1.0, 0.5, 0.2]
        y = module(x)
        errors = loss(y, [0.2,0.7,0.1], grad=True)
        out_grad = module(b=loss.grads)
        print("Error: ", errors)

