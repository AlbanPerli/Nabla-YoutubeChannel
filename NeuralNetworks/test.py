import random
import math

from structure.flattenLayer import FlattenLayer
from structure.chunkLayer import Chunk1DLayer, Chunk2DLayer
from utils.drawNet import draw
from loss.binary_cross_entropy import BinaryCrossEntropy
from loss.categorical_cross_entropy import CategoricalCrossEntropy
from activations.activations import Sigmoid
from layers.perceptron import Perceptron
from layers.activationLayers import ReLULayer, SigmoidLayer, SoftmaxLayer, TanhLayer
from layers.denseLayer import Dense
from structure.module import Module
from loss.mae import MAE
from loss.mse import MSE

if __name__ == "__main__":
    
    chunker = Chunk1DLayer(3, 3, keep_dim=True)
    print(chunker([1,2,3,4,5,6,7,8,9,10]))
    flatten = FlattenLayer()
    print(flatten(chunker.outputs))
    
    # chunker2D = Chunk2DLayer(1, 5, auto_pad = True)
    # print(chunker2D([[1,2,3,4,5,6,7,8,9,10],
    #                  [1,2,3,4,5,6,7,8,9,10],
    #                  [1,2,3,4,5,6,7,8,9,10],
    #                  [1,2,3,4,5,6,7,8,9,10],
    #                  [1,2,3,4,5,6,7,8,9,10]]))
    
    # p = Perceptron(3, use_bias=True, learning_rate=0.1)
    # a = Sigmoid()
    # loss = MAE()
    
    # for i in range(1000):
        
    #     yLin = p([0.4, -1.0, 0.5])
    #     yNLin = a(yLin)
        
    #     error = loss(yNLin, 0.9, grad=True)
    #     d_loss = loss.grads[0]

    #     d = a(b=d_loss)
    #     dP = p(b=d)
        
        
    # module = Module( Dense(4, 8, lr=0.1),
    #                  SigmoidLayer(8),
    #                  Dense(8, 3, lr=0.1),
    #                  SoftmaxLayer()
    # )

    # draw(module, 4)

    # loss = CategoricalCrossEntropy()
    # ec = 0.0
    # for i in range(200):
    #     x = [0.4, -1.0, 0.5, 0.2]
    #     y = module(x)
    #     ec += loss(y, [1.0,0.0,0.0], grad=True)
    #     out_grad = module(b=loss.grads)

    # print("Error: ", ec/200)

    # module = Module( Dense(4, 8, lr=0.1),
    #                  SigmoidLayer(8),
    #                  Dense(8, 1, lr=0.1),
    #                  TanhLayer(1)
    # )

    # loss = BinaryCrossEntropy()
    # eb = 0.0
    # for i in range(1000):
    #     x = [0.4, -1.0, 0.5, 0.2]
    #     y = module(x)
    #     eb += loss(y[0], 1.0, grad=True)
    #     out_grad = module(b=loss.grads)
        
    # print("Error: ", eb/1000)