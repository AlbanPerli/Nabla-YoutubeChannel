import math
from loss.abstractLoss import LossFunction

class CategoricalCrossEntropy(LossFunction):
    
    def forward(self, y_true, y_pred, grad=False):
        if type(y_true) == float:
            y_true = [y_true]
        if type(y_pred) == float:
            y_pred = [y_pred]
        assert len(y_true) == len(y_pred)
        
        if grad:
            self.grads = self.gradients(y_true, y_pred)
                
        sum = 0
        for y, y_pred in zip(y_true, y_pred):
            sum = y * math.log(y_pred + self.EPSILON)
        return -sum
    
    def gradients(self, y_true, y_pred):
        if type(y_true) == float:
            y_true = [y_true]
        if type(y_pred) == float:
            y_pred = [y_pred]  
            
        assert len(y_true) == len(y_pred)
        grads = []
        n = len(y_true)
        for y, y_pred in zip(y_true, y_pred):
            grads.append( y_pred - y )
        return grads