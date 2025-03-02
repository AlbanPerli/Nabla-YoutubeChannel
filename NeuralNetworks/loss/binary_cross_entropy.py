
import math
from loss.abstractLoss import LossFunction

class BinaryCrossEntropy(LossFunction):
    
    def forward(self, y_true, y_pred, grad=False):
        assert type(y_true) == float or type(y_true) == int
        assert type(y_pred) == float or type(y_pred) == int
            
        if grad:
            self.grads = self.gradients(y_true, y_pred)
            
        y_pred = max(min(y_pred, 1 - self.EPSILON), self.EPSILON)
        return - (y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

    def gradients(self, y_true, y_pred):
        assert type(y_true) == float or type(y_true) == int
        assert type(y_pred) == float or type(y_pred) == int

        grads = []
        grads.append( y_pred - y_true )
        return grads