from loss.abstractLoss import LossFunction

class MAE(LossFunction):
    
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
            sum = abs(y - y_pred)
        return sum / len(y_true)

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