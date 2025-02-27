from abstractActivation import Activation
import math

class Linear(Activation):
    
    def forward(self, x):
        self.output = x
        return self.output
    
    def backward(self, x):
        return x
    
class Sigmoid(Activation):
    
    def forward(self, x):
        if x < 0:
            self.output = math.exp(x) / (1 + math.exp(x))
        self.output = 1 / (1 + math.exp(-x))
        return self.output
    
    def backward(self, x):
        return x * (self.output * (1.0 - self.output))
    
class Tanh(Activation):
    
    def forward(self, x):
        self.output = math.tanh(x)
        return self.output
    
    def backward(self, x):
        return x * (1.0 - self.output * self.output)
    
class ReLU(Activation):
    
    def forward(self, x):
        self.output = max(0.0, x)
        return self.output
    
    def backward(self, x):
        return x * ( 0.0 if self.output <= 0.0 else 1.0)
    
class LeakyReLU(Activation):
    
    def forward(self, x):
        self.output =  x if x > 0.0 else 0.01 * x
        return self.output
    
    def backward(self, x):
        return x * (1.0 if self.output > 0.0 else 0.01)
