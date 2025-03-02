from layers.abstractLayer import Layer


class Chunk1DLayer(Layer):
    
    def __init__(self, chunk_size, step_size, keep_dim=True):
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.keep_size = keep_dim
        self.outputs = []
        self.chunked_input = []
        
    def forward(self, x):
        assert isinstance(x, list), "x is not a list"
        assert len(x) > 0, "x is empty"
        
        original_length = len(x)

        # Si keep_size est activé, on ajoute du padding pour s'assurer d'avoir autant de sorties que l'entrée
        if self.keep_size:
            needed_length = (original_length - 1) * self.step_size + self.chunk_size
            pad_size = max(0, needed_length - original_length)
            x = [0] * (pad_size // 2) + x + [0] * (pad_size - pad_size // 2)

        # Création des fenêtres
        self.chunked_input = [x[i:i+self.chunk_size] for i in range(0, len(x) - self.chunk_size + 1, self.step_size)]
        
        # S'assurer que la sortie contient bien `original_length` chunks
        while len(self.chunked_input) < original_length:
            self.chunked_input.append([0] * self.chunk_size)

        self.outputs = self.chunked_input
        return self.outputs
                    
    def backward(self, output_grad):
        return output_grad
    
class Chunk2DLayer(Layer):
    
    def __init__(self, width, height, center_padd):
        self.width = width
        self.height = height
        self.center_padd = center_padd
        self.outputs = []
        self.chunked_input = []
        
    def forward(self, x):
        assert type(x) is list, "x is not a list".format(x)
        assert len(x) > 0, "x is empty".format(x)
        assert type(x[0]) is list, "x[0] must be a list"
        
        if self.center_padd:
            x = [[0] + i + [0] for i in x]
            x = [[0] * len(x[0])] + x + [[0] * len(x[0])]
        
        self.chunked_input = []
        for i in range(0, len(x)-self.height+1, self.height):
            for j in range(0, len(x[0])-self.width+1, self.width):
                self.chunked_input.append([x[k][j:j+self.width] for k in range(i, i+self.height)])
        self.outputs = self.chunked_input
        return self.outputs
    
    def backward(self, output_grad):
        return output_grad