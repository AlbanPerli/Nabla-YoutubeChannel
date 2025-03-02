from layers.abstractLayer import Layer


class Parallel(Layer):
    
    def __init__(self, *args):
        self.layers = args
        self.struct_outputs = []
        self.flatten_outputs = []

    def forward(self, x):
        self.struct_outputs = []
        self.flatten_outputs = []
        for layer in self.layers:
            self.struct_outputs.append(layer(x))
        for o in self.struct_outputs:
            if isinstance(o, list):
                self.flatten_outputs.extend(o)
            else:
                self.flatten_outputs.append(o)
        return self.flatten_outputs
    
    def backward(self, output_grad):
        if len(output_grad) != len(self.flatten_outputs):
            raise ValueError("output_grad: {}, is not the same size as the number of outputs:{}".format(len(output_grad), len(self.flatten_outputs)))
        struct_output_sizes = [len(o) if isinstance(o, list) else 1 for o in self.struct_outputs]
        output_grads = []
        for size in struct_output_sizes:
            output_grads.append(output_grad[:size])
            output_grad = output_grad[size:]
        o = [layer(b=g) for layer, g in zip(self.layers, output_grad)]
        return o