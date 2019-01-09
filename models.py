# Perceptron:
#     p = input_vector
#     w = weight_vector
#     b = bias
#     n = (p * w) + b
#     f = transfer_function
#     a = transfer_function(n)

# Layer:
#     p = input_vector
#     W = weight_matrix
#     b = bias_vector
#     n = (p * W) + B
#     f  transfer_function
#     a = transfer_function(n)

# Network:
#     p = vector_of_inputs
#     w = vector_of_layers
#     a = vector_of_outputs

# Delay Unit:
#     u(t) = input_vector
#     a(t) = u(t-1)
#     a(0) = starting_conditions_vector

# Integrator Unit:
#     u(t) = input_vector
#     a(t) = Intgral 0_t[ u(t)dt +a(0) ]
#     a(0) = starting_conditions_vector

import numpy as np

class Perceptron():
    #class
    @classmethod
    def with_bias(cls, input_size, transfer_function, bias):
        return cls(input_size, transfer_function, bias)

    @classmethod
    def default(cls, input_size, transfer_function):
        return cls(input_size, transfer_function, 1)

    #instance
    def __init__(self, input_size, transfer_function, bias):
        self.w = np.random.rand(input_size, 1)
        self.f = transfer_function
        self.b = bias

    def activate(self):
        self.n = np.dot(self.p, self.w) + self.b
        self.a = self.f(self.n)
        return self.a
    
    def activate_with(self, input_vector):
        self.p = input_vector
        return self.activate()

class Layer():
    #class
    @classmethod
    def default(cls, input_size, number_of_neurons, transfer_function):
        return cls(input_size, number_of_neurons, transfer_function, 1)

    #instance
    def __init__(self, input_size, number_of_neurons, transfer_function, bias_vector):
        self.W = np.random.rand(input_size, number_of_neurons)
        self.f = transfer_function
        self.b = bias_vector

    def activate(self):
        self.N = np.dot(self.p, self.W) + self.b
        self.a = self.f(self.N)
        return self.a

    def activate_with(self, input_vector):
        self.p = input_vector
        return self.activate()

class Stack_layer():
    #class
    @classmethod
    def default(cls, vector_of_layers):
        return cls(vector_of_layers)

    #instance
    def __init__(self, vector_of_layers):
        self.L = vector_of_layers

    def activate(self):
        outputs = []
        for layer in self.L:
            outputs.append(layer.activate_with(self.p))
        self.a = outputs[0]
        for index in range(1,len(outputs)):
            self.a = np.hstack((self.a, outputs[index]))
        return self.a

    def activate_with(self, input_vector):
        self.p = input_vector
        return self.activate()

class Forward_network():
    #class
    @classmethod
    def default(cls, vector_of_layers):
        return cls(vector_of_layers)

    #instance
    def __init__(self, vector_of_layers):
        self.l = vector_of_layers

    def activate(self):
        self.a = [self.l[0].activate_with(self.p[0])]
        self.p.append(self.a[0])
        for index in range(1, len(self.l)):
            a = self.l[index].activate_with(self.p[index])
            self.p.append(a)
            self.a.append(a)
        return self.a[-1]

    def activate_with(self, input_vector):
        self.p = [input_vector]
        return self.activate()

class Delay_unit():
    #class
    @classmethod
    def default(cls):
        return cls(None)

    def __init__(self, starting_conditions_vector):
        self.a = starting_conditions_vector

    def activate(self):
        if self.a is not None:
            a = self.a
        else:
            a = self.u
        self.a = self.u
        return a

    def activate_with(self, input_vector):
        self.u = input_vector
        return self.activate()

class Recurrent_network():
    #class
    @classmethod
    def default(cls, network):
        return cls(None, network)

    def __init__(self, delay, network):
        if delay is not None:
            self.delay = delay
        else:
            self.delay = Delay_unit.default()
        self.network = network

    def activate(self):
        p = self.delay.activate_with(self.p)
        self.a = self.network.activate_with(p)
        return self.a

    def activate_with(self, input_vector):
        self.p = input_vector
        self.activate()
        return self.activate_recursive()

    def activate_recursive(self):
        self.p = self.a
        return self.activate()