from models import Perceptron
from models import Layer
from models import Stack_layer
from models import Forward_network
from models import Recurrent_network
import numpy as np

transfer_function = np.vectorize(lambda a: a )

default_neuron = Perceptron.default(2, transfer_function)

value = default_neuron.activate_with(np.random.rand(1, 2))

print(value)

default_layer = Layer.default(4, 2, transfer_function)

value = default_layer.activate_with(np.random.rand(1, 4))

print(value)

first_default_layer = Layer.default(4, 2, transfer_function)

second_default_layer = Layer.default(4, 3, transfer_function)

default_stack_layer = Stack_layer.default([first_default_layer, second_default_layer])

value = default_stack_layer.activate_with(np.random.rand(1, 4))

print(value)

first_default_layer = Layer.default(4, 3, transfer_function)

second_default_layer = Layer.default(3, 2, transfer_function)

default_network = Forward_network.default([first_default_layer, second_default_layer])

value = default_network.activate_with(np.random.rand(1, 4))

print(value)

first_default_layer = Layer.default(2, 5, transfer_function)

second_default_layer = Layer.default(5, 2, transfer_function)

default_network = Forward_network.default([first_default_layer, second_default_layer])

default_recurrent_network = Recurrent_network.default(default_network)

value = default_recurrent_network.activate_with(np.random.rand(1, 2))

value = default_recurrent_network.activate_recursive()
value = default_recurrent_network.activate_recursive()
value = default_recurrent_network.activate_recursive()
value = default_recurrent_network.activate_recursive()
value = default_recurrent_network.activate_recursive()
value = default_recurrent_network.activate_recursive()
value = default_recurrent_network.activate_recursive()

print(value)