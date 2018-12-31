from models import Perceptron
from models import Layer
from models import Stack_layer
from models import Forward_network
from models import Recurrent_network
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.linear_function = np.vectorize(lambda a: a )

    def test_single_perceptron(self):
        default_perceptron = Perceptron.default(2, self.linear_function)
        value = default_perceptron.activate_with(np.random.rand(1, 2))

    def test_single_layer(self):
        default_layer = Layer.default(4, 2, self.linear_function)
        value = default_layer.activate_with(np.random.rand(1, 4))

    def test_stack_layer(self):
        first_default_layer = Layer.default(4, 2, self.linear_function)
        second_default_layer = Layer.default(4, 3, self.linear_function)
        default_stack_layer = Stack_layer.default([first_default_layer, second_default_layer])
        value = default_stack_layer.activate_with(np.random.rand(1, 4))

    def test_feed_forward_network(self):
        first_default_layer = Layer.default(4, 3, self.linear_function)
        second_default_layer = Layer.default(3, 2, self.linear_function)
        default_network = Forward_network.default([first_default_layer, second_default_layer])
        value = default_network.activate_with(np.random.rand(1, 4))

    def test_recurrent_network(self):
        first_default_layer = Layer.default(2, 5, self.linear_function)
        second_default_layer = Layer.default(5, 2, self.linear_function)
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

if __name__ == '__main__':
    unittest.main()