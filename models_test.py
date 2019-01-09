from models import Perceptron
from models import Layer
from models import Stack_layer
from models import Forward_network
from models import Recurrent_network
import numpy as np
import unittest

class Basic_models_test(unittest.TestCase):

    def setUp(self):
        self.linear_function = np.vectorize(lambda a: a )

    def single_perceptron_test(self):
        default_perceptron = Perceptron.default(2, self.linear_function)
        value = default_perceptron.activate_with(np.ones(1, 2))
        self.assertTrue(default_perceptron.get_weights() <= 1)

    def single_layer_test(self):
        default_layer = Layer.default(4, 2, self.linear_function)
        value = default_layer.activate_with(np.ones(1, 4))

    def stack_layer_test(self):
        first_default_layer = Layer.default(4, 2, self.linear_function)
        second_default_layer = Layer.default(4, 3, self.linear_function)
        default_stack_layer = Stack_layer.default([first_default_layer, second_default_layer])
        value = default_stack_layer.activate_with(np.ones(1, 4))

    def feed_forward_network_test(self):
        first_default_layer = Layer.default(4, 3, self.linear_function)
        second_default_layer = Layer.default(3, 2, self.linear_function)
        default_network = Forward_network.default([first_default_layer, second_default_layer])
        value = default_network.activate_with(np.ones(1, 4))

    def recurrent_network_test(self):
        first_default_layer = Layer.default(2, 5, self.linear_function)
        second_default_layer = Layer.default(5, 2, self.linear_function)
        default_network = Forward_network.default([first_default_layer, second_default_layer])
        default_recurrent_network = Recurrent_network.default(default_network)
        value = default_recurrent_network.activate_with(np.ones(1, 2))
        value = default_recurrent_network.activate_recursive()
        value = default_recurrent_network.activate_recursive()
        value = default_recurrent_network.activate_recursive()
        value = default_recurrent_network.activate_recursive()
        value = default_recurrent_network.activate_recursive()
        value = default_recurrent_network.activate_recursive()
        value = default_recurrent_network.activate_recursive()

if __name__ == '__main__':
    unittest.main()