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

if __name__ == '__main__':
    unittest.main()