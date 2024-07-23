import numpy as np
from typing import Tuple

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""

def cost(y, t):
    return 0.5*(y-t)**2

def cost_prime(y, t):
    return -(y-t)

@np.vectorize
def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0, x)


@np.vectorize
def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    return int(x > 0)


@np.vectorize
def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x


@np.vectorize
def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1


class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):
        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.ones((3, 2), float)
        self.hidden_activation = rectified_linear_unit
        self.hidden_act_deriv = rectified_linear_unit_derivative
        self.hidden_to_output_weights = np.ones((1, 3), float)
        self.biases = np.zeros((3, 1), float)
        self.output_activation = output_layer_activation
        self.output_act_deriv = output_layer_activation_derivative
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [(np.array([[2],[1]]), 10),
                                (np.array([[3],[3]]), 21),
                                (np.array([[4],[5]]), 32),
                                (np.array([[6],[6]]), 42)]
        self.testing_points = [np.array([[1],[1]]),
                               np.array([[2],[2]]), np.array([[3],[3]]), np.array([[5],[5]]), np.array([[10],[10]])]


    def _forward_prop(self, input_values) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ### Forward propagation step for prediction and training ###
        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = self.input_to_hidden_weights @ input_values + self.biases
        hidden_layer_activation = self.hidden_activation(hidden_layer_weighted_input)

        output = np.dot(self.hidden_to_output_weights, hidden_layer_activation)
        activated_output = self.output_activation(output)

        return hidden_layer_weighted_input, hidden_layer_activation, output, activated_output

    def train(self, input_values, y):
        """
        Performs backpropagation gradient descent on a single input
        """
        # input_values = np.array([[x1], [x2]])  # 2 by 1
        hidden_layer_weighted_input, hidden_layer_activation, output, activated_output = self._forward_prop(input_values)

        ### Backpropagation ###

        # Compute gradients
        mat_inputs = np.repeat(input_values.T, self.input_to_hidden_weights.shape[0], axis=0)
        output_layer_error = (activated_output - y) * self.output_act_deriv(activated_output)  # 1x1
        hidden_layer_error = output_layer_error * self.hidden_to_output_weights # 1x3

        hidden_to_output_weight_gradients = output_layer_error * hidden_layer_activation.T  # 1x3
        bias_gradients = hidden_layer_error.T * self.hidden_act_deriv(hidden_layer_activation)  # 3x1
        input_to_hidden_weight_gradients = bias_gradients * mat_inputs  # 3x2

        # Use gradients to adjust weights and biases using gradient descent
        self.biases -= self.learning_rate * bias_gradients
        self.input_to_hidden_weights -= self.learning_rate * input_to_hidden_weight_gradients
        self.hidden_to_output_weights -= self.learning_rate * hidden_to_output_weight_gradients

    def predict(self, input_values):
        ### Returns predicted output
        # input_values = np.array([[x1], [x2]])
        _, _, _, activated_output = self._forward_prop(input_values)
        return activated_output.item()


    def train_neural_network(self):
        # Run this to train your neural network once you complete the train method
        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x, y)


    def test_neural_network(self):
        # Run this to test your neural network implementation for correctness after it is trained
        for point in self.testing_points:
            print("Point,", point.reshape(2), "Prediction,", self.predict(point))
            if abs(self.predict(point) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point.reshape(2), " failed to be predicted correctly.")
                return


def main():
    x = NeuralNetwork()
    x.train_neural_network()
    x.test_neural_network()


if __name__ == '__main__':
    main()