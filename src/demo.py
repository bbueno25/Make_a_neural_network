from numpy import array
from numpy import dot
from numpy import exp
from numpy import random

class NeuralNetwork:
    """
    We model a single neuron, with 3 input connections and 1 output connection.
    We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1 and mean 0.
    """

    def __init__(self):
        """
        Seed the random number generator so it generates 
        the same numbers every time the program runs.
        """
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        """
        The Sigmoid function, which describes an S shaped curve.
        We pass the weighted sum of the inputs through this 
        function to normalise them between 0 and 1.
        """
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        """
        The derivative of the Sigmoid function.
        This is the gradient of the Sigmoid curve.
        It indicates how confident we are about the existing weight.
        """
        return x * (1 - x)

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        """
        We train the neural network through a process of trial and error.
        Adjusting the synaptic weights each time.
        """
        for _ in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights:")
    print(neural_network.synaptic_weights)

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training:")
    print(neural_network.synaptic_weights)

    print("Considering new situation [1, 0, 0] -> ?:")
    print(neural_network.think(array([1, 0, 0])))
