#!/usr/bin/env python
# encoding:utf8

#https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a#.k4its0tsf

from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers 
        self.end_layer = len(layers)   # inputs -> layer 1 -> layer 2 -> ..... -> layer N (the output)

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_layers = self.think(training_set_inputs)

            layer_deltas = {}
            for i in range(self.end_layer, 0, -1):  
                error = None  
                delta = None
                adjustment = None 
                if i == self.end_layer:
                    error = training_set_outputs - output_layers[self.end_layer]
                    delta = error * self.__sigmoid_derivative(output_layers[i]) 
                    adjustment = output_layers[i-1].T.dot(delta)
                elif i == 1 : 
                    error = layer_deltas[i+1].dot(self.layers[i+1].synaptic_weights.T)
                    delta = error * self.__sigmoid_derivative(output_layers[i]) 
                    adjustment = training_set_inputs.T.dot(delta)
                else:
                    error = layer_deltas[i+1].dot(self.layers[i+1].synaptic_weights.T)
                    delta = error * self.__sigmoid_derivative(output_layers[i]) 
                    adjustment = output_layers[i-1].T.dot(delta)

                layer_deltas[i] = delta

                self.layers[i].synaptic_weights += adjustment


    # The neural network thinks.
    def think(self, inputs):
        outputs = {} 
        for i in range(1, self.end_layer+1):  
            if i == 1: 
                output = self.__sigmoid(dot(inputs,self.layers[i].synaptic_weights))
            else:
                output = self.__sigmoid(dot(outputs[i-1],self.layers[i].synaptic_weights))
            outputs[i] = output
        return outputs 

    # The neural network prints its weights
    def print_weights(self):
        for i in range(1, self.end_layer+1):  
            layer = self.layers[i]
            print("  Layer {0} ({1} neurons, each with {2} inputs):".format(i,layer.number_of_neurons, layer.number_of_inputs_per_neuron)) 
            print(layer.synaptic_weights)       

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)


    layers = {} 
    layers[1] = NeuronLayer(4, 3)   #the first layer -- 4 neurons 3 inputs
    layers[2] = NeuronLayer(3, 4)
    layers[3] = NeuronLayer(2, 3) 
    layers[4] = NeuronLayer(1, 2)   #the last one layer -- the output, 1 neourons, 6 inputs  


    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layers)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("")
    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("")
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    outputs = neural_network.think(array([1, 1, 0]))
    print(outputs[len(layers)])
