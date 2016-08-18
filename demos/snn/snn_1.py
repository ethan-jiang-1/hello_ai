#!/usr/bin/env python
# encoding:utf8


from numpy import exp, array, random, dot 

# A single neuron, with  3 inputs connections and 1 output connection.
# The weight is 3 x 1 matrix, with value in the range of -1 to 1 (mean is 0)

class NeuralNetwork():
	def __init__(self):
		
		#seed the random number generator, so it generates the same numbers every time
		random.seed(1)


		# a randome initial weight
		self.synaptic_weights = 2 * random.random((3,1)) -1


	#The Sigmoid function, whcih describes an S Shaped curve.
	# Normalise the value to 0 and 1  
	def __sigmoid(self,x):
		return 1 / (1 + exp(-x))


	#The dervivative of the Sigmoid function 
	#The is the gradient of the Sigmoid curve
	#It is indicates how confident we are about the e xisting weight
	def __sigmoid_derivative(self, x):
		return x * (1 -x)

	#the traning 
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):

			output = self.think(training_set_inputs)

			error = training_set_outputs - output 

			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			self.synaptic_weights += adjustment

	# the neural network thinks.
	def think(self,inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))




if __name__ == "__main__":

	nn = NeuralNetwork()

	print("Random starting synaptic_weights: ")
	print(str(nn.synaptic_weights))

	training_set_inputs  = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T

	nn.train(training_set_inputs,training_set_outputs,1000000)

	print("New synaptic_weights after training:")
	print(str(nn.synaptic_weights))

	print("Considering new siutation [1,0,0] -> ?: " + str(nn.think(array([1,0,0]))))