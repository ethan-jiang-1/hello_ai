#!/usr/bin/env python
# encoding:utf8
#
# Reference: http://blog.csdn.net/zzukun/article/details/49968129




import copy
import numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
binary_dim = 8
largest_number = pow(2,binary_dim)
# input variables
alpha = 0.2
input_dim = 2
hidden_dim = 16
output_dim = 1


int2binary = {}
int2binary_init = False
def int2bin(num):
    global int2binary_init,int2binary
    if not int2binary_init:
        binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
        for i in range(largest_number):
            int2binary[i] = binary[i]
        int2binary_init = True
    return int2binary[num]

def bin2int(bin):
    out_int = 0
    for index,x in enumerate(reversed(bin)):
        out_int += x*pow(2,index)
    return out_int


def report_sum(a_int,b_int,out_int):
    result_lead = "x"
    result_msg = "result does not matched yet"
    if out_int == a_int + b_int:
        result_msg = "result matched"
        result_lead = "v"

    print("{0} result: {1} + {2} = {3} {4}".format(result_lead, a_int,b_int,out_int,result_msg))
    print("")


def report_result(j,overallError,d,c,a_int,b_int):
    print("------------")
    print("Loop:" + str(j))
    print("Error:" + str(overallError))
    print("Pred:" + str(d))
    print("True:" + str(c))
    out_int = bin2int(d)
    report_sum(a_int,b_int,out_int)

#flags = ""

def trained_plus(a_int,b_int,synapse_0,synapse_1,synapse_h):

    a = int2bin(a_int)
    b = int2bin(b_int)

    print("")
    print("Test: {0} + {1} with trained plus (multiple tmp layer1_value - we need last one only for predict)".format(a_int,b_int))
    print("  a: " + str(a))
    print("  b: " + str(b))

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(a)

    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    print("init")
    print("  layer_1(vlen): " + str(len(layer_1_values)))
    print("  layer_1(data): " + str(layer_1_values))
    print("  layer_2(bin): " + str(d))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):

        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))

        print("loop: " + str(position))
        print("  layer_0: " + str(X))
        print("  layer_1: " + str(layer_1))
        print("  layer_1(vlen): " + str(len(layer_1_values)))
        print("  layer_1(data): " + str(layer_1_values))
        print("  layer_2: " + str(layer_1))
        print("  layer_2(bin): " + str(d))

    out_int = bin2int(d)
    report_sum(a_int,b_int,out_int)
    return out_int


def trained_plus_one(a_int,b_int,synapse_0,synapse_1,synapse_h):

    a = int2bin(a_int)
    b = int2bin(b_int)

    print("")
    print("Test: {0} + {1} with trained plus (one tmp layer1_value)".format(a_int,b_int))
    print("  a: " + str(a))
    print("  b: " + str(b))

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(a)

    layer_1_values = np.zeros(hidden_dim)

    print("init")
    print("  layer_1: " + str(layer_1_values))
    print("  layer_2(bin): " + str(d))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):

        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values,synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        layer_1_values = copy.deepcopy(layer_1)

        print("loop: " + str(position))
        print("  layer_0: " + str(X))
        print("  layer_1: " + str(layer_1))
        print("  layer_1(data): " + str(layer_1_values))
        print("  layer_2: " + str(layer_1))
        print("  layer_2(bin): " + str(d))

    out_int = bin2int(d)
    report_sum(a_int,b_int,out_int)
    return out_int


def validate_trained_plus(synapse_0,synapse_1,synapse_h):
    print("##Validate trained plus:")
    print("synapse_0")
    print(str(synapse_0))
    print("synapse_1")
    print(str(synapse_1))
    print("synapse_h")
    print(str(synapse_h))
    print("")
    trained_plus(6,6,synapse_0,synapse_1,synapse_h)
    trained_plus(1,2,synapse_0,synapse_1,synapse_h)
    trained_plus(11,3,synapse_0,synapse_1,synapse_h)
    trained_plus(19,6,synapse_0,synapse_1,synapse_h)
    trained_plus(79,86,synapse_0,synapse_1,synapse_h)
    trained_plus(109,16,synapse_0,synapse_1,synapse_h)
    trained_plus(200,55,synapse_0,synapse_1,synapse_h)


    trained_plus_one(8,8,synapse_0,synapse_1,synapse_h)
    trained_plus_one(13,3,synapse_0,synapse_1,synapse_h)
    trained_plus_one(199,56,synapse_0,synapse_1,synapse_h)


def rnn_run():
    global binary_dim, largest_number
    global alpha, input_dim, hidden_dim, output_dim

    # initialize neural network weights
    synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
    synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
    synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

    synapse_0_update = np.zeros_like(synapse_0)
    synapse_1_update = np.zeros_like(synapse_1)
    synapse_h_update = np.zeros_like(synapse_h)

    # training logic
    print("##Training: 10000 times...")
    for j in range(10001):

        # generate a simple addition problem (a + b = c)
        a_int = np.random.randint(largest_number/2) # int version
        a = int2bin(a_int) # binary encoding

        b_int = np.random.randint(largest_number/2) # int version
        b = int2bin(b_int) # binary encoding

        # true answer
        c_int = a_int + b_int
        c = int2bin(c_int)

        # where we'll store our best guess (binary encoded)
        d = np.zeros_like(c)

        overallError = 0

        layer_2_deltas = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(hidden_dim))

        # moving along the positions in the binary encoding
        for position in range(binary_dim):

            # generate input and output
            X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
            y = np.array([[c[binary_dim - position - 1]]]).T

            # hidden layer (input ~+ prev_hidden)
            layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

            # output layer (new binary representation)
            layer_2 = sigmoid(np.dot(layer_1,synapse_1))

            # did we miss?... if so by how much?
            layer_2_error = y - layer_2
            layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
            overallError += np.abs(layer_2_error[0])

            # decode estimate so we can print it out
            d[binary_dim - position - 1] = np.round(layer_2[0][0])

            # store hidden layer so we can use it in the next timestep
            layer_1_values.append(copy.deepcopy(layer_1))

        future_layer_1_delta = np.zeros(hidden_dim)

        for position in range(binary_dim):

            X = np.array([[a[position],b[position]]])
            layer_1 = layer_1_values[-position-1]
            prev_layer_1 = layer_1_values[-position-2]

            # error at output layer
            layer_2_delta = layer_2_deltas[-position-1]
            # error at hidden layer
            layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + \
                layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
            # let's update all our weights so we can try again
            synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            synapse_0_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta


        synapse_0 += synapse_0_update * alpha
        synapse_1 += synapse_1_update * alpha
        synapse_h += synapse_h_update * alpha

        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0

        # print out progress
        if(j % 1000 == 0):
            report_result(j,overallError,d,c,a_int,b_int)

    print("")
    print("## Testing:...")
    validate_trained_plus(synapse_0,synapse_1,synapse_h)


rnn_run()