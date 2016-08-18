#!/usr/bin/env python
# encoding:utf8

#
# Reference: http://mp.weixin.qq.com/s?__biz=MjM5NzU0MzU0Nw==&mid=2651371472&idx=1&sn=d9150993a80fc0e206becd9f2abd3375&scene=1&srcid=0817XVLGOMnUNU9jod6cc86q#wechat_redirect

from numpy import exp, array, random, dot
import pdb


#existing traning inputs (4 sets of 3 vectors ) amd output (4 set of 1 vector)
training_set_inputs = array([[0,0,1],
                             [1,1,1],
                             [1,0,1],
                             [0,1,1]])

training_set_outputs = array([[0,1,1,0]]).T  # i.e [[0],[1],[1],[0]]

#new set of input and expecting output...
new_set_input = array([1,0,0])  # the new inputs...and we may expect the output is [1]

new_set_output = array([1])  # we are expection this result

bool_print_training = False

def gml(obj):
    strobj = str(obj)
    line = strobj.replace("\n","\t")
    return line


def print_details_core(inputs,weights, outputs, msg):
    line = msg + "\n"
    line += "inputs:  \t" + gml(inputs) + "\n"
    line += "weights: \t" + gml(weights) + "\n"
    line += "outputs: \t" + gml(outputs) + "\n"
    print(line)


def print_details_training(inputs,weights,outputs,iteration_num,msg):
    if bool_print_training:
        msg_all = "iteration_" + str(iteration_num) + " :" + msg
        print_details_core(inputs,weights,outputs,msg_all)


def print_details_new(inputs,weights,outputs,msg):
    msg_all = "new set input/output 100: " + msg
    print_details_core(inputs,weights,outputs,msg_all)



def get_outputs(inputs,weights):
    return 1 / (1 + exp(-(dot(inputs, weights))))


def training(iteration_max):

    synaptic_weights = 2 * random.random((3,1)) -1   # an weights of 3 vectors [[r1],[r2],[r3]] - random at beginging

    #the output for new_set_input ([1,0,0]) - initialized
    new_temp_output  = get_outputs(new_set_input, synaptic_weights)
    print_details_new(new_set_input,synaptic_weights,new_temp_output,"The initial:")


    print_details_training(training_set_inputs,synaptic_weights,training_set_outputs,0,"initialization: -- try to change weight to get expected outputs like training_set_outputs")

    for iteration in xrange(iteration_max):

        #the output based on current weights on known traning_set_input
        tmp_outputs = get_outputs(training_set_inputs, synaptic_weights)

        #Adjust the weights
        adjust_weights = dot(training_set_inputs.T, (training_set_outputs - tmp_outputs) * tmp_outputs * (1 - tmp_outputs))
        synaptic_weights += adjust_weights

        print_details_training(training_set_inputs,synaptic_weights,tmp_outputs,iteration,"")

    #the output for new_set_input ([1,0,0]) - the finial
    new_temp_output  = get_outputs(new_set_input, synaptic_weights)
    print_details_new(new_set_input,synaptic_weights,new_temp_output,"The final (after {0} times training):".format(iteration))




def main():
    random.seed(1)
    training(100000)

main()