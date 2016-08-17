#
# Reference: http://mp.weixin.qq.com/s?__biz=MjM5NzU0MzU0Nw==&mid=2651371472&idx=1&sn=d9150993a80fc0e206becd9f2abd3375&scene=1&srcid=0817XVLGOMnUNU9jod6cc86q#wechat_redirect

from numpy import exp, array, random, dot
import pdb

random.seed(1)

#existing traning inputs (4 sets of 3 vectors ) amd output (4 set of 1 vector)


training_set_inputs = array([[0,0,1],
                             [1,1,1],
                             [1,0,1],
                             [0,1,1]])

training_set_outputs = array([[0,1,1,0]]).T  # i.e [[0],[1],[1],[0]]


synaptic_weights = 2 * random.random((3,1)) -1   # an weights of 3 vectors [[r1],[r2],[r3]]




for iteration in xrange(10000):

    output = 1 / (1 +  exp(-(dot(training_set_inputs, synaptic_weights))))

    adjust_weights = dot(training_set_inputs.T, (training_set_outputs - output) * output * (1-output))
    synaptic_weights += adjust_weights

    result = 1 / (1 + exp(-(dot(array([1,0,0]),synaptic_weights))))

    line = "iteration: " + str(iteration) + "\n"
    line += "" + str(synaptic_weights) + "\n"
    line += "" + str(result) + "\n"

    print(line)
    #print(result)