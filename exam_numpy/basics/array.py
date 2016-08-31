import numpy as np


def print_array(x, msg=None):
    if msg != None:
        print(msg)
    print(x)
    print("shape:" + str(x.shape))
    print("ndim:" + str(x.ndim))
    print("size:" + str(x.size))
    print("itemsize:" + str(x.itemsize))
    print("nbytes:" + str(x.nbytes))
    print("dtype:" + str(x.dtype))
    print("")


a1 = np.array([2,3,4])
print_array(a1)

a2 = np.array([2.0,1.0,0.0])
print_array(a2)


a3 = np.array([(1.5,2,3),(4,5,6)])
print_array(a3)


a4 = np.array([[2,3],[3,4]], dtype=complex)
print_array(a4)


s0 = np.zeros((3,4))
print_array(s0,"np.zeros")

s1 = np.ones((2,3,4))
print_array(s1,"np.ones")

s3 = np.empty((2,3))
print_array(s3,"np.empty")

#import pdb; pdb.set_trace()