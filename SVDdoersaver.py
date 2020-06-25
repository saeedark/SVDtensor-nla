# Written in timeless time so sorry for any wrongness
# this code will find any picture in location near you file + Face/1/*.pgm
# making them vector and for any subject make matrix
# after that this will make tensor by concatenating subjects

# usig tensorly.decomposition.Tucker to run High Order SVD
# this takes times
# so we will use pickle to save them as file and then get them back


# some import to be fully runnable
import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import copy
from tensorly.decomposition import tucker
import old as s
import pickle
import sys


# detect our current location
currentlocation = os.getcwd()


# the function which convert picture to numpy matrix
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))



# this simple function make our numpy image a vector
def i2v(x,h=116,w=98): # change Problem
    return np.reshape(x,h*w)

# and  this function just revert that
def v2i(x,h,w):
    return np.reshape(x,(h,w))


# will give all pgm exit in Directory which is Argument
dir = glob.glob(currentlocation + "/Face/3/*.pgm")

# will do the sort and make the matrix of our Directory
dir.sort()
imnpls = list(map(read_pgm,dir))
h,w = imnpls[0].shape
vp = list(map(i2v,imnpls))
pim = []
for i in range(len(vp)//11):
    matrix = vp[i*11].T
    matrix=matrix[:,np.newaxis]
    for j in range(1,11):
        v=vp[(i)*11+j]
        v=v[:,np.newaxis]
        matrix=np.concatenate((matrix, v),axis =1)
    #print(matrix.shape)
    pim.append(matrix)
# and this part of code make Tensor of our pictures
tlA =  pim[0].T
tlA = tlA[np.newaxis,:,:]
for i in range(1,len(pim)):
    v = pim[i].T
    v=v[np.newaxis,:,:]
    tlA = np.concatenate((tlA,v),axis=0)

#print(tlA.shape) # expected (15 ya 16 ya 17 , 11, 45045)

# let's have Another Copy of our data
A = copy.deepcopy(tlA)


# Do the Higher Order SVD
# and Oh My Gooooood this Takes time
core, factors = tucker(A)


# And for that Reason We will Save them as pkl file ;)
output = open('c.pkl', 'wb')
pickle.dump(core, output)
output.close()

output = open('f0.pkl', 'wb')
pickle.dump(factors[0], output)
output.close()

output = open('f1.pkl', 'wb')
pickle.dump(factors[1], output)
output.close()

output = open('f2.pkl', 'wb')
pickle.dump(factors[2], output)
output.close()
