# This code will load the tensor core and factors
# and will get the picture
# and will find who was that picture

#  importing things
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

#Least Squre Problem Solver
def least(A,b):
    x, residuals, rank, s = np.linalg.lstsq(A,b)
    return x

# the Tensor that we want to load
#directorySelected = 'Face/1'
SelectDataBase = input("type 1 or 2 or 3 for input ")
directorySelected = 'Face/' + SelectDataBase

feeling = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
subject = ['subject01','subject02','subject03','subject04','subject05','subject06','subject07','subject08','subject09','subject10','subject11','subject12','subject13','subject14','subject15','subject16','subject17']
# defining for Frobenius norm for Ease of use
def twonorm(x):
    return np.linalg.norm(x,2)

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


# vector and matrix converter
h,w = 116,98
def i2v(x,h=116,w=98): # change Problem
    return np.reshape(x,h*w)
def v2i(x,h,w):
    return np.reshape(x,(h,w))


# loading the tensor
pkl_file = open(directorySelected + '/c.pkl', 'rb')
core = pickle.load(pkl_file)
pkl_file = open(directorySelected + '/f0.pkl', 'rb')
f0 = pickle.load(pkl_file)
pkl_file = open(directorySelected + '/f1.pkl', 'rb')
f1 = pickle.load(pkl_file)
pkl_file = open(directorySelected + '/f2.pkl', 'rb')
f2 = pickle.load(pkl_file)
factors = [f0,f1,f2]


# People , Face Expression , Picture  # is the way our tensor is

#make C for A = C *3 facetor[0] = People
C = copy.deepcopy(core)
C = tl.tenalg.mode_dot(C,factors[2],2)
C = tl.tenalg.mode_dot(C,factors[1],1)

samefeeling = 0
samePerson = 0
sameDetect = 0
for fff in range(len(feeling)):
    for sss in range(core.shape[0]): # len(subject)
        TestDirectory =  'Face/'+SelectDataBase+'/'+subject[sss]+'.'+feeling[fff]+'.pgm'
        # make vector of searching for thing
        Tim = read_pgm(TestDirectory)
        Tv = i2v(Tim)
        Tv = Tv[:,np.newaxis]


        # Search Algorithm for finding mach item
        outlist = []
        for i in range(core.shape[1]):
            Ce = C[:,i,:]
            Ce = Ce.T
            aa = least(Ce,Tv)
            for j in range(core.shape[0]):
                hh = factors[0]
                hh = hh.T
                hp = hh[:,j]
                hp = hp[:,np.newaxis]
                tttt = aa-hp
                outlist.append(twonorm(tttt))

        # Finding the min part of list
        minn= min(outlist)
        index =  outlist.index(minn)
        if (index//15 == fff):
            samefeeling += 1
        if (index%15 == sss):
            samePerson += 1
        if (index%15 == sss and index//15 == fff):
            sameDetect +=1
        print(index ,index//15,index%15,fff,sss) # feeling , subject

print("Same Feeling Detection " + str(samefeeling) + " /165")
print("Same person Detection " + str(samePerson) + " /165")
print("Same Correct Detection " + str(sameDetect) + " /165")
