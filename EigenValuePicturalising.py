# this code will show eigenvalue of our tensor
# for Tree Tensor we hvae
# 1 for Yale Date base
# 2 Reza is added to yale
# 3 Reza and me are Added to yale
# I should say that how they Are changing ans so On

# Importing some neccesery Imports
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import old as s
import pickle

# defining for Frobenius norm for Ease of use
def fnorm(x):
    return np.linalg.norm(x,'fro')

def mode1eigenValues(x):
    # notice that i is starting with Zero
    retlist = []
    tshape = x.shape
    l = tshape[0]
    for i in range(l):
        tmp = x[i,:,:]
        eig = fnorm(tmp)
        retlist.append(eig)
    retlist.sort(reverse = True)
    return retlist

def mode2eigenValues(x):
    # notice that i is starting with Zero
    retlist = []
    tshape = x.shape
    l = tshape[1]
    for i in range(l):
        tmp = x[:,i,:]
        eig = fnorm(tmp)
        retlist.append(eig)
    retlist.sort(reverse = True)
    return retlist

def mode3eigenValues(x):
    # notice that i is starting with Zero
    retlist = []
    tshape = x.shape
    l = tshape[2]
    for i in range(l):
        tmp = x[:,:,i]
        eig = fnorm(tmp)
        retlist.append(eig)
    retlist.sort(reverse = True)
    return retlist

#This next three part will load numpy Arrays of them
pkl_file = open('Face/1/c.pkl', 'rb')
core1 = pickle.load(pkl_file)
pkl_file = open('Face/1/f0.pkl', 'rb')
f01 = pickle.load(pkl_file)
pkl_file = open('Face/1/f1.pkl', 'rb')
f11 = pickle.load(pkl_file)
pkl_file = open('Face/1/f2.pkl', 'rb')
f21 = pickle.load(pkl_file)
factors1 = [f01,f11,f21]

pkl_file = open('Face/2/c.pkl', 'rb')
core2 = pickle.load(pkl_file)
pkl_file = open('Face/2/f0.pkl', 'rb')
f02 = pickle.load(pkl_file)
pkl_file = open('Face/2/f1.pkl', 'rb')
f12 = pickle.load(pkl_file)
pkl_file = open('Face/2/f2.pkl', 'rb')
f22 = pickle.load(pkl_file)
factors2 = [f01,f11,f21]

pkl_file = open('Face/3/c.pkl', 'rb')
core3 = pickle.load(pkl_file)
pkl_file = open('Face/3/f0.pkl', 'rb')
f03 = pickle.load(pkl_file)
pkl_file = open('Face/3/f1.pkl', 'rb')
f13 = pickle.load(pkl_file)
pkl_file = open('Face/3/f2.pkl', 'rb')
f23 = pickle.load(pkl_file)
factors3 = [f01,f11,f21]


# next 9 line will get Eigenvlues of modes for 3 case
e11 = list(mode1eigenValues(core1))
e12 = list(mode2eigenValues(core1))
e13 = list(mode3eigenValues(core1))

e21 = list(mode1eigenValues(core2))
e22 = list(mode2eigenValues(core2))
e23 = list(mode3eigenValues(core2))

e31 = list(mode1eigenValues(core3))
e32 = list(mode2eigenValues(core3))
e33 = list(mode3eigenValues(core3))


# Showing plot
plt.figure("Mode 1 EigenValues (People)")
plt.plot(e11 , label='Yale')
plt.plot(e21 , label='Yale+Reza')
plt.plot(e31 , label='Yale+Reza+me')
plt.legend()

plt.figure("Mode 2 EigenValues (Face expression)")
plt.plot(e12 , label='Yale')
plt.plot(e22 , label='Yale+Reza')
plt.plot(e32 , label='Yale+Reza+me')
plt.legend()

plt.figure("Mode 3 EigenValues (Picture)")
plt.plot(e13 , label='Yale')
plt.plot(e23 , label='Yale+Reza')
plt.plot(e33 , label='Yale+Reza+me')
plt.legend()

plt.show()

#See how much differs new data on Eigenvlues
