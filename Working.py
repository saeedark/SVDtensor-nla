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


currentlocation = os.getcwd()

feeling = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']




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


def i2v(x,h=116,w=98): # change Problem
    return np.reshape(x,h*w)
def v2i(x,h,w):
    return np.reshape(x,(h,w))

isall = " yale "
dir = glob.glob(currentlocation + "/Face/3/*.pgm")
# if (len(sys.argv) > 1):
#     if ( sys.argv[1] == "1"):
#         isall = " all "
#         dir = glob.glob("~/Fac/jkbgale/1/*.pgm")

dir.sort()
imnpls = list(map(read_pgm,dir))
h,w = imnpls[0].shape
vp = list(map(i2v,imnpls))


#calculation


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

tlA =  pim[0].T
tlA = tlA[np.newaxis,:,:]
for i in range(1,len(pim)):
    v = pim[i].T
    v=v[np.newaxis,:,:]
    tlA = np.concatenate((tlA,v),axis=0)

#print(tlA.shape) #= (15 ya 17 , 11, 45045)

A = copy.deepcopy(tlA)


#  sudo echo 1 > /proc/sys/vm/overcommit_memory
#Memory problemsssssssssssss
#print(A.shape)
#u3 , u2 , u1 , s , vh1 , vh2 , vh3 = s.svd3(A)

# 1000 speed and calculation problemmmmsmsmmsmsmsm
#print(A.shape[2])
#core, factors = tucker(A, ranks=[A.shape[0], A.shape[1],11368])

core, factors = tucker(A)

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

"""

if (isall == " yale ") :
    pkl_file = open('sa/c.pkl', 'rb')
    core = pickle.load(pkl_file)
    pkl_file = open('sa/f0.pkl', 'rb')
    f0 = pickle.load(pkl_file)
    pkl_file = open('sa/f1.pkl', 'rb')
    f1 = pickle.load(pkl_file)
    pkl_file = open('sa/f2.pkl', 'rb')
    f2 = pickle.load(pkl_file)
    factors = [f0,f1,f2]

if (isall == " all ") :
    pkl_file = open('sa/ca.pkl', 'rb')
    core = pickle.load(pkl_file)
    pkl_file = open('sa/f0a.pkl', 'rb')
    f0 = pickle.load(pkl_file)
    pkl_file = open('sa/f1a.pkl', 'rb')
    f1 = pickle.load(pkl_file)
    pkl_file = open('sa/f2a.pkl', 'rb')
    f2 = pickle.load(pkl_file)
    factors = [f0,f1,f2]

"""

# showing eigenvalues
"""

#run with argv of system to get better
# in my mind What i find as I search I lose Hope and this code is Wrong for eigenvalue calculation u should laugh at me that how much stupid I Am
fold1 = tl.unfold(A,2)
u1 , s1, vh1 = np.linalg.svd(fold1 , full_matrices=False)
print(s1)
plt.figure("Image-mode" + isall)
plt.plot(list(s1))

fold1 = tl.unfold(A,0)
u1 , s1, vh1 = np.linalg.svd(fold1 , full_matrices=False)
print(s1)
plt.figure("person-mode" + isall)
plt.plot(list(s1))

fold1 = tl.unfold(A,1)
u1 , s1, vh1 = np.linalg.svd(fold1 , full_matrices=False)
print(s1)
plt.figure("expression-mode"+ isall)
plt.plot(list(s1))

plt.show()

"""
#end showing eigenvalues
#print(core.shape , f0.shape , f1.shape , f2.shape) #(15, 11, 11368) (15, 15) (11, 11) (11368, 11368)

"""
## make face part
D = tl.tenalg.mode_dot(core,f2,2)

for i in range(len(feeling)):
    print(i ,feeling[i])
e = int(input())
p = int(input("Select person in range 0 to 15+2 "))

v = f0[p,:]
v=v[:,np.newaxis]
D = tl.tenalg.mode_dot(D,v.T,0)

v = f1[e,:]
v=v[:,np.newaxis]
D = tl.tenalg.mode_dot(D,v.T,1)

plt.figure("subject " + str(p+1) + " " + feeling[e])
plt.imshow(v2i(list(D),h,w), plt.cm.gray)
plt.show()

##end  make face part
"""
