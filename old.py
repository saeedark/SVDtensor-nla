import numpy as np
import tensorly as tl
import copy
from tensorly.decomposition import tucker


def on(x):
    out = "["
    row = x.shape[0]
    col = x.shape[1]
    for i in range(0,int(row)):
        strr="["
        for k in range(0,int(col)):
            strr+=str(x[i,k])+" "
        strr+="]\n"
        out+=strr
    return out


A=np.zeros((3,3,3,3), dtype=int)
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                A.itemset((i,j,k,l),(i+1)*1000+(j+1)*100+(k+1)*10+l+1)

#example A for ijkl number for indexes for foldiing and unfolding
#print(A)

B=np.zeros((3,3,3), dtype=int)
for i in range(3):
    for j in range(3):
        for k in range(3):
            B.itemset((i,j,k),(i+1)*100+(j+1)*10+k+1)

#3 by 3 by 3 tensor
#print(B)

C=np.random.rand(3,3,3)

movie = np.zeros((3,3,3), dtype=int)
#index = person , movie , day[2,3,fri]
movie.itemset((0,2,2),1)
movie.itemset((0,0,0),1)
movie.itemset((0,1,1),1)
movie.itemset((1,0,1),1)
movie.itemset((1,0,0),1)
movie.itemset((1,1,1),1)
movie.itemset((1,1,0),1)
movie.itemset((2,1,2),1)
movie.itemset((2,0,1),1)
movie.itemset((2,2,0),1)
#unfolding Example
"""
print(A)
x=tl.unfold(A,3)
#print(x)
print(on(x))
"""

"""
sh= A.shape
p = tl.unfold(A,2)
t = tl.fold(p,2,sh)

print(t)
"""



#HOSVD with tensorly tucker
"""
core, factors = tucker(A, ranks=[3, 3, 3, 3])
print(core)
print(factors)

print("_________________________")

x = tl.tenalg.mode_dot(core,factors[0],0)
for i in range(1,len(factors)):
    x = tl.tenalg.mode_dot(x,factors[i],i)
print(x)
"""

#HOSVD for 3 dim
def svd3(x):
    f1 = tl.unfold(x,0)
    f2 = tl.unfold(x,1)
    f3 = tl.unfold(x,2)
    u1 , s1, vh1 = np.linalg.svd(f1, full_matrices=True)
    u2 , s2, vh2 = np.linalg.svd(f2, full_matrices=True)
    u3 , s3, vh3 = np.linalg.svd(f3, full_matrices=True)
    x = tl.tenalg.mode_dot(x,u1.transpose(),0)
    x = tl.tenalg.mode_dot(x,u2.transpose(),1)
    x = tl.tenalg.mode_dot(x,u3.transpose(),2)

    return u3 , u2 , u1 , x , vh1 , vh2 , vh3

"""
u3 , u2 , u1 , s , vh1 , vh2 , vh3 = svd3(B)
a = tl.tenalg.mode_dot(s,u1,0)
a = tl.tenalg.mode_dot(a,u2,1)
a = tl.tenalg.mode_dot(a,u3,2)
"""
"""
print(a)
"""

"""
print(movie)
u3 , u2 , u1 , s , vh1 , vh2 , vh3 = svd3(movie)
print(s)
"""

#Approx

def  approx3(s, u1 , u2 , u3 , m=3):
    x = copy.deepcopy(s)
    lss = []
    for i in range(3):
        t = np.zeros((3,3))
        for j in range(3):
            for k in range(3):
                t.itemset((j,k), x[j,k,i])

        ap = tl.tenalg.mode_dot(t,u1,0)
        ap = tl.tenalg.mode_dot(ap,u2,1)
        #ap = np.dot(t,u1)
        #ap = np.dot(u2,ap)
        #print(ap)
        #ap =  np.array([ copy.deepcopy(ap),copy.deepcopy(ap) ,copy.deepcopy(ap) ])
        #print(ap)

        #ap = tl.tenalg.mode_dot(ap,u3[:,i],2)
        #print(ap)
        ap = np.outer(ap,u3[:,i])
        ap = ap.reshape((3,3,3))
        lss.append(ap)
    #print(lss)
    return(lss)


#it's not working and it's weird

"""
print(movie)
u3 , u2 , u1 , s , vh1 , vh2 , vh3 = svd3(movie)
x = approx3(s,u1,u2,u3,3)
print(x[0]+x[1]+x[2])
"""
