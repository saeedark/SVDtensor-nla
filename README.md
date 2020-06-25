# SVDtensor-nla
Implementation of tensor SVD decomposition 

book : Matrix Methods in Data Mining and Pattern Recognition 

Classification algorithm (preliminary version)

'''
z is a test image.

for e = 1,2,...,n_e

  solve min_(a_e) ||C_e a_e - z ||_2
  
  for p = 1,2,...,n_p
  
    if ||a_e - h_p||_2 < tol, then classify as person p and stop.
    
    end
    
 end
 
'''
