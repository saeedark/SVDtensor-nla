plt.figure(200)
image = read_pgm("Data/subject01.glasses.pgm", byteorder='<')
plt.imshow(image, plt.cm.gray)
#plt.show()
plt.figure(300)
image = read_pgm("Data/subject02.glasses.pgm", byteorder='<')
plt.imshow(image, plt.cm.gray)
plt.show()


"""
for i in range(tlA.shape[0]):
    for j in range(tlA.shape[1]):
        plt.figure("subject " + str(i+1) + " " + feeling[j])
        v = tlA[i,j,...]
        plt.imshow(v2i(v,h,w), plt.cm.gray)
        plt.show()
"""

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
