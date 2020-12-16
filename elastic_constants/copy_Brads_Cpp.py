import numpy as np
from scipy import linalg as LA
from time import time

# highest order polynomial
order = 3

# mass in kg
m = 0.045e-3

# dimensions of sample in m
Lx = 0.145e-2 / 2
Ly = 0.201e-2 / 2
Lz = 0.302e-2 / 2

# density in kg/m^3
density = m / (8*Lx*Ly*Lz)


# number of basis functions
R = int( 3 * (order+1) * (order+2) * (order+3) / 6 )



# elastic constants in GPa
c11 = 321.61990e9
c22 = c11
c33 = c11
c12 = 103.50101e9
c13 = c12
c23 = c12
c44 = 124.99627e9
c55 = c44
c66 = c44



ctens = np.zeros([3,3,3,3])
ctens[0,0,0,0] = c11
ctens[1,1,1,1] = c22
ctens[2,2,2,2] = c33
ctens[0,0,1,1] = ctens[1,1,0,0] = c12
ctens[2,2,0,0] = ctens[0,0,2,2] = c13
ctens[1,1,2,2] = ctens[2,2,1,1] = c23
ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = c44
ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = c55
ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = c66





# initialize set of basis functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# calculates the parity of the basis functions
# parity is categorized into 8 different groups
basisPop = np.zeros(8) # stores how many basis functions are in each group

# define a function to determine the parity of a basis function
def parity (k, l, m, i):
    exponents = np.array([k,l,m])
    pvec = (-1)**exponents
    pvec[i] = (-1) ** (exponents[i]+1)

    if np.all(pvec == np.array([1, 1, 1])) == True:
        basisPop[0] = basisPop[0] + 1
        return (0)

    elif np.all(pvec == np.array([1, 1, -1])) == True:
        basisPop[1] = basisPop[1] + 1
        return (1)
    
    elif np.all(pvec == np.array([1, -1, 1])) == True:
        basisPop[2] = basisPop[2] + 1
        return (2)
    
    elif np.all(pvec == np.array([1, -1, -1])) == True:
        basisPop[3] = basisPop[3] + 1
        return (3)

    elif np.all(pvec == np.array([-1, 1, 1])) == True:
        basisPop[4] = basisPop[4] + 1
        return (4)

    elif np.all(pvec == np.array([-1, 1, -1])) == True:
        basisPop[5] = basisPop[5] + 1
        return (5)

    elif np.all(pvec == np.array([-1, -1, 1])) == True:
        basisPop[6] = basisPop[6] + 1
        return (6)
	
    elif np.all(pvec == np.array([-1, -1, -1])) == True:
        basisPop[7] = basisPop[7] + 1
        return (7)


# basisfunctions = np.zeros([R, 5]) # the first element gives the coordinate, i.e. 0 for x, 1 for y, 2 for z
basisCategories = [ [], [], [], [], [], [], [], [] ]

# basisPoint = 0
for k in np.arange(order+1):
    for l in np.arange(order+1):
        for m in np.arange(order+1):
            if k+l+m <= order:
                for i in np.arange(3):
                    basisCategories[parity(k,l,m,i)].append([i, k, l, m])
                    # basisfunctions[basisPoint+i, 0] = int(i) # tells you whether this is for x (0), y (1), or z (2) coordinate
                    # basisfunctions[basisPoint+i, 1] = k # gives exponent for x-coordinate
                    # basisfunctions[basisPoint+i, 2] = l # gives exponent for y-coordinate
                    # basisfunctions[basisPoint+i, 3] = m # gives exponent for z-coordinate
                    # basisfunctions[basisPoint+i, 4] = parity(k, l, m, i) # gives exponent for z-coordinate
                # basisPoint = basisPoint + 3    

basisPop = [int(N) for N in basisPop]
basisfunctions = {}
index = 0
for bN in np.arange(8):
    for i in np.arange(basisPop[bN]):
        basisfunctions[index] = basisCategories[bN][i]
        index += 1



# calculate energy matrices >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# calculate kinetic energy matrix >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# first I define a function which integrates the product of two basis functions
def integrateBasis (b1, b2, Lx, Ly, Lz): 
    # b1, b2 are elements of the basisfunctions array
    # Lx, Ly, Lz are dimensions of the sample
    exponent_x = b1[1] + b2[1]
    exponent_y = b1[2] + b2[2]
    exponent_z = b1[3] + b2[3]

    if (exponent_x%2 == 1) or (exponent_y%2 == 1) or (exponent_z%2 == 1):
        intVal = 0
    else:
        intVal = 8 * ( 1/(exponent_x+1) * Lx**(exponent_x+1) ) * ( 1/(exponent_y+1) * Ly**(exponent_y+1) ) * ( 1/(exponent_z+1) * Lz**(exponent_z+1) )

    return (intVal)


Emat = np.zeros([R, R])

for i in np.arange(R):
    for j in np.arange(i, R): # this is a symmetric matrix, jsut calculating the upper part should be enough
        b1 = basisfunctions[i]
        b2 = basisfunctions[j]
        if b1[0] == b2[0]:
            Emat[i, j] = density * integrateBasis(b1, b2, Lx, Ly, Lz)
            Emat[j, i] = Emat[i, j]
            


# calcualte potential energy matrix >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# first I define a function which integrates the product of the derivative of two basis functions
def integrateGradBasis (b1, b2, d1, d2, Lx, Ly, Lz):
    # d1, d2 defines with respect to which coordinate the derivative will be taken in b1, b2, i.e. 0 for d/dx, 1 for d/dy, 2 for d/dz

    exponents = np.array( [b1[1] + b2[1], b1[2] + b2[2], b1[3] + b2[3]] )

    initVal = b1[1+d1] * b2[1+d2] # multiply by the coefficients we get from taking a derivative

    exponents[d1] = exponents[d1] - 1 # decrement the powers on the coordinates that had the derivatives
    exponents[d2] = exponents[d2] - 1
    
    if np.any(exponents<0):
        return (0)
    elif np.any(exponents%2 == 1):
        return (0)
    else:
        initVal = initVal * 8 * ( 1/(exponents[0]+1) * Lx**(exponents[0]+1) ) * ( 1/(exponents[1]+1) * Ly**(exponents[1]+1) ) * ( 1/(exponents[2]+1) * Lz**(exponents[2]+1) )
        return (initVal)


# create a dictionary to store the result of all relevant integrals of the derivatives of two basis functions
# this is done because when fitting elastic constants, this has to be done only once, since it is independent of the elastic tensor
integratedGradBasis = {}
integratedGradBasis = np.zeros([R, 3, R, 3])
basisTotal = 0
for bN in np.arange(8): # this is for the different quadrants (see paper by Albert for details)
    for i in np.arange(int(basisPop[bN])):
        for j in np.arange(i, int(basisPop[bN])):
            b1 = basisfunctions[basisTotal+i]
            b2 = basisfunctions[basisTotal+j]
            for k in np.arange(3):
                for l in np.arange(3):
                    # integratedGradBasis[(basisTotal+i, basisTotal+j, k, l)] = integrateGradBasis (b1, b2, k, l, Lx, Ly, Lz)
                    integratedGradBasis[basisTotal+i, k, basisTotal+j, l] = integrateGradBasis (b1, b2, k, l, Lx, Ly, Lz)
    basisTotal = basisTotal + int(basisPop[bN])



Gmat = np.zeros([R,R])
basisTotal = 0
for bN in np.arange(8): # this is for the different quadrants (see paper by Albert for details)
    for i in np.arange(int(basisPop[bN])):
        for j in np.arange(i, int(basisPop[bN])):
            b1 = basisfunctions[basisTotal+i]
            b2 = basisfunctions[basisTotal+j]
            temporarySum = 0
            for k in np.arange(3):
                for l in np.arange(3):
                    # temporarySum = temporarySum + ctens[int(b1[0]), k, int(b2[0]), l] * integratedGradBasis[(basisTotal+i, basisTotal+j, k, l)] #integrateGradBasis (b1, b2, k, l, Lx, Ly, Lz)
                    temporarySum = temporarySum + ctens[int(b1[0]), k, int(b2[0]), l] * integratedGradBasis[basisTotal+i, k, basisTotal+j, l] #integrateGradBasis (b1, b2, k, l, Lx, Ly, Lz)
            Gmat[basisTotal+i, basisTotal+j] = temporarySum
            Gmat[basisTotal+j, basisTotal+i] = Gmat[basisTotal+i, basisTotal+j]
    basisTotal = basisTotal + int(basisPop[bN])


# print (Gmat[0])
# print (Emat[0])
t1 = time()
w = LA.eigh(Gmat, Emat, eigvals_only= True)
print (time()-t1)

# blocks = basisPop
# H = []
# K = []
# ss = 0
# for i in range(8):
#     H.append(Gmat[ss:ss+blocks[i], ss:ss+blocks[i]])
#     K.append(Emat[ss:ss+blocks[i], ss:ss+blocks[i]])
#     ss += blocks[i]
# w = np.array([])
# for i in range(8):
#     w = np.concatenate((w, LA.eigh(H[i], K[i], eigvals_only= True)))

    
f = np.sqrt(abs(w)) / 2 / np.pi / 1e6




for i in f[:30]:
    print (i)