# Import packages

import numpy as np; print("Numpy version: ",np.__version__)
import scipy as sc; print("Scipy version: ",sc.__version__)
import scipy.spatial as sps
from scipy.linalg import svd

# Set number of decimal places to 3
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Functions

# Define function to calculate the covariances

def covar ( t, d, r ):
    h = d / r
    if t == 1: #Spherical
        c = 1 - h * (1.5 - 0.5 * h**2)
        c[h > 1] = 0
    elif t == 2: #Exponential
        c = np.exp( -3 * h )
    elif t == 3: #Gaussian
        c = np.exp( -3 * h**2 )
    return c


# Data

""" This piece of code is used to generate correlated data given a correlation matrix. We used it to generate the vector of data
in the small example section"""

# Correlation matrix
corr_mat= np.array([[1.0, 0.62],
                    [0.62, 1.0]])

# Compute the (upper) Cholesky decomposition matrix
upper_chol = sc.linalg.cholesky(corr_mat)

# Generate series of normally distributed (Gaussian) numbers
rnd = np.random.normal(0.0, 1.0, size=(2, 2))

# Finally, compute the inner product of upper_chol and rnd
data = rnd @ upper_chol

# Problem setup

# Consider Z1 and Z2 from the example
z = np.array([[0.146, -0.264],[-1.207,  1.155]])
print("Vector of data values, z:")
print(z)

# Define the LMC coefficients
A = np.array([[ 0.843 , 0.504 , 0.168 , 0.084], [ 0.347 , 0.347,  0.867 , 0.087]])

print("LMC squared coefficients explaining the contribution of the direct variogram structures")
print(A**2)

print("LMC cross coefficients explaining the contribution of the cross variogram structures")
print(np.prod(A, axis=0))

print("Matrix of A coefficients:")
print(A,"\n")

print("Sum to unity check (check if the LMC coefficients add up to 1)")
print(np.sum(A**2,1),"\n")

print("A'A == correlation check (check if the LMC fits the correlation of 0.62)")
print(np.matmul(A,A.T))

# Define some variables for the example
nvars = 2 # number of variables
nstruct = 4 # number of LMC structures
structtype = np.array([1,1,1,1]) # structure types (1 = spherical)
ranges = [2,4,7,10] # variogram ranges

# Define data locations
locs = np.array([[0,0],[0,1]])
print("Data locations")
print(locs)

# Covariances and cokriging

# Compute distance matrix and Setup block covariance matrix for all Y's

a = 0 * 3.14159265358979 / 180 # setting rotation to zero (no anisotropy)
rmat = np.asarray([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]) # define rotation matrix
p = locs.copy() # make a copy of data location
pm = np.mean(p) # centre calculations
q0 = np.matmul(p-pm,rmat)

nxy = locs.shape[0] # number of locations

C = [] # store covariances
for i in range(nstruct): # calculate covariances
    Q = q0.copy()
    Q[:,0] = Q[:,0] / ranges[i]
    Q[:,1] = Q[:,1] / ranges[i]
    d = sps.distance_matrix(Q,Q)
    c = covar(structtype[i],d,1)
    C.append(c)

print("Covariance matrices (by structure):")
for i in range(nstruct):
    print("Structure {}".format(i+1))
    print(C[i])

print('Generating SigmaY (CY) and reshape A matrices (AY)')

CY = np.zeros([nstruct*nxy,nstruct*nxy])
AY = np.zeros([nvars*nxy,nstruct*nxy])
for i in range(nstruct):
    CY[i*nxy:(i+1)*nxy,i*nxy:(i+1)*nxy] = C[i]
    AY[0:nxy,i*nxy:(i+1)*nxy] = np.eye(nxy) * A[0,i]
    AY[nxy:2*nxy,i*nxy:(i+1)*nxy] = np.eye(nxy) * A[1,i]

print("CY = \n",CY)
print("\nAY = \n",AY)

print("Covariance matrix between Y and Z")
print("Cyz = CY @ AY.T")
Cyz = CY @ AY.T

print("Cyz = \n",Cyz)

print("\nCovariance matrix of Z")
print("CZ = AY @ CY @ AY.T")
CZ = AY @ CY @ AY.T

print("CZ = \n",CZ)

print("Conditional mean solution (cokriging y given z)")
d = np.linalg.solve(CZ,np.reshape(z.T,[nvars*nxy,1]))
ybar = Cyz @ d
y0 = np.reshape(ybar,[nstruct,nxy]).T

print("y0 =\n",y0)

print("Checking the cokriging estimates to ensure numerical consistency of Zs")
print("Estimates matrix / reference matrix")

zt = np.matmul(A,y0.T).T
for i in range(2):
    print(zt[i,:],z[i,:])

# Singular value decomposition

u,s,v = svd(AY, full_matrices=True, lapack_driver='gesvd')

print("SVD matrices \n")
print("Matrix U =\n",u)
print("\nMatrix S =\n",s)
print("\nMatrix V =\n",v)

# Orthogonal complement - row space of v
vorth = v[0:nvars*nxy,:]
vperp = v[nvars*nxy:,:]

print("Orthonormal basis, =\n",vorth)
print("\nOrthogonal complement, (row space of v) =\n",vperp)

# Computation of Lambda for computing covariance of X
Lambda = vorth @ (CY @ vorth.T)
Lambda = vorth.T @ np.linalg.solve(Lambda,vorth)
Lambda = np.eye(CY.shape[0]) - CY @ Lambda
Lambda = vperp @ Lambda

print("\nLambda = \n",Lambda)

# Covariance of X and mean of X (should be zero)
CX = Lambda @ (CY @ vperp.T)
Xm = Lambda @ np.reshape(-y0.T,[nstruct*nxy,1])

print("Mean of Xm (should be zero):\n",np.mean(Xm),"\n")
print("Covariance of Xm (should be zero) \n",np.cov(Xm,rowvar=False),"\n")

print('SigmaX (CX):')
print(CX)

# Simulation of the factors

print("Performing Cholesky decomposition of SigmaX")
# Cholesky of covariance of X for generating realizations
BigLower = np.linalg.cholesky(CX)

print("Lower L matrix:")
print(BigLower)

"""Generate realizations of Y. This piece of code generates realizations of Y for any vector of random samples"""

# Consider the two vectors from the article
r = np.array([[-0.355,1.909,0.593,-1.076],
              [-0.260,0.129,-0.698,1.482]])

nreal = 2
numx = 2 # number of structures Y (4) - number of Z (2)
for ir in range(nreal):
    print("Realization #{}\n".format(ir+1))
    # r = np.random.normal(0,1,[numx*nxy,1])
    print("random vector r:")
    print(r[ir])
    
    x = BigLower @ r[ir]
    yi = np.reshape(vperp.T @ x,[nstruct,nxy]).T
    yt = yi + y0
    print("\n simulated y vector:")
    print(np.reshape(yt.T,[nstruct*nxy,1]))

    print("\n Computed z values:")
    zt = AY @ np.reshape(yt.T,[nstruct*nxy,1])
    zt = np.reshape(zt,[2,nxy]).T
    print(zt)
    print("\n Rreference z values:")
    print(z)
    print()