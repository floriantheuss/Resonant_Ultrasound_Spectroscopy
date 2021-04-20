import numpy as np
from numpy import cos, sin



def elastic_tensor (pars):
    """
    returns the elastic tensor from given elastic constants in pars
    (a dictionary of elastic constants)
    based on the length of pars it decides what crystal structure we the sample has
    """
    ctens = np.zeros([3,3,3,3])

    if len(pars) == 3:                      # cubic
        c11 = c22 = c33 = pars['c11']
        c12 = c13 = c23 = pars['c12']
        c44 = c55 = c66 = pars['c44']

    elif len(pars) == 5:                    # hexagonal
        indicator = np.any( np.array([i=='c11' for i in pars]) )
        if indicator == True:
            c11 = c22       = pars['c11']
            c33             = pars['c33']
            c12             = pars['c12']
            c13 = c23       = pars['c13']
            c44 = c55       = pars['c44']
            c66             = (pars['c11']-pars['c12'])/2
        else:
            c11 = c22       = 2*pars['c66'] + pars['c12']
            c33             = pars['c33']
            c12             = pars['c12']
            c13 = c23       = pars['c13']
            c44 = c55       = pars['c44']
            c66             = pars['c66']
        
    elif len(pars) == 6:                    # tetragonal
        c11 = c22       = pars['c11']
        c33             = pars['c33']
        c12             = pars['c12']
        c13 = c23       = pars['c13']
        c44 = c55       = pars['c44']
        c66             = pars['c66']
        
    elif len(pars) == 9:                    # orthorhombic
        c11             = pars['c11']
        c22             = pars['c22']
        c33             = pars['c33']
        c12             = pars['c12']
        c13             = pars['c13']
        c23             = pars['c23']
        c44             = pars['c44']
        c55             = pars['c55']
        c66             = pars['c66']
        
    else:
        print ('You have not given a valid Crystal Structure')

    ctens[0,0,0,0] = c11
    ctens[1,1,1,1] = c22
    ctens[2,2,2,2] = c33
    ctens[0,0,1,1] = ctens[1,1,0,0] = c12
    ctens[2,2,0,0] = ctens[0,0,2,2] = c13
    ctens[1,1,2,2] = ctens[2,2,1,1] = c23
    ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = c66
    ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = c55
    ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = c44

    return ctens


def to_Voigt (ctens):
    """
    takes an elastic tensor and returns a dictionary of elastic constants in Voigt notation
    """
    c_Voigt = {}
    c_Voigt['c11'] = ctens[0,0,0,0]
    c_Voigt['c22'] = ctens[1,1,1,1]
    c_Voigt['c33'] = ctens[2,2,2,2]
    c_Voigt['c44'] = ctens[1,2,1,2]
    c_Voigt['c55'] = ctens[0,2,0,2]
    c_Voigt['c66'] = ctens[0,1,0,1]
    c_Voigt['c12'] = ctens[0,0,1,1]
    c_Voigt['c13'] = ctens[0,0,2,2]
    c_Voigt['c14'] = ctens[0,0,1,2]
    c_Voigt['c15'] = ctens[0,0,0,2]
    c_Voigt['c16'] = ctens[0,0,0,1]
    c_Voigt['c23'] = ctens[1,1,2,2]
    c_Voigt['c24'] = ctens[1,1,1,2]
    c_Voigt['c25'] = ctens[1,1,0,2]
    c_Voigt['c26'] = ctens[1,1,0,1]
    c_Voigt['c34'] = ctens[2,2,1,2]
    c_Voigt['c35'] = ctens[2,2,0,2]
    c_Voigt['c36'] = ctens[2,2,0,1]
    c_Voigt['c45'] = ctens[1,2,0,2]
    c_Voigt['c46'] = ctens[1,2,0,1]
    c_Voigt['c56'] = ctens[0,2,0,1]
    return (c_Voigt)




def rotatation_matrix (alpha, beta, gamma):
    """
    define general 3D rotation matrix with rotation angles alpha, beta, gamma about x, y, z 
    axes respectively;
    angles are given in degrees
    """
    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180
    Rx = np.array([[1, 0, 0], [0, cos(alpha), -sin(alpha)] ,[0, sin(alpha), cos(alpha)]])
    Ry = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
    Rz = np.array([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])
    return np.matmul(Rz, np.matmul(Ry, Rx))

    

def rotate_ctens (alpha, beta, gamma, ctens):
    """
    takes angles alpha, beta, gamma and an elastic tensor and returns the rotated elastic tensor
    """
    crot =  np.zeros([3,3,3,3])
    R = rotatation_matrix (alpha, beta, gamma)
    for i in np.arange(3):
        for j in np.arange(3):
            for k in np.arange(3):
                for l in np.arange(3):
                    ctemp = 0
                    for a in np.arange(3):
                        for b in np.arange(3):
                            for c in np.arange(3):
                                for d in np.arange(3):
                                    ctemp += R[i,a]*R[j,b]*R[k,c]*R[l,d]*ctens[a,b,c,d]
                    crot[i,j,k,l] = ctemp
    return crot


cin = {'c11':110, 'c33':90, 'c13':70, 'c12':50, 'c44':30, 'c66':10}
alpha = 10
beta = 0
gamma = 0
crot = to_Voigt(rotate_ctens(alpha, beta, gamma, elastic_tensor(cin)))

print(crot)
