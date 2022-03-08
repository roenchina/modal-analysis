import numpy as np
from numpy.linalg import *
from scipy.linalg import eigh


def getXYZ_list(point):
    # point: [0,0,0] list
    x = point[0]
    y = point[1]
    z = point[2]
    return x, y, z

def calDet6(a2, a3, b2, b3, c2, c3):
    mat = np.array([[1, a2, a3], [1, b2, b3], [1, c2, c3]])
    return det(mat)

def getBetaGammaDelta(ele_points):
    x1, y1, z1 = getXYZ_list(ele_points[0])
    x2, y2, z2 = getXYZ_list(ele_points[1])
    x3, y3, z3 = getXYZ_list(ele_points[2])
    x4, y4, z4 = getXYZ_list(ele_points[3])

    beta_list = [0, 0, 0, 0]
    gamma_list = [0 ,0, 0, 0]
    delta_list = [0, 0, 0, 0]

    beta_list[0] = - calDet6(y2, z2, y3, z3, y4, z4)
    beta_list[1] =   calDet6(y1, z1, y3, z3, y4, z4)
    beta_list[2] = - calDet6(y1, z1, y2, z2, y4, z4)
    beta_list[3] =   calDet6(y1, z1, y2, z2, y3, z3)

    gamma_list[0]=   calDet6(x2, z2, x3, z3, x4, z4)
    gamma_list[1]= - calDet6(x1, z1, x3, z3, x4, z4)
    gamma_list[2]=   calDet6(x1, z1, x2, z2, x4, z4)
    gamma_list[3]= - calDet6(x1, z1, x2, z2, x3, z3)

    delta_list[0]= - calDet6(x2, y2, x3, y3, x4, y4)
    delta_list[1]=   calDet6(x1, y1, x3, y3, x4, y4)
    delta_list[2]= - calDet6(x1, y1, x2, y2, x4, y4)
    delta_list[3]=   calDet6(x1, y1, x2, y2, x3, y3)

    return beta_list, gamma_list, delta_list

def getTetVolume(ele_points):
    point0 = np.resize(ele_points[0], (3, 1))
    point1 = np.resize(ele_points[1], (3, 1))
    point2 = np.resize(ele_points[2], (3, 1))
    point3 = np.resize(ele_points[3], (3, 1))
    Dm = np.hstack((point0 - point3, point1 - point3, point2 - point3))
    volume = det(Dm) / 6
    return volume

def getBSubMatrix(beta, gamma, delta):
    b1 = np.array([beta, 0, 0])
    b2 = np.array([0, gamma, 0])
    b3 = np.array([0, 0, delta])
    b4 = np.array([gamma, beta, 0])
    b5 = np.array([0, delta, gamma])
    b6 = np.array([delta, 0, beta])
    b = np.array([b1, b2, b3, b4, b5, b6])
    return b

def getBMatrix(beta_list, gamma_list, delta_list):
    B0 = getBSubMatrix(beta_list[0], gamma_list[0], delta_list[0])
    B1 = getBSubMatrix(beta_list[1], gamma_list[1], delta_list[1])
    B2 = getBSubMatrix(beta_list[2], gamma_list[2], delta_list[2])
    B3 = getBSubMatrix(beta_list[3], gamma_list[3], delta_list[3])
    B = np.hstack((B0, B1, B2, B3))
    return B

def getDMatrix(young, poisson):
    q_ = young / (1 + poisson) / (1 - 2 * poisson)
    r_ = 1 - poisson
    s_ = (1 - 2 * poisson) / 2
    D1 = np.array([[r_, poisson, poisson], [poisson, r_, poisson], [poisson, poisson, r_], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    I3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    D2 = s_ * I3
    D = np.hstack((D1, D2))
    D = D * q_
    return D 
    
def getElementStiffness(ele_points, young, poisson):
    beta_list, gamma_list, delta_list = getBetaGammaDelta(ele_points)
    volumn = getTetVolume(ele_points)
    B = getBMatrix(beta_list, gamma_list, delta_list)
    D = getDMatrix(young, poisson)
    k = B.T.dot(D).dot(B) * volumn
    return k

# mass: mass of the tet
def getElementMass(mass):
    m = np.zeros((12, 12))
    I3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(4):
        for j in range(4):
            if i==j:
                m[3*i:3*i+3, 3*j:3*j+3] = 2 * I3
            else:
                m[3*i:3*i+3, 3*j:3*j+3] = I3
    m = m * mass / 20
    return m

# matrix: 12*12 expand to global size
# idx: the global index of vertices in the element matrix
def element2Global(matrix, n_vtx, idx):
    M = np.zeros((3*n_vtx, 3*n_vtx))
    for i in range(4):
        for j in range(4):
            for k in range(3):
                for l in range(3):
                    I = idx[i]
                    J = idx[j]
                    M[3*I+k][3*J+l] = matrix[3*i+k][3*j+l]
    return M

def testElement():
    num_vtx = 5
    ele_points_1 = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    ele_points_2 = [[0,0,0], [1,0,0], [0,1,0], [0,0,-1]]
    young = 9
    poisson = 2
    density = 6
    fixed_vtx = [0, ]

    M_ori = np.zeros((3*num_vtx, 3*num_vtx))
    K_ori = np.zeros((3*num_vtx, 3*num_vtx))

    m = getElementMass(1)
    volume1 = getTetVolume(ele_points_1)
    volume2 = getTetVolume(ele_points_2)
    M1 = element2Global(m*density*abs(volume1), num_vtx, [0, 1, 2, 3])
    M2 = element2Global(m*density*abs(volume2), num_vtx, [0, 1, 2, 4])
    
    k1 = getElementStiffness(ele_points_1, young, poisson)
    K1 = element2Global(k1, num_vtx, [0, 1, 2, 3])
    k2 = getElementStiffness(ele_points_2, young, poisson)
    K2 = element2Global(k2, num_vtx, [0, 1, 2, 4])

    # assembly of elements
    M_ori += M1 + M2
    K_ori += K1 + K2

    # remove the fixed vertexs
    M = np.delete(M_ori, fixed_vtx, axis=0)
    M = np.delete(M, fixed_vtx, axis=1)
    K = np.delete(K_ori, fixed_vtx, axis=0)
    K = np.delete(K, fixed_vtx, axis=1)

    # generalized eigenvalue decomposition
    evals, evecs = eigh(K,M)
    
    print(M)
    print(K)
    print(evals)
    print(evecs)

    return M, K, evals, evecs

M, K, evals, evecs = testElement()
