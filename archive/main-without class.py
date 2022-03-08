import numpy as np
from numpy.linalg import *


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
    gamma_list = [0, 0, 0, 0]
    delta_list = [0, 0, 0, 0]

    beta_list[0] = -calDet6(y2, z2, y3, z3, y4, z4)
    beta_list[1] =  calDet6(y1, z1, y3, z3, y4, z4)
    beta_list[2] = -calDet6(y1, z1, y2, z2, y4, z4)
    beta_list[3] =  calDet6(y1, z1, y2, z2, y3, z3)

    gamma_list[0] =  calDet6(x2, z2, x3, z3, x4, z4)
    gamma_list[1] = -calDet6(x1, z1, x3, z3, x4, z4)
    gamma_list[2] =  calDet6(x1, z1, x2, z2, x4, z4)
    gamma_list[3] = -calDet6(x1, z1, x2, z2, x3, z3)

    delta_list[0] = -calDet6(x2, y2, x3, y3, x4, y4)
    delta_list[1] =  calDet6(x1, y1, x3, y3, x4, y4)
    delta_list[2] = -calDet6(x1, y1, x2, y2, x4, y4)
    delta_list[3] =  calDet6(x1, y1, x2, y2, x3, y3)

    return beta_list, gamma_list, delta_list


def getSignedTetVolume(ele_points):
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
    D1 = np.array([[r_, poisson, poisson],
                   [poisson, r_, poisson],
                   [poisson, poisson, r_],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    I63 = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    D2 = s_ * I63
    D = np.hstack((D1, D2))
    D = D * q_
    return D


def getElementStiffness(ele_points, young, poisson):
    beta_list, gamma_list, delta_list = getBetaGammaDelta(ele_points)
    volumn = abs(getSignedTetVolume(ele_points))
    B = getBMatrix(beta_list, gamma_list, delta_list)
    D = getDMatrix(young, poisson)
    k = B.T.dot(D).dot(B) * volumn
    return k


# mass: mass of the tet
def getElementMass():
    m = np.zeros((12, 12))
    I3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(4):
        for j in range(4):
            if i == j:
                m[3*i:3*i+3, 3*j:3*j+3] = 2 * I3
            else:
                m[3*i:3*i+3, 3*j:3*j+3] = I3
    m = m / 20
    return m


# matrix: 12*12 expand to global size
# idx: the global index of vertices in the element matrix
def element2Global(matrix, n_vtx, idx):
    M = np.zeros((3 * n_vtx, 3 * n_vtx))
    for i in range(4):
        for j in range(4):
            for k in range(3):
                for l in range(3):
                    I = idx[i]
                    J = idx[j]
                    M[3*I+k][3*J+l] = matrix[3*i+k][3*j+l]
    return M


def getMeshInfo_vtk(filename):
    import meshio
    mesh = meshio.read(filename)
    mesh_points = mesh.points.tolist()
    mesh_elements = mesh.cells[0].data.tolist()
    return mesh_points, mesh_elements

# read mesh
vtk_filename = "cube.vtk"
mesh_points, mesh_elements = getMeshInfo_vtk(vtk_filename)

# set the parameters
num_vtx = len(mesh_points)

young = 1.9e11
poisson = 0.27
density = 7.2e3
fixed_vtx = []

# construct M, K
M_ori = np.zeros((3 * num_vtx, 3 * num_vtx))
K_ori = np.zeros((3 * num_vtx, 3 * num_vtx))

m = getElementMass()
for i, ele_pts_idx in enumerate(mesh_elements):
    ele_pts_pos = [mesh_points[ele_pts_idx[p]] for p in range(4)]
    volume = abs(getSignedTetVolume(ele_pts_pos))
    M_i = element2Global(m * density * volume, num_vtx, ele_pts_idx)
    k_i = getElementStiffness(ele_pts_pos, young, poisson)
    K_i = element2Global(k_i, num_vtx, ele_pts_idx)
    M_ori += M_i
    K_ori += K_i

# remove the fixed vertexs
remove_index = []
for i in fixed_vtx:
    remove_index.append(3*i)
    remove_index.append(3*i + 1)
    remove_index.append(3*i + 2)

M = np.delete(M_ori, remove_index, axis=0)
M = np.delete(M, remove_index, axis=1)
K = np.delete(K_ori, remove_index, axis=0)
K = np.delete(K, remove_index, axis=1)


from scipy.linalg import eigh
# generalized eigenvalue decomposition
evals, evecs = eigh(K, M)
omega = np.sqrt(evals)

f = open("./output.txt", 'wt')
print("\n# of vtx", file=f)
print(num_vtx, file=f)
print("\nMass Matrix", file=f)
print(M, file=f)
print("\nStiffness Matrix", file=f)
print(K, file=f)
print("\nEigen values", file=f)
print(evals, file=f)
print("\nOmega", file=f)
print(omega, file=f)
print("\nEigen vectors", file=f)
print(evecs, file=f)