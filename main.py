from cmath import pi
from msilib.schema import Directory
import numpy as np
from numpy.linalg import *
import os
import time

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

from configparser import ConfigParser
from scipy.linalg import eigh
class ModalAnalysis:
    # vtk_filepath = "./model/cube.vtk"
    # mesh_points = []
    # mesh_elements = []
    # num_vtx = 0
    # material
    #   young
    #   poisson
    #   density
    #   alpha
    #   beta
    # M
    # K
    # evals
    # evecs

    def __init__(self, vtk_file, material_file, output_path) -> None:
        if( os.path.exists(output_path) ):
            output_path = output_path + '-' + str(int(time.time()))
        self.output_path = output_path


        self.vtk_filepath = vtk_file
        self.mesh_points, self.mesh_elements = getMeshInfo_vtk(self.vtk_filepath)
        self.num_vtx = len(self.mesh_points)

        cp = ConfigParser()
        cp.read(material_file, 'utf-8')
        self.material = {}
        for key in cp['DEFAULT']:
            self.material[key] = float(cp['DEFAULT'][key])
        # test begins ###
        for key in self.material:
            print(self.material[key])
        # test ends #####
        
        self.constructMK()
        self.eignDecom()

    def constructMK(self):
        M_ori = np.zeros((3 * self.num_vtx, 3 * self.num_vtx))
        K_ori = np.zeros((3 * self.num_vtx, 3 * self.num_vtx))

        m = getElementMass()
        for i, ele_pts_idx in enumerate(self.mesh_elements):
            ele_pts_pos = [self.mesh_points[ele_pts_idx[p]] for p in range(4)]
            volume = abs(getSignedTetVolume(ele_pts_pos))
            M_i = element2Global(m * self.material['density'] * volume, self.num_vtx, ele_pts_idx)
            k_i = getElementStiffness(ele_pts_pos, self.material['young'], self.material['poisson'])
            K_i = element2Global(k_i, self.num_vtx, ele_pts_idx)
            M_ori += M_i
            K_ori += K_i

        self.M = M_ori
        self.K = K_ori

    def eignDecom(self):
        # generalized eigenvalue decomposition
        self.evals, self.evecs = eigh(self.K, self.M)
        # omega = np.sqrt(evals)

    def printToFile(self):
        if( not os.path.exists(self.output_path) ):
            os.mkdir(self.output_path)
        
        print('[ DEBUG] The output dir is ' + self.output_path)

        f = open(os.path.join(self.output_path, "print.txt"), 'wt')

        print("\n# of vtx", file=f)
        print(self.num_vtx, file=f)

        # print("\nMass Matrix", file=f)
        # print(self.M, file=f)

        # print("\nStiffness Matrix", file=f)
        # print(self.K, file=f)

        # print("\nEigen values", file=f)
        # print(self.evals, file=f)

        print("\nfreq", file=f)
        print(np.sqrt(self.evals) / 2 / pi, file=f)

        # print("\nEigen vectors", file=f)
        # print(self.evecs, file=f)
    
    def saveData(self):
        if( not os.path.exists(self.output_path) ):
            os.mkdir(self.output_path)

        print('[ DEBUG] The output dir is' + self.output_path)

        np.savetxt( os.path.join(self.output_path, "mass.txt"), self.M)
        np.savetxt( os.path.join(self.output_path, "stiff.txt"), self.K)
        np.savetxt( os.path.join(self.output_path, "evals.txt"), self.evals)
        np.savetxt( os.path.join(self.output_path, "evecs.txt"), self.evecs)


ma_instance = ModalAnalysis('./model/cube.vtk', './material/material-0.cfg', './output/cube-0')
ma_instance.printToFile()
ma_instance.saveData()
