import numpy as np
from numpy.linalg import *
import argparse, os
import time
from configparser import ConfigParser
from scipy.linalg import eigh

from scipy.io.wavfile import write


def getSignedTetVolume(ele_points):
    point0 = np.resize(ele_points[0], (3, 1))
    point1 = np.resize(ele_points[1], (3, 1))
    point2 = np.resize(ele_points[2], (3, 1))
    point3 = np.resize(ele_points[3], (3, 1))
    Dm = np.hstack((point0 - point3, point1 - point3, point2 - point3))
    volume = det(Dm) / 6
    return volume


def getElementStiffness(ele_points, youngs, poisson):
    lambda_ = youngs * poisson / (1 + poisson) / (1 - 2 * poisson)
    mu_ = youngs * 0.5 / (1 + poisson) 
    stiff = np.zeros((12, 12))
    mb = np.zeros((4, 4))
    for i in range(4):
        mb[0][i] = ele_points[i][0]
        mb[1][i] = ele_points[i][1]
        mb[2][i] = ele_points[i][2]
        mb[3][i] = 1
    beta_ = np.linalg.inv(mb)
    for i in range(4):
        for j in range(4):
            for a in range(3):
                for b in range(3):
                    I = i*3 + a
                    J = j*3 + b
                    stiff[I][J] = lambda_ * beta_[i][a] * beta_[j][b]
                    # stiff[I][J] = lambda_ * beta_[i][a] * beta_[j][b] + mu_ * beta_[i][b] * beta_[j][a]
                    if ( a == b ):
                        sum = 0
                        for k in range(3):
                            sum += beta_[i][k] * beta_[j][k]
                        stiff[I][J] += mu_ * sum
                    stiff[I][J] *= 0.5 * abs(getSignedTetVolume(ele_points))
    return stiff


def getMeshInfo_vtk(filename):
    import meshio
    mesh = meshio.read(filename)
    mesh_points = mesh.points.tolist()
    mesh_elements = mesh.cells[0].data.tolist()
    return mesh_points, mesh_elements


class ModalAnalysis:

    def __init__(self) -> None:
        pass

    def setVtkFile(self, vtk_file):
        print("[ INFO] Reading mesh info...")
        self.vtk_filepath = vtk_file
        self.mesh_points, self.mesh_elements = getMeshInfo_vtk(self.vtk_filepath)
        self.num_vtx = len(self.mesh_points)
        print('[ INFO] done')

    def setMaterial(self, material_file):
        print("[ INFO] Reading material file...", material_file)
        cp = ConfigParser()
        cp.read(material_file, 'utf-8')
        self.material = {}
        for key in cp['DEFAULT']:
            self.material[key] = float(cp['DEFAULT'][key])
        print('[ INFO] done')

    def setOutputPath(self, output_path):
        self.output_path = output_path


    def constructM_ori(self):
        print("[ INFO] Generating M ori matrix...")
        M_ori = np.zeros((3 * self.num_vtx, 3 * self.num_vtx))
        for ele, ele_pts_idx in enumerate(self.mesh_elements):
            ele_pts_pos = [self.mesh_points[ele_pts_idx[p]] for p in range(4)]
            volume = abs(getSignedTetVolume(ele_pts_pos))
            for i in range(4):
                for j in range(4):
                    for k in range(3):
                        I = ele_pts_idx[i]
                        J = ele_pts_idx[j]
                        M_ori[3*I+k][3*J+k] += 0.05 * self.material['density'] * volume * ( 1 + (i == j))

            if(ele % 50 == 0):
                print("at element ", ele)
        self.M_ori = M_ori
        self.M_fix = M_ori
        print('[ INFO] done')

    def constructK_ori(self):
        print("[ INFO] Generating K ori matrix...")
        K_ori = np.zeros((3 * self.num_vtx, 3 * self.num_vtx))
        for ele, ele_pts_idx in enumerate(self.mesh_elements):
            ele_pts_pos = [self.mesh_points[ele_pts_idx[p]] for p in range(4)]

            # k_i = getElementStiffness(ele_pts_pos, self.material['youngs'], self.material['poisson'])

            youngs = self.material['youngs']
            poisson = self.material['poisson']

            # same as getElementStiffness
            lambda_ = youngs * poisson / (1 + poisson) / (1 - 2 * poisson)
            mu_ = youngs * 0.5 / (1 + poisson) 
            mb = np.zeros((4, 4))
            for i in range(4):
                mb[0][i] = ele_pts_pos[i][0]
                mb[1][i] = ele_pts_pos[i][1]
                mb[2][i] = ele_pts_pos[i][2]
                mb[3][i] = 1
            beta_ = np.linalg.inv(mb)

            volume = abs(getSignedTetVolume(ele_pts_pos))
            
            for i in range(4):
                for j in range(4):
                    for a in range(3):
                        for b in range(3):
                            value = lambda_ * beta_[i][a] * beta_[j][b]
                            if ( a == b ):
                                sum_ = 0
                                for k in range(3):
                                    sum_ += beta_[i][k] * beta_[j][k]
                                value += mu_ * sum_
                            value *= 0.5 * volume

                            I = ele_pts_idx[i]
                            J = ele_pts_idx[j]
                            K_ori[3*I+a][3*J+b] += value
            if(ele % 50 == 0):
                print("at element ", ele)
        self.K_ori = K_ori
        self.K_fix = K_ori
        print('[ INFO] done')

    def saveMK_npz(self):
        print('[ INFO] saving Mass & Stiff matrix to' + self.output_path)
        np.savez(self.output_path, mass=self.M_ori, stiff=self.K_ori)
        print('[ INFO] done')

# ./main.py -m 0 -ip './model/r02.vtk' -op './output/r02' 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--material', type=int, default=0, help='Material Num 0-7 [default: 0]')
parser.add_argument('-ip', '--inputpath', type=str, default='./test_vtk/T1.vtk', help='Input path')
parser.add_argument('-op', '--outputpath', type=str, default='./test_eigen/T1.npz', help='Output path')

FLAGS = parser.parse_args()

material_path = './material/material-{}.cfg'.format(FLAGS.material)

ma_instance = ModalAnalysis()
ma_instance.setVtkFile(FLAGS.inputpath)
ma_instance.setMaterial(material_path)
ma_instance.setOutputPath(FLAGS.outputpath)

print("[ INFO] Constructing MK...")
ma_instance.constructM_ori()
ma_instance.constructK_ori()

print("[ INFO] Saving MK...")
ma_instance.saveMK_npz()

print("[ INFO] All completed.")