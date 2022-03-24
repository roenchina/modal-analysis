from cmath import pi
import mailbox
from msilib.schema import Directory
import numpy as np
from numpy.linalg import *
import argparse, os
import time
from configparser import ConfigParser
from scipy.linalg import eigh

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
                    # stiff[I][J] = (lambda_ * beta_[i][a] * beta_[j][b] + mu_ * beta_[i][b] * beta_[j][a])
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
    # vtk_filepath = "./model/cube.vtk"
    # mesh_points = []
    # mesh_elements = []
    # num_vtx = 0
    # material
    #   youngs
    #   poisson
    #   density
    #   alpha
    #   beta
    # M
    # K
    # evals
    # evecs
    # fixed_vtx = []

    def __init__(self, vtk_file) -> None:
        print("[ INFO] Reading mesh info...")
        self.vtk_filepath = vtk_file
        self.mesh_points, self.mesh_elements = getMeshInfo_vtk(self.vtk_filepath)
        self.num_vtx = len(self.mesh_points)
        print('[ INFO] done')

    def setMaterial(self, material_file):
        print("[ INFO] Reading material file...")
        cp = ConfigParser()
        cp.read(material_file, 'utf-8')
        self.material = {}
        for key in cp['DEFAULT']:
            self.material[key] = float(cp['DEFAULT'][key])
        print('[ INFO] done')

    def setOutputPath(self, output_path):
        if( os.path.exists(output_path) ):
            output_path = output_path + '-' + str(int(time.time()))
        self.output_path = output_path
        if( not os.path.exists(self.output_path) ):
            os.mkdir(self.output_path)

    def setFixedVtx(self, new_fv):
        self.fixed_vtx = new_fv
        self.remove_index = []
        for i in self.fixed_vtx:
            self.remove_index.append(3*i)
            self.remove_index.append(3*i + 1)
            self.remove_index.append(3*i + 2)


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
            k_i = getElementStiffness(ele_pts_pos, self.material['youngs'], self.material['poisson'])
            for i in range(4):
                for j in range(4):
                    for k in range(3):
                        for l in range(3):
                            I = ele_pts_idx[i]
                            J = ele_pts_idx[j]
                            K_ori[3*I+k][3*J+l] += k_i[3*i+k][3*j+l]
            if(ele % 50 == 0):
                print("at element ", ele)
        self.K_ori = K_ori
        self.K_fix = K_ori
        print('[ INFO] done')

    def getM_fix(self):
        self.M_fix = np.delete(self.M_ori, self.remove_index, axis=0)
        self.M_fix = np.delete(self.M_fix, self.remove_index, axis=1)

    def getK_fix(self):
        self.K_fix = np.delete(self.M_ori, self.remove_index, axis=0)
        self.K_fix = np.delete(self.K_fix, self.remove_index, axis=1)


    def eignDecom(self):
        self.evals, self.evecs = eigh(self.K_fix, self.M_fix)

    
    def saveAllData(self):
        # if( not os.path.exists(self.output_path) ):
        #     os.mkdir(self.output_path)

        print('[ INFO] The output dir is' + self.output_path)

        np.savetxt( os.path.join(self.output_path, "mass_ori.txt"), self.M_ori)
        np.savetxt( os.path.join(self.output_path, "stiff_ori.txt"), self.K_ori)
        np.savetxt( os.path.join(self.output_path, "mass_fix.txt"), self.M_fix)
        np.savetxt( os.path.join(self.output_path, "stiff_fix.txt"), self.K_fix)
        np.savetxt( os.path.join(self.output_path, "evals.txt"), self.evals)
        np.savetxt( os.path.join(self.output_path, "evecs.txt"), self.evecs)

    def saveM_ori(self):
        # if( not os.path.exists(self.output_path) ):
        #     os.mkdir(self.output_path)
        print('[ INFO] saving Mass matrix to' + self.output_path)
        np.savetxt( os.path.join(self.output_path, "mass_ori.txt"), self.M_ori)
        print('[ INFO] done')

    def saveK_ori(self):
        # if( not os.path.exists(self.output_path) ):
        #     os.mkdir(self.output_path)
        print('[ INFO] saving Stiff matrix to' + self.output_path)
        np.savetxt( os.path.join(self.output_path, "stiff_ori.txt"), self.K_ori)
        print('[ INFO] done')

    def saveEvals(self):
        # if( not os.path.exists(self.output_path) ):
        #     os.mkdir(self.output_path)
        print('[ INFO] saving Eigen Values to' + self.output_path)
        np.savetxt( os.path.join(self.output_path, "evals.txt"), self.evals)
        print('[ INFO] done')


# ./main.py -m 1 -ip './model/r02.vtk' -op './output/r02' -fn 3
# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--material', type=int, default=1, help='Material Num 0-7 [default: 1]')
# parser.add_argument('-ip', '--inputpath', type=str, default='./model/r01.vtk', help='Input path')
# parser.add_argument('-op', '--outputpath', type=str, default='./output/r01', help='Output path')
# parser.add_argument('-fn', '--fixednum', type=int, default=5, help='# of fixed vertices [default: 5]')


# FLAGS = parser.parse_args()

# material_path = './material/material-{}.cfg'.format(FLAGS.material)
# fixed_vtx = [i for i in range(FLAGS.fixednum)]


# ma_instance = ModalAnalysis(FLAGS.inputpath)
# ma_instance.setMaterial(material_path)
# ma_instance.setOutputPath(FLAGS.outputpath)

# print("[ INFO] Constructing MK...")
# ma_instance.constructM_ori()
# ma_instance.saveM_ori()

# ma_instance.constructK_ori()

# ma_instance.setFixedVtx(fixed_vtx)
# ma_instance.getM_fix()
# ma_instance.getK_fix()

# # print("[ INFO] Eigen decomposition...")
# # ma_instance.eignDecom()

# # print("[ INFO] Saving data...")
# # ma_instance.saveAllData()

# print("[ INFO] All completed.")