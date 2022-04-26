import numpy as np
from numpy.linalg import *
import os
import time
from configparser import ConfigParser
from scipy.linalg import eigh

import pyaudio
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
        # if( os.path.exists(output_path) ):
        #     output_path = output_path + '-' + str(int(time.time()))
        self.output_path = output_path
        if( not os.path.exists(self.output_path) ):
            os.makedirs(self.output_path)

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
        self.K_fix = np.delete(self.K_ori, self.remove_index, axis=0)
        self.K_fix = np.delete(self.K_fix, self.remove_index, axis=1)


    def eignDecom(self):
        self.evals, self.evecs = eigh(self.K_fix, self.M_fix)

        self.c_evals = self.evals                   # 压缩之后的特征值
        # self.c_evecs = self.evecs                   # 压缩之后的特征向量
        self.c_evecs_transpose = self.evecs.transpose()

        self.num_modes = len(self.c_evals)          # 模态个数 = 特征值个数
        self.len_evec = self.c_evecs_transpose.shape[1]       # 特征向量长度 = 关键点*3
    
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

#### Compact Mode ###############################################
    def packModes(self, pack_n):
        n_evlas = len(self.evals)

        print('[ INFO] packing {} modes into {} modes'.format(n_evlas, pack_n))
        
        self.c_evals = np.zeros(pack_n)
        self.c_evecs_transpose = np.zeros((pack_n, self.len_evec))
        evecs_transpose = self.evecs.transpose()

        modes_each_pack = n_evlas // pack_n
        modes_left = n_evlas - modes_each_pack * pack_n
        offset = modes_each_pack // 2

        print('[ INFO] modes_each_pack = ', modes_each_pack)
        print('[ INFO] modes_left = ', modes_left)
        print('[ INFO] offset = ', offset)

        # 单独取其中一个模态，不可行
        # for i in range(pack_n):
        #     idx = modes_each_pack * i + offset
        #     self.c_evals[i] = self.evals[idx]
        #     self.c_evecs_transpose[i] = evecs_transpose[idx]
        #     print('new evals [{}] = old evals [{}]'.format(i, idx))

        # 取组内平均值，不可行
        # for i in range(pack_n):
        #     tmp_evec = np.zeros(self.len_evec)
        #     tmp_evals = 0
        #     for offset in range(modes_each_pack):
        #         idx = modes_each_pack * i + offset
        #         tmp_evals += self.evals[idx]
        #         tmp_evec += evecs_transpose[idx]
        #     self.c_evals[i] = tmp_evals / modes_each_pack
        #     self.c_evecs_transpose[i] = tmp_evec / modes_each_pack
        #     print('new evals [{}] = average old evals [{}-{}]'.format(i, modes_each_pack*i, modes_each_pack*(i+1)-1))



        self.num_modes = pack_n          # 模态个数 = 特征值个数
        # self.len_evec = self.c_evecs_transpose.shape[1]       # 特征向量长度 = 关键点*3

#### Sound Part ###############################################
    # omegas
    # ksi
    # omega_ds
    # valid_map # 1 if the mode is valid
    # samples
    # mode_samples

    def calOmega(self):
        self.omegas = np.zeros(self.num_modes)
        self.valid_map = np.ones(self.num_modes)
        self.ksi = np.zeros(self.num_modes)
        self.omega_ds = np.zeros(self.num_modes)

        print('[ INFO] calculating omega of each mode...')
        for i in range(self.num_modes):
            if (self.c_evals[i] < 0):
                self.valid_map[i] = 0
                print('evals < 0 at ', i)
                continue
            self.omegas[i] = np.sqrt(self.c_evals[i])

            self.ksi[i] = (self.material['beta'] + self.material['alpha'] * self.c_evals[i]) / 2 / self.omegas[i]
            scale = 1 - self.ksi[i] * self.ksi[i]
            if (scale < 0 ):
                self.valid_map[i] = 0
                print('1 - ksi^2 < 0 at', i)
                continue
            self.omega_ds[i] = self.omegas[i] * np.sqrt(scale)
            if (self.omega_ds[i] > 600 and self.omega_ds[i] < 7e4):
                pass
            else:
                print('omega_d[{}] = {} is out of 20hz 20000hz range'.format(i, self.omega_ds[i]))
                self.valid_map[i] = 0

    def setDuration(self, duration):
        self.duration = duration

    def setSampRate(self, sampling_rate):
        self.fs = sampling_rate

    def setForce(self, force):
        self.force = force
    
    def setForce(self, contact_idx, f_x, f_y, f_z):
        force = np.zeros(self.len_evec)
        force[3*contact_idx] = f_x
        force[3*contact_idx+1] = f_y
        force[3*contact_idx+2] = f_z
        self.force = force

    def genSound(self):
        self.calOmega()

        time_slot = np.arange(self.fs * self.duration) / self.fs

        self.mode_sample = np.zeros((self.num_modes, len(time_slot)))
        self.samples = np.zeros(len(time_slot))

        Uf = np.dot(self.c_evecs_transpose, self.force)

        print('[ INFO] summing up all valid modes...')
        for i in range(len(self.valid_map)):
            if(self.valid_map[i]):
                amplitude = np.exp(time_slot * (-1) * self.ksi[i] * self.omegas[i]) * abs(Uf[i]) / self.omega_ds[i]
                self.mode_sample[i] = (np.sin(self.omega_ds[i] * time_slot ) * amplitude).astype(np.float32)
                print('mode ', i, ' omega_d = ', self.omega_ds[i])
                self.samples += self.mode_sample[i]

################### pack test begin ###################
        # print('[ INFO] packing test...')

        # time_slot = np.arange(self.fs * self.duration) / self.fs
        # self.pack_samples = np.zeros(len(time_slot))

        # pack_n = 4
        # valid_modes = 61

        # modes_each_pack = valid_modes // pack_n
        # modes_left = valid_modes - modes_each_pack * pack_n
        # offset = 1

        # print('[ INFO] modes_each_pack = ', modes_each_pack)
        # print('[ INFO] modes_left = ', modes_left)
        # print('[ INFO] offset = ', offset)

        # # 单独取其中一个模态
        # for i in range(pack_n):
        #     idx = modes_each_pack * i + offset + 3
        #     self.pack_samples += self.mode_sample[idx]
        #     print('new modes [{}] = old evals [{}]'.format(i, idx))

################### pack test end ###################

    def saveSound(self):
        filename = os.path.join(self.output_path, 'res_sound.wav')
        tmp_samples = self.samples * 1e6
        write(filename, self.fs, tmp_samples)

        # filename = os.path.join(self.output_path, 'pack_sound.wav')
        # tmp_samples = self.pack_samples * 1e6
        # write(filename, self.fs, tmp_samples)

    def saveEachMode(self):
        path = os.path.join(self.output_path, 'modes')
        if( not os.path.exists(path) ):
            os.mkdir(path)
        for i in range(self.num_modes):
            if (self.valid_map[i]):
                tmp_samples = self.mode_sample[i] * 1e6
                write(os.path.join(path, 'mode'+str(i)+'.wav'), self.fs, tmp_samples)

############# bugs ###############################
    def playSound(self):
        pass

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