import pyaudio
import numpy as np
import math
import os
from configparser import ConfigParser


class SoundGenerator:
    # material
    # M
    # K
    # evals
    # evecs
    # omegas
    # ksi
    # omega_ds
    # valid_map # 1 if the mode is valid

    def __init__(self, material_file, data_path, sampling_rate, duration) -> None:
        # read material
        cp = ConfigParser()
        cp.read(material_file, 'utf-8')
        self.material = {}
        for key in cp['DEFAULT']:
            self.material[key] = float(cp['DEFAULT'][key])

        # read nessessary data
        self.M = np.loadtxt( os.path.join(data_path, 'mass.txt'))
        self.K = np.loadtxt( os.path.join(data_path, 'stiff.txt'))
        self.evals = np.loadtxt( os.path.join(data_path, 'evals.txt'))
        self.evecs = np.loadtxt( os.path.join(data_path, 'evecs.txt'))

        self.fs = sampling_rate
        self.duration = duration

        self.calOmega()

    def calOmega(self):
        self.omegas = np.zeros(len(self.evals))
        self.valid_map = np.ones(len(self.evals))
        self.ksi = np.zeros(len(self.evals))
        self.omega_ds = np.zeros(len(self.evals))
        # self.ksi = (self.material['alpha'] + self.material['beta'] * self.evals) / 2 / self.omegas

        for i in range(len(self.evals)):
            if (self.evals[i] < 0):
                self.valid_map[i] = 0
                print('evals < 0 at ', i)
                continue
            self.omegas[i] = np.sqrt(self.evals[i])
            self.ksi[i] = (self.material['alpha'] + self.material['beta'] * self.evals[i]) / 2 / self.omegas[i]
            scale = 1 - self.ksi[i] * self.ksi[i]
            if (scale < 0 ):
                self.valid_map[i] = 0
                print('1 - ksi^2 < 0 at', i)
                continue
            self.omega_ds[i] = self.omegas[i] * np.sqrt(scale)
        
        # print(self.evals)
        # print(self.omegas)
        # print(self.omega_ds)
        # print(self.ksi)
        # print(self.valid_map)
        

sg_instance = SoundGenerator('./material/material-0.cfg', './output/cube-0', 44100, 1.0)
# print(sg_instance.material)
# print(sg_instance.M)
# print(sg_instance.K)
# print(sg_instance.evals)
# print(sg_instance.evecs)
