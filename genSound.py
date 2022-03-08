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
    # valid map

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

    def calOmega_d(self):
        self.omegas = np.sqrt(self.evals)
        self.ksi = (self.material['alpha'] + self.material['beta'] * self.evals) / 2 / self.omegas
        # self.omega_ds = self.omegas * np.sqrt(1 -self.ksi * self.ksi )
        # TODO
        # 遍历omegas，如果omega<0 pass 如果1-ksi*ksi<0 或ksi为nan pass 如果omega_d小于20*2pi大于20000*2pi，pass
        # valid map指示哪些omegas是有意义的
        

sg_instance = SoundGenerator('./material/material-0.cfg', './output/cube-0', 44100, 1.0)
# print(sg_instance.material)
# print(sg_instance.M)
# print(sg_instance.K)
# print(sg_instance.evals)
# print(sg_instance.evecs)
