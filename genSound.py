import pyaudio
import numpy as np
import math
import os
from configparser import ConfigParser
from scipy.io.wavfile import write


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
    # samples
    # each_sample

    def __init__(self, material_file, matrix_path) -> None:
        # read material
        cp = ConfigParser()
        cp.read(material_file, 'utf-8')
        self.material = {}
        for key in cp['DEFAULT']:
            self.material[key] = float(cp['DEFAULT'][key])

        # read nessessary data
        self.M = np.loadtxt( os.path.join(matrix_path, 'mass.txt'))
        self.K = np.loadtxt( os.path.join(matrix_path, 'stiff.txt'))
        self.evals = np.loadtxt( os.path.join(matrix_path, 'evals.txt'))
        self.evecs = np.loadtxt( os.path.join(matrix_path, 'evecs.txt'))

        self.calOmega()

    def calOmega(self):
        self.omegas = np.zeros(len(self.evals))
        self.valid_map = np.ones(len(self.evals))
        self.ksi = np.zeros(len(self.evals))
        self.omega_ds = np.zeros(len(self.evals))

        for i in range(len(self.evals)):
            if (self.evals[i] < 0):
                self.valid_map[i] = 0
                # print('evals < 0 at ', i)
                continue
            self.omegas[i] = np.sqrt(self.evals[i])

            # TEST exchange alpha and beta ############################
            self.ksi[i] = (self.material['beta'] + self.material['alpha'] * self.evals[i]) / 2 / self.omegas[i]
            scale = 1 - self.ksi[i] * self.ksi[i]
            if (scale < 0 ):
                self.valid_map[i] = 0
                # print('1 - ksi^2 < 0 at', i)
                continue
            self.omega_ds[i] = self.omegas[i] * np.sqrt(scale)
        
        # print('self.evals=======================')
        # print(self.evals)
        # print('self.omegas=======================')
        # print(self.omegas)
        # print('self.omega_ds=======================')
        # print(self.omega_ds)
        # print('self.ksi=======================')
        # print(self.ksi)
        # print('self.valid_map=======================')
        # print(self.valid_map)

    def setDuration(self, duration):
        self.duration = duration

    def setSampRate(self, sampling_rate):
        self.fs = sampling_rate

    def setForce(self, force):
        self.force = force

    def getEigenLen(self):
        return len(self.evals)

    def genSound(self):
        time_slot = np.arange(self.fs * self.duration) / self.fs
        self.each_sample = np.zeros((len(self.evals), len(time_slot)))
        self.samples = np.zeros(len(time_slot))
        Uf = np.dot(self.evecs.transpose(), self.force)
        for i in range(len(self.valid_map)):
            if(self.valid_map[i]):
                # print('[ DEBUG] at mode', i, ' ======================')
                amplitude = np.exp(time_slot * (-1) * self.ksi[i] * self.omegas[i]) * abs(Uf[i]) / self.omega_ds[i]
                self.each_sample[i] = (np.sin(self.omega_ds[i] * time_slot ) * amplitude).astype(np.float32)

                # TEST ##########################################
                if (i > 10):
                    self.samples += self.each_sample[i]
                # print(self.omega_ds[i])
                # print(amplitude)
                # print(sample_i)
        print(self.samples)

    def playSound(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.fs,
                        output=True)
        stream.write(self.samples.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()

    def saveSound(self, filename):
        # TEST *1E6 ######################################
        tmp_samples = self.samples * 1e6
        write(filename, self.fs, tmp_samples)

    def saveEachMode(self, path):
        if( not os.path.exists(path) ):
            os.mkdir(path)
        for i in range(len(self.evals)):
            if (self.valid_map[i]):
                # TEST *1E6 ######################################
                tmp_samples = self.each_sample[i] * 1e6
                write(os.path.join(path, 'mode'+str(i)+'.wav'), self.fs, tmp_samples)




sg_instance = SoundGenerator('./material/material-0.cfg', './output/cube-0')
sg_instance.setDuration(3.0)
sg_instance.setSampRate(44100)

force = np.zeros(sg_instance.getEigenLen())
# force = (0.5, 0.1, 0) applied at point 9
force[24] = 0.5
force[25] = 0.1
# force[26] = 0

sg_instance.setForce(force)

sg_instance.genSound()

# sg_instance.playSound()
sg_instance.saveSound('./output/cube-0/res_sound.wav')
sg_instance.saveEachMode('./output/cube-0/modes/')
