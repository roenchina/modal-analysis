import pyaudio
import numpy as np
import math
import argparse, os
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
    # material_file
    # matrix_path

    def __init__(self, material_file, matrix_path) -> None:
        # read material
        self.material_file = material_file
        self.matrix_path = matrix_path
        cp = ConfigParser()
        cp.read(material_file, 'utf-8')
        self.material = {}
        for key in cp['DEFAULT']:
            self.material[key] = float(cp['DEFAULT'][key])

        # read nessessary data
        self.M = np.loadtxt( os.path.join(matrix_path, 'mass_fix.txt'))
        self.K = np.loadtxt( os.path.join(matrix_path, 'stiff_fix.txt'))
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

            # TEST exchange alpha and beta ( which is correct by ModalModel.cpp line 98)############################
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
                print('mode ', i, ' omega_d = ', self.omega_ds[i])

                # TEST ##########################################
                if (self.omega_ds[i] > 3500 and self.omega_ds[i] < 4e10):
                    self.samples += self.each_sample[i]
                    print('mode ', i, 'is in 20hz 20000hz range')
                # print(self.omega_ds[i])
                # print(amplitude)
                # print(sample_i)
        # print(self.samples)

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

    def saveSound(self):
        # TEST *1E6 ######################################
        filename = os.path.join(self.matrix_path, 'res_sound.wav')
        tmp_samples = self.samples * 1e6
        write(filename, self.fs, tmp_samples)

    def saveEachMode(self):
        path = os.path.join(self.matrix_path, 'modes')
        if( not os.path.exists(path) ):
            os.mkdir(path)
        for i in range(len(self.evals)):
            if (self.valid_map[i]):
                # TEST *1E6 ######################################
                tmp_samples = self.each_sample[i] * 1e6
                write(os.path.join(path, 'mode'+str(i)+'.wav'), self.fs, tmp_samples)


# ./genSound.py -m 1 -p ./output/plate-nt-zcx -d 3.1 -sr 44101 

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--material', type=int, default=1, help='Material Num 0-7 [default: 1]')
parser.add_argument('-p', '--matrixpath', type=str, default='./output/plate-nt', help='The Matrix path (same as sound output path) [default: \'./output/plate-nt\']')
parser.add_argument('-d', '--duration', type=float, default=2.0, help='Sound duration [default: 2.0]')
parser.add_argument('-sr', '--samprate', type=int, default=44100, help='Sample rate (must be an integer) [default: 44100]')
parser.add_argument('-cp', '--contactpoint', type=int, default=0, help='Contact point index [default: 0]')
parser.add_argument('-f', '--force', type=float, default=[0,0.5,0.1], nargs=3, help='Force - 3d list [default: [0, 0.5, 0.1]]')


FLAGS = parser.parse_args()

material_path = './material/material-{}.cfg'.format(FLAGS.material)
matrix_path = FLAGS.matrixpath
# print(FLAGS)
# print(material_path, matrix_path)

sg_instance = SoundGenerator(material_path, matrix_path)
sg_instance.setDuration(FLAGS.duration)
sg_instance.setSampRate(FLAGS.samprate)

force = np.zeros(sg_instance.getEigenLen())
for i in range(3):
    force[3*FLAGS.contactpoint + i] = FLAGS.force[i]

# print(force)

sg_instance.setForce(force)

print("[ INFO] Generating Modal Sound...")
sg_instance.genSound()

# sg_instance.playSound()
print("[ INFO] Saving Sound Data...")
sg_instance.saveSound()
sg_instance.saveEachMode()
