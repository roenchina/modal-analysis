import os
from posixpath import dirname
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy

BASE_DIR = './EXP/3/selected'

for root, dirs, files in os.walk(BASE_DIR, topdown=False):
    for name in files:
        split_res = os.path.splitext(os.path.basename(name))
        filename_ = split_res[0]
        dirname_ = root
        extension_ = split_res[-1]
        if(extension_ == '.wav'):
            print(f'[ PROCESSING] at:{dirname_}, filename: {filename_}')
            filepath_ = os.path.join(dirname_, name)
            outputfigpath_ = os.path.join(dirname_, filename_+'.png')

            y, sr = librosa.load(filepath_, sr=None, duration=0.5)
            plt.figure(figsize=(8, 8))

            plt.subplot(2,1,1)
            plt.ylim((-500, 500))
            # plt.xticks([])
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            librosa.display.waveshow(y[0:11025], sr=sr)
            plt.ylabel('Amplitude')

            plt.subplot(2,1,2)

            Pxx, freqs, bins, im = plt.specgram(y, NFFT=1024, Fs=sr)
            plt.axis((None, None, 0, 20000))

            # plt.colorbar(format='%+2.0f dB')
            plt.ylabel('Mel Scale')

            plt.savefig(outputfigpath_)
            # plt.show()
            plt.close()