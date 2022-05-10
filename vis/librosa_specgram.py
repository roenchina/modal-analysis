import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


file_name = './EXP_srtp/_T150 Wood s10/T150-p196_test.wav'
y, sr = librosa.load(file_name, sr=None, duration=1)

plt.figure(figsize=(8, 8))

plt.subplot(2,1,1)
plt.ylim((-500, 500))
# plt.xticks([])
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
librosa.display.waveshow(y[0:22050], sr=sr)

plt.subplot(2,1,2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, x_axis='s', y_axis='log')
# plt.colorbar(format='%+2.0f dB')

plt.show()

