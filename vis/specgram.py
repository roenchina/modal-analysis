import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load a wav file
y, sr = librosa.load('./EXP/T353/T353-p190-f0.0.-1_test.wav', sr=None)
# extract mel spectrogram feature
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
# convert to log scale
logmelspec = librosa.power_to_db(melspec)
# plot mel spectrogram
plt.figure()
librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.title('Beat wavform')
plt.show()