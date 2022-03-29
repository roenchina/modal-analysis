# import pyaudio
import numpy as np

# p = pyaudio.PyAudio()

fs = 44100       # sampling rate, Hz, must be integer
duration = 2.0   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
samples_1 = (np.sin(2*np.pi*np.arange(fs*duration)*440/fs)).astype(np.float32)
samples_2 = (np.sin(2*np.pi*np.arange(fs*duration)*800/fs)).astype(np.float32)


import sounddevice as sd
import time
sd.play(samples_1+samples_2, fs)
time.sleep(duration)
sd.stop()
# # for paFloat32 sample values must be in range [-1.0, 1.0]
# stream = p.open(format=pyaudio.paFloat32,
#                 channels=1,
#                 rate=fs,
#                 output=True)

# # play. May repeat with different volume values (if done interactively) 
# stream.write(samples_1)
# stream.write(samples_2)
# stream.write(samples_1+samples_2)

# stream.stop_stream()
# stream.close()

# p.terminate()