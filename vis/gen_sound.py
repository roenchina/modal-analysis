import os
import numpy as np


# return ksi, omegas, omega_d, freqs
def calFreqs(evals, beta, alpha):
    num_modes = len(evals)
    valid_map = np.zeros(num_modes)
    omegas = np.zeros(num_modes)
    omega_d = np.zeros(num_modes)
    ksi = np.zeros(num_modes)
    freqs = np.zeros(num_modes)

    for i in range(num_modes):
        if (evals[i] < 0):
            valid_map[i] = 0
            print('evals < 0 at ', i)
            continue

        omegas[i] = np.sqrt(evals[i])

        if (omegas[i] < 100 or omegas[i] > 2e5):
            print(f'omegas[{i}] = {omegas[i]} is out of 20hz 20000hz range')
            valid_map[i] = 0
            continue
        
        ksi[i] = (beta + alpha * evals[i]) / 2 / omegas[i]
        scale = 1 - ksi[i] * ksi[i]
        if (scale < 0 ):
            valid_map[i] = 0
            print('1 - ksi^2 < 0 at', i)
            continue

        omega_d[i] = omegas[i] * np.sqrt(scale)
        freqs[i] = 0.5 * omega_d[i] / np.pi
    return ksi, omegas, omega_d, freqs

# return mode_sample, samples
def genSound(ksi, omegas, omega_d, scales, fs, duration):
    num_modes = len(ksi)

    time_slot = np.arange(fs * duration) / fs

    mode_sample = np.zeros((num_modes, len(time_slot)))
    samples = np.zeros(len(time_slot))

    for i in range(num_modes):
        if(omega_d[i] != 0):
            amplitude = np.exp(time_slot * (-1) * ksi[i] * omegas[i]) * abs(scales[i]) / omega_d[i]
            mode_sample[i] = (np.sin(omega_d[i] * time_slot ) * amplitude).astype(np.float32)
            samples += mode_sample[i]
    return mode_sample, samples

def saveSound(filename, fs, samples):
    from scipy.io.wavfile import write
    tmp_samples = samples * 1e6
    write(filename, fs, tmp_samples)


###### Load #####
filename = 'T' + '0'
# obj_dir = '../DATA/obj/test'
gt_dir = '../DATA/eigen'
pred_dir = '../DATA/test_results'

# obj_file = os.path.join(obj_dir, filename+'.obj')
gt_file = os.path.join(gt_dir, filename, 'eigen.npz')
pred_file = os.path.join(pred_dir, filename+'.npy')

gt_read = np.load(gt_file)
gt = gt_read['evals']
evecs = gt_read['evecs']
pred = np.load(pred_file)
pred = pred * 1e9


###### Cal Freqs #####
pd_ksi, pd_omegas, pd_omega_d, pd_freqs = calFreqs(pred, beta=5.0, alpha=1e-7)
gt_ksi, gt_omegas, gt_omega_d, gt_freqs = calFreqs(gt, beta=5.0, alpha=1e-7)


###### Set Force #####

contact_pos = 179
contact_force = [0, 0, -10]
scales = np.zeros(50)
for dir in range(3):
    scales += contact_force[dir] * evecs[3*contact_pos + dir]

# print(scales)

# a = input()
# scales = np.array([-0.02614267,  0.26238343,  0.29445853,  0.07136185,  0.03069723,
#        -0.10452014,  0.26181166, -0.08404987, -0.31661772, -0.18206124,
#        -0.00725318,  0.2248846 , -0.19443599, -0.02147535,  0.20651412,
#         0.31736385,  0.09827227, -0.27085757, -0.08045651,  0.05799157,
#         0.11239805,  0.26280163,  0.03608398, -0.29905944, -0.24821235,
#        -0.1274047 ,  0.27859512, -0.04861437, -0.01590443,  0.00476769,
#        -0.30062828, -0.10974673,  0.31553769,  0.19965295, -0.11703281,
#        -0.21711631,  0.14274808, -0.00605103, -0.05636535, -0.33345507,
#        -0.12275211,  0.2758167 , -0.22557653, -0.10211547, -0.24294398,
#        -0.19374149, -0.00876933, -0.19204382,  0.26614045, -0.15465599])


###### Save as file #####
duration = 3
fs = 44100
gt_mode_sample, gt_sample = genSound(gt_ksi, gt_omegas, gt_omega_d, scales, fs, duration)
pd_mode_sample, pd_sample = genSound(pd_ksi, pd_omegas, pd_omega_d, scales, fs, duration)

saveSound(os.path.join('./', filename+'_gt.wav'), fs, gt_sample)
saveSound(os.path.join('./', filename+'_test.wav'), fs, gt_sample)