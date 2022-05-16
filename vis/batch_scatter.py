import os
import numpy as np
import matplotlib.pyplot as plt

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
            # print('evals < 0 at ', i)
            continue

        omegas[i] = np.sqrt(evals[i])

        if (omegas[i] < 100 or omegas[i] > 2e5):
            # print(f'omegas[{i}] = {omegas[i]} is out of 20hz 20000hz range')
            valid_map[i] = 0
            continue
        
        ksi[i] = (beta + alpha * evals[i]) / 2 / omegas[i]
        scale = 1 - ksi[i] * ksi[i]
        if (scale < 0 ):
            valid_map[i] = 0
            # print('1 - ksi^2 < 0 at', i)
            continue

        omega_d[i] = omegas[i] * np.sqrt(scale)
        freqs[i] = 0.5 * omega_d[i] / np.pi
    return ksi, omegas, omega_d, freqs

BASE_DIR = '../DATA/test_results'
gt_dir = '../DATA/eigen'
pred_dir = '../DATA/test_results'
##### CONFIG ######
DOT_SIZE = 60
COLOR = 'steelblue'
MARKER = 'o'
ALPHA = 0.8
LINE_WIDTH = 0.3
EDGE_COLOR = 'white'

for root, dirs, files in os.walk(BASE_DIR, topdown=False):
    for name in files:
        split_res = os.path.splitext(os.path.basename(name))
        filename_ = split_res[0]
        dirname_ = root
        extension_ = split_res[-1]
        if(extension_ == '.npy'):
            gt_file = os.path.join(gt_dir, filename_, 'eigen.npz')
            pred_file = os.path.join(pred_dir, filename_+'.npy')
            outputfigpath_ = os.path.join(dirname_, filename_+'.png')


            gt_read = np.load(gt_file)
            gt = gt_read['evals']
            pred = np.load(pred_file) * 1e9

            gt_ksi, gt_omegas, gt_omega_d, gt_freqs = calFreqs(gt, beta=5.0, alpha=1e-7)
            pd_ksi, pd_omegas, pd_omega_d, pd_freqs = calFreqs(pred, beta=5.0, alpha=1e-7)

            plt.style.use('seaborn')

            plt.figure(figsize=(6,6))
            plt.scatter(
                        gt_freqs[3:50],
                        pd_freqs[3:50],
                        s = DOT_SIZE,
                        c = COLOR,
                        marker = MARKER,
                        alpha = ALPHA,
                        linewidths = LINE_WIDTH,
                        edgecolors = EDGE_COLOR
                        )

            plt.xlabel('Ground Truth Freq')
            plt.ylabel('ModalNet Predition')
            plt.title('ModalNet')
            ax=plt.gca()
            ax.set_aspect(1)

            plt.savefig(outputfigpath_)
            # plt.show()
            plt.close()
