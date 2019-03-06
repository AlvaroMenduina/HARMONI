"""
==========================================================
                    POP Machine Learning
==========================================================

Cleaned version of the Machine Learning method for NCPA in
the .

PSF is Nyquist sampled

Approach: DEFOCUS

Method: Iterative. 1 LOW network and 1 HIGH network
"""

import os
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import zern_core as zern
from pyzdde.zdde import readBeamFile
from sklearn.neural_network import MLPRegressor
import matplotlib.cm as cm

""" PARAMETERS """
# Wavelength
wave_nom = 1500          # nmn
defocus = 0.20

# Slices
list_slices = [17, 19, 21, 55, 57]
i_central = 19

# POP arrays - Nyquist sampled PSF
x_size = 2.08           # Physical size of array at Image Plane
N_pix = 32              # Number of pixels in the Zemax BFL
N_crop = 16             # Crop to a smaller region around the PSF
min_pix = N_pix//2 - N_crop//2
max_pix = N_pix//2 + N_crop//2
extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]
xc = np.linspace(-x_size / 2, x_size / 2, 10)

# Machine Learning parameters

N_iter = 150
N_epochs = 4
random_train = True
rand_state = RandomState(1234)

n_test = 100        # Number of samples to check performance

# Zernikes: Defocus, Astigmatism x2, Coma x2
zern_list_low = ['Defocus', 'Astig_X', 'Astig_Y', 'Coma_X', 'Coma_Y']
N_zern = 5              # Number of Zernike aberrations
sampling = 5            # Points per aberration: [-0.2, -0.1, 0.0, 0.1, 0.2]
N_train = sampling ** N_zern

# Sampling
z_min = -0.2
z_max = 0.2
delta = (z_max - z_min) / (sampling - 1)

""" HELPER FUNCTIONS """




def reshape_flattened(array, pix=N_crop):
    """
    It takes a PSF composed of a flattened nominal
    and a flattened defocused PSF and rebuilds them
    and concatenates them into a 2D image
    """
    n_flat = array.shape[0]
    nominal = array[:n_flat//2].reshape((pix, pix))
    defocus = array[n_flat//2:].reshape((pix, pix))
    return np.concatenate((nominal, defocus), axis=1)

def generate_sampling(sampling, N_zern, delta, start=0.0):
    """
    Mimics the sampling of the Zernike coefficients from Zemax
    It returns an array of [N_train, N_zern] that matches the PSFs
    generated with POP in the ZPL Macro loops
    """
    coefs = np.empty((sampling**N_zern, N_zern))
    for i in range(N_zern):
        n = sampling ** (N_zern - (i + 1))
        a = start * np.ones(n)
        for j in range(sampling - 1):
            b = (start + (j + 1) * delta) * np.ones(n)
            a = np.concatenate((a, b))
        index = np.tile(a, sampling ** i)
        coefs[:, i] = index
    return coefs

def generate_training_set(N_train, n_test, flat_PSFs, zern_coefs, random_train=True):
    """
    Splits the PSF dataset into a TRAINING set and a TEST set

    :param random_train: if TRUE, it will select a random TRAINING set of size N_train - n_test
    :return: (TRAINING set, targets) & (TEST set, true values)
    """

    if random_train == True:
        random_choice = rand_state.choice(N_train, N_train - n_test, replace=False)
        test_choice = np.delete(np.arange(N_train), random_choice)

        training_set = flat_PSFs[random_choice, :]
        targets = zern_coefs[random_choice]
        test_set = flat_PSFs[test_choice, :]
        test_coef = zern_coefs[test_choice, :]

    else:   # Just exclude the last N_test to check performance
        training_set = flat_PSFs[:N_train - n_test, :]
        targets = zern_coefs[:N_train - n_test, :]
        test_set = flat_PSFs[N_train - n_test:N_train, :]
        test_coef = zern_coefs[N_train - n_test:N_train, :]

    return (training_set, targets), (test_set, test_coef)

def transform_zemax_to_noll(coef, twisted=False):
    """
    Rearranges the order of the coefficients from
    Zemax [Defocus, Astig X, Astig Y, Coma X, Coma Y, Spherical, Trefoil X, Trefoil Y]
    Zern [Astig X, Defocus, Astig Y, Trefoil X, Coma X, Coma Y, Trefoil Y]
    """
    N, N_z = coef.shape
    print(N_z)
    if not twisted and N_z == 5:
        new_coef = np.zeros((N, 7))
        new_coef[:, 0] = coef[:, 1]
        new_coef[:, 1] = coef[:, 0]
        new_coef[:, 2] = coef[:, 2]
        new_coef[:, 4] = coef[:, 3]
        new_coef[:, 5] = coef[:, 4]

    return new_coef

def evaluate_wavefront_performance(N_zern, test_coef, guessed_coef, zern_list, twisted=False, show_predic=False):
    """
    Evaluates the performance of the ML method regarding the final
    RMS wavefront error. Compares the initial RMS NCPA and the residual
    after correction
    """

    # Transform the ordering to match the Zernike matrix
    new_test_coef = 2*np.pi*transform_zemax_to_noll(test_coef, twisted=False)
    new_guessed_coef = 2*np.pi*transform_zemax_to_noll(guessed_coef, twisted)

    x = np.linspace(-1, 1, 512, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho, theta = np.sqrt(xx**2 + yy**2), np.arctan2(xx, yy)
    pupil = rho <= 1.0
    rho, theta = rho[pupil], theta[pupil]
    zernike = zern.ZernikeNaive(mask=pupil)
    _phase = zernike(coef=np.zeros(new_test_coef.shape[1] + 3), rho=rho, theta=theta, normalize_noll=False, mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix[:,3:]   # remove the piston and tilts
    H_matrix = zern.invert_model_matrix(H_flat, pupil)
    # print(H_flat.shape)

    # Elliptical mask
    ellip_mask = (xx / 0.5)**2 + (yy / 1.)**2 <= 1.0

    H_flat = H_matrix[ellip_mask]
    # print(H_flat.shape)

    N = test_coef.shape[0]
    initial_rms = np.zeros(N)
    residual_rms = np.zeros(N)

    for k in range(N):
        phase = np.dot(H_flat, new_test_coef[k])
        residual_phase = phase - np.dot(H_flat, new_guessed_coef[k])
        before, after = np.std(phase), np.std(residual_phase)
        initial_rms[k] = before
        residual_rms[k] = after

    average_initial_rms = np.mean(initial_rms)
    average_residual_rms = np.mean(residual_rms)
    improvement = (average_initial_rms - average_residual_rms) / average_initial_rms * 100

    print('\nWAVEFRONT PERFORMANCE DATA')
    print('\nNumber of samples in TEST dataset: %d' %N)
    print('Average INITIAL RMS: %.3f waves (%.1f nm @1.5um)' %(average_initial_rms, average_initial_rms*wave_nom))
    print('Average RESIDUAL RMS: %.3f waves (%.1f nm @1.5um)' %(average_residual_rms, average_residual_rms*wave_nom))
    print('Improvement: %.2f percent' %improvement)

    if show_predic == True:

        plt.figure()
        plt.scatter(range(N), initial_rms * wave_nom, c='blue', s=8, label='Initial')
        plt.scatter(range(N), residual_rms * wave_nom, c='red', s=8, label='Residual')
        plt.xlabel('Test PSF')
        plt.title('RMS wavefront [nm]')
        plt.legend()

        N_ok = (np.argwhere(residual_rms * wave_nom < 100)).shape[0]
        plt.figure()
        plt.scatter(initial_rms * wave_nom, residual_rms * wave_nom, c='blue', s=8)
        plt.axhline(y=100, linestyle='--')
        plt.xlabel('Initial RMS [nm]')
        plt.ylabel('Residual RMS [nm]')
        plt.title('%d / %d cases with RMS < 100 nm' %(N_ok, N))
        plt.ylim(bottom=0)

        plt.figure()
        n_bins = 20
        for k in range(N_zern):
            guess = guessed_coef[:, k]
            coef = test_coef[:, k]
            residual = coef - guess
            mu, s2 = np.mean(residual), (np.std(residual))
            label = zern_list[k] + r'  ($\mu$=%.3f, $\sigma$=%.2f)' %(mu, s2)
            plt.hist(residual, histtype='step', label=label)
        plt.legend(title=r'Residual aberrations [waves]', loc=2)
        plt.xlabel(r'Residual [waves]')
        plt.xlim([-0.075, 0.075])

        for k in range(N_zern):
            guess = guessed_coef[:, k]
            coef = test_coef[:, k]

            colors = wave_nom * residual_rms
            colors -= colors.min()
            colors /= colors.max()
            colors = cm.rainbow(colors)

            plt.figure()
            ss = plt.scatter(coef, guess, c=colors, s=20)
            x = np.linspace(coef.min(), coef.max(), 10)
            # plt.colorbar(ss)
            plt.plot(x, x, color='black', linestyle='--')
            title = zern_list[k] + '  (N_train=%d, N_test=%d)' % (N_train - N, N)
            plt.title(title)
            plt.xlabel('True Value [waves]')
            plt.ylabel('Predicted Value [waves]')


    return initial_rms, residual_rms

# ============================================================================== #
#                                ZEMAX INTERFACE                                 #
# ============================================================================== #

def read_beam_file(file_name):
    """
    Reads a Zemax Beam File and returns the Irradiance
    of the Magnetic field E
    """
    beamData = readBeamFile(file_name)
    (version, (nx, ny), ispol, units, (dx, dy), (zposition_x, zposition_y),
     (rayleigh_x, rayleigh_y), (waist_x, waist_y), lamda, index, re, se,
     (x_matrix, y_matrix), (Ex_real, Ex_imag, Ey_real, Ey_imag)) = beamData

    E_real = np.array([Ex_real, Ey_real])
    E_imag = np.array([Ex_imag, Ey_imag])

    re = np.linalg.norm(E_real, axis=0)
    im = np.linalg.norm(E_imag, axis=0)

    irradiance = (re ** 2 + im ** 2).T
    power = np.sum(irradiance)
    print('Total Power: ', power)
    return (nx, ny), (dx, dy), irradiance, power

def read_all_zemax_files(path_zemax, name_convention, file_list):
    """
    Goes through the ZBF Zemax Beam Files of all Slices and
    extracts the beam information (X_size, Y_size) etc
    as well as the Irradiance distribution
    """
    info, data, powers = [], [], []

    for k in file_list:
        print('\n======================================')

        if k < 10:
            file_id = name_convention + ' ' + str(k) + '_POP.ZBF'
        else:
            file_id = name_convention + str(k) + '_POP.ZBF'
        file_name = os.path.join(path_zemax, file_id)

        print('Reading Beam File: ', file_id)

        NM, deltas, beam_data, power = read_beam_file(file_name)
        Dx, Dy = NM[0] * deltas[0], NM[1] * deltas[1]
        info.append([k, Dx, Dy])
        data.append(beam_data)
        powers.append(power)

    beam_info = np.array(info)
    irradiance_values = np.array(data)
    powers = np.array(powers)

    return beam_info, irradiance_values, powers


def load_files(path, file_list, N=N_train):
    """
    Loads the Zemax beam files, constructs the PSFs
    and normalizes everything by the intensity of the PSF
    at i_norm (the Nominal PSF)
    """

    pop_slicer_nom = POP_Slicer()
    pop_slicer_foc = POP_Slicer()

    flat_PSFs = np.empty((N, 2 * N_crop * N_crop))
    PSFs = np.empty((N, 2, N_crop, N_crop))

    for k in range(N):
        if k < 10:
            # We have to adjust for the ZBF format. Before 10 it adds a space []3
            name_nominal = 'IFU_TopAB_HARMONI_light' + '% d_' % k
            name_defocus = 'IFU_TopAB_HARMONI_light' + '% d_FOC_' % k
        else:
            name_nominal = 'IFU_TopAB_HARMONI_light' + '%d_' % k
            name_defocus = 'IFU_TopAB_HARMONI_light' + '%d_FOC_' % k

        pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
        slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

        pop_slicer_foc.get_zemax_files(path, name_defocus, file_list)
        slicers_foc = np.sum(pop_slicer_foc.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

        PSFs[k, 0, :, :], PSFs[k, 1, :, :] = slicers_nom, slicers_foc
        flat_PSFs[k, :] = np.concatenate((slicers_nom.flatten(), slicers_foc.flatten()))

    return [flat_PSFs, PSFs]

class POP_Slicer(object):
    """
    Physical Optics Propagation (POP) analysis of an Image Slicer
    """
    def __init__(self):
        pass

    def get_zemax_files(self, zemax_path, name_convention, file_list):
        _info, _data, _power = read_all_zemax_files(zemax_path, name_convention, file_list)
        self.beam_info = _info
        self.beam_data = _data
        self.powers = _power



if __name__ == "__main__":

    path_zemax = os.path.join('POP', 'NYQUIST')

    # Random
    N_rand = 1000
    path_rand1 = os.path.join(path_zemax, 'RANDOM')
    zern_coefs_rand1 = np.load(os.path.join(path_rand1, 'rand_coef.npy'))
    PSFs_rand1 = load_files(path_rand1, N=N_rand, file_list=list_slices)

    path_rand2 = os.path.join(path_zemax, 'RANDOM_1')
    zern_coefs_rand2 = np.load(os.path.join(path_rand2, 'rand_coef.npy'))
    PSFs_rand2 = load_files(path_rand2, N=N_rand, file_list=list_slices)

    path_rand3 = os.path.join(path_zemax, 'RANDOM_2')
    zern_coefs_rand3 = np.load(os.path.join(path_rand3, 'rand_coef.npy'))
    PSFs_rand3 = load_files(path_rand3, N=600, file_list=list_slices)

    PSFs = np.concatenate((PSFs_rand1[0], PSFs_rand2[0], PSFs_rand3[0]), axis=0)
    PSFs_square = np.concatenate((PSFs_rand1[1], PSFs_rand2[1], PSFs_rand3[1]), axis=0)

    zern_coefs = np.concatenate((zern_coefs_rand1, zern_coefs_rand2, zern_coefs_rand3[:600]), axis=0)
    PEAK = np.max(PSFs_square[:, 0])

    PSFs /= PEAK
    PSFs_square /= PEAK

    plt.figure()
    plt.plot(np.max(PSFs_square[:,0], axis=(1,2)))

    random_choice = rand_state.choice(2*N_rand, 2*N_rand - n_test, replace=False)
    for kk in random_choice[:5]:
        plt.figure()
        plt.imshow(np.concatenate((PSFs_square[kk, 0], PSFs_square[kk, 1]), axis=1), cmap='viridis')
        # plt.title(title)
        plt.colorbar(orientation='horizontal')
        plt.axis('off')
        plt.tight_layout()



    # ============================================================================== #
    #                                TRAIN THE MODEL                                 #
    # ============================================================================== #

    # PSFs = PSFs_rand1[0]
    # zern_coefs = zern_coefs_rand1
    # PSFs /= PSFs.max()
    training, testing = generate_training_set(2*N_rand + 600, 100, PSFs, zern_coefs, True)

    # low_training_noisy, low_coefs_noisy = train_with_noise(low_training[0], low_training[1], N_repeat=5)
    N_layer = (150, 100, 50)
    model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                         solver='adam', max_iter=N_iter, verbose=True,
                         batch_size='auto', shuffle=True, tol=1e-9,
                         warm_start=True, alpha=1e-4, random_state=1234)
    model.fit(X=training[0], y=training[1])

    guessed = model.predict(X=testing[0])
    print("\nLOW model guesses:")
    print(guessed[:5])
    print("\nTrue Values")
    print(testing[1][:5])

    print('\nModel:')
    rms0, rms = evaluate_wavefront_performance(N_zern, testing[1], guessed,
                                                   zern_list=zern_list_low, show_predic=True)

    # ============================================================================== #
    #                                ENSEMBLE APPROACH                               #
    # ============================================================================== #

    """ (1) Model with SAME PARAMETERS """

    training, testing = generate_training_set(2*N_rand + 600, 100, PSFs, zern_coefs, True)
    guesses = []
    errors, stds = [], []

    for i in range(30):
        print(i)
        model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=False,
                             batch_size='auto', shuffle=True, tol=1e-9,
                             warm_start=True, alpha=1e-4)
        model.fit(X=training[0], y=training[1])

        guessed = model.predict(X=testing[0])
        guesses.append(guessed)

        rms0, rms = evaluate_wavefront_performance(N_zern, testing[1], guessed,
                                                   zern_list=zern_list_low, show_predic=False)
        errors.append(wave_nom * np.mean(rms))
        stds.append(wave_nom * np.std(rms))

    guess = np.array(guesses)
    y, erry = [], []
    # As a function of the N_comb
    N_comb = np.arange(2, guess.shape[0])
    for i in N_comb:
        g = guess[:i]
        print(g.shape)
        rms0, ens_rms = evaluate_wavefront_performance(N_zern, testing[1], np.mean(g, axis=0),
                                                       zern_list=zern_list_low, show_predic=False)
        ens_mu, ens_std = wave_nom * np.mean(ens_rms), wave_nom * np.std(ens_rms)
        y.append(ens_mu)
        erry.append(ens_std)

    # Final ENSEMBLE estimation
    rms0, ens_rms = evaluate_wavefront_performance(N_zern, testing[1], np.mean(g, axis=0),
                                                   zern_list=zern_list_low, show_predic=True)

    plt.figure()
    plt.scatter(N_comb, y)
    plt.xlabel('N combined predictions')
    plt.ylabel('RMS residual')


    # ========================
    # Aberrations from -0.20 to 0.20 waves
    path_nom = os.path.join(path_zemax, '-20_20')
    zern_coefs_nom = generate_sampling(sampling, N_zern, delta, start=z_min)
    i_nom = np.argwhere(np.linalg.norm(zern_coefs_nom, axis=-1) == 0.0)
    PSFs_nom = load_files(path_nom, N=N_train, file_list=list_slices)
    PEAK = np.max(PSFs_nom[1][i_nom, 0])
    PSFs_nom[0] /= PEAK
    PSFs_nom[1] /= PEAK

    # Aberrations from -0.15 to 0.15 waves
    path_15 = os.path.join(path_zemax, '-15_15')
    zern_coefs_15 = generate_sampling(4, N_zern, delta, start=-0.15)
    PSFs_15 = load_files(path_15, N=4**N_zern, file_list=list_slices)
    PSFs_15[0] /= PEAK
    PSFs_15[1] /= PEAK

    # Aberrations from -0.10 to 0.10 waves
    nn = 1713
    path_10 = os.path.join(path_zemax, '-10_10')
    zern_coefs_10 = generate_sampling(5, N_zern, delta, start=-10)
    zern_coefs_10 = zern_coefs_10[:nn, :]
    PSFs_10 = load_files(path_10, N=nn-1, file_list=list_slices)
    PSFs_10[0] /= PEAK
    PSFs_10[1] /= PEAK

    # Aberrations from -0.075 to 0.075 waves
    path_075 = os.path.join(path_zemax, '-075_075')
    zern_coefs_075 = generate_sampling(4, N_zern, 0.05, start=-0.075)
    PSFs_075 = load_files(path_075, N=4**N_zern, file_list=list_slices)
    PSFs_075[0] /= PEAK
    PSFs_075[1] /= PEAK

    # Concatenate them
    PSFs = np.concatenate((PSFs_nom[0], PSFs_15[0], PSFs_075[0]), axis=0)
    zern_coefs = np.concatenate((zern_coefs_nom, zern_coefs_15, zern_coefs_075), axis=0)

    N_total = PSFs.shape[0]

    random_choice = rand_state.choice(N_train, N_train - n_test, replace=False)
    for kk in random_choice[:5]:
        plt.figure()
        plt.imshow(np.concatenate((PSFs_nom[1][kk, 0], PSFs_nom[1][kk, 1]), axis=1), cmap='viridis')
        # plt.title(title)
        plt.colorbar(orientation='horizontal')
        plt.axis('off')
        plt.tight_layout()

    N_PSF = 2000
    a_min, a_max = -0.05, 0.05
    rand_coef = np.random.uniform(a_min/2, a_max/2, size=(N_PSF, N_zern))
    np.save(os.path.join(path_zemax, 'rand_coef'), rand_coef)
    np.savetxt(os.path.join(path_zemax, 'rand_coef.txt'), rand_coef, fmt='%.5f')







    """ (2) Model with different PARAMETERS """
    guesses = []
    errors, stds = [], []
    training, testing = generate_training_set(N_total, 200, PSFs, zern_coefs, True)

    alfa = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    states = [1234, 1435, 6563, 1342, 9834, 1932]
    for i in range(6):

        model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9,
                             warm_start=True, alpha=alfa[i], random_state=states)
        model.fit(X=training[0], y=training[1])