"""
==========================================================
                    POP Machine Learning
==========================================================

Influence of Photon Noise and Flat Field errors in the
performance of the Machine Learning Method
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


""" HELPER FUNCTIONS """

# ============================================================================== #
#                              PERFORMANCE EVALUATION                            #
# ============================================================================== #


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
    new_test_coef = transform_zemax_to_noll(test_coef, twisted=False)
    new_guessed_coef = transform_zemax_to_noll(guessed_coef, twisted)

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
        plt.scatter(range(N), initial_rms * wave_nom, c='blue', s=6, label='Initial')
        plt.scatter(range(N), residual_rms * wave_nom, c='red', s=6, label='Residual')
        plt.xlabel('Test PSF')
        plt.xlim([0, N])
        plt.ylim(bottom=0)
        plt.ylabel('RMS wavefront [nm]')
        plt.title(r'$\lambda=1.5$ $\mu$m (defocus: 0.20 waves)')
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
            x = np.linspace(-0.10, 0.10, 10)
            # plt.colorbar(ss)
            plt.plot(x, x, color='black', linestyle='--')
            title = zern_list[k]
            plt.title(title)
            plt.xlabel('True Value [waves]')
            plt.ylabel('Predicted Value [waves]')
            plt.xlim([-0.10, 0.10])
            plt.ylim([-0.10, 0.10])


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

def downsample_slicer_pixels(square_PSFs):
    """
    Raw PSF files sample the slice width with 2 pixels that can take different values
    This is not exactly true, as the detector pixels are elongated at the slicer,
    with only 1 true value covering 2 square pixels

    This functions fixes that issue by taking the average value pairwise
    :param array: PSF array
    :return:
    """

    n_psf, n_pix = square_PSFs.shape[0], square_PSFs.shape[-1]
    downsampled_PSFs = np.zeros_like(square_PSFs)
    flat_PSFs = np.empty((n_psf, 2 * n_pix * n_pix))
    for k in range(n_psf):
        for i in np.arange(1, n_pix-1, 2):
            # print(i)
            row_foc = square_PSFs[k, 0, i, :]
            next_row_foc = square_PSFs[k, 0, i+1, :]
            mean_row_foc = 0.5*(row_foc + next_row_foc)

            row_defoc = square_PSFs[k, 1, i, :]
            next_row_defoc = square_PSFs[k, 1, i+1, :]
            mean_row_defoc = 0.5*(row_defoc + next_row_defoc)

            downsampled_PSFs[k, 0, i, :] = mean_row_foc
            downsampled_PSFs[k, 0, i + 1, :] = mean_row_foc

            downsampled_PSFs[k, 1, i, :] = mean_row_defoc
            downsampled_PSFs[k, 1, i + 1, :] = mean_row_defoc

        flat_PSFs[k] = np.concatenate((downsampled_PSFs[k, 0].flatten(), downsampled_PSFs[k, 1].flatten()))

    return square_PSFs, downsampled_PSFs, flat_PSFs

# ============================================================================== #
#                                   REAL EFFECTS                                 #
# ============================================================================== #


def photon_noise(PSF_array, factor):
    """
    Rescales the PSF according to Poisson noise
    to simulate photon noise:

    Noisy_PSF = 1/factor * Poisson {factor * PSF}
    :param PSF_array: Raw PSFs with Peaks of around 1.0
    :param factor: controls the amount of photon noise, while keeping the peaks close to 1.0
    :return:

    NOTE: the 2 factor is there to ensure that the Poisson uses the rate for the
    SUM of pixel pairs (as the numpy array contains pairs of pixels of equal value)
    that in reality are integrated together
    """
    noisy_PSF = (np.random.poisson(lam = 2 * PSF_array * factor)) / (2 * factor)
    return noisy_PSF

def flat_field(PSF_array, sigma, show=False):
    """
    Applies a Flat Field error to all PSFs equally,
    including the nominal and defocus version
    """

    N_flat_pix = PSF_array.shape[1] // 2
    N_pix = int(np.sqrt(N_flat_pix))
    delta = sigma * np.sqrt(3.)
    a, b = 1 - delta, 1 + delta

    # flat_map = np.random.uniform(a, b, size=N_flat_pix)
    flat_map = np.random.normal(loc=1, scale=sigma, size=N_flat_pix)
    flat_map_dual = np.concatenate([flat_map, flat_map])
    flat_all_PSFS = flat_map_dual[np.newaxis, :]

    noisy_PSF = flat_all_PSFS * PSF_array

    if show:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(flat_map, histtype='step', color='Green')
        plt.xlabel('Flat field []')

        plt.subplot(1, 2, 2)
        im = flat_map.reshape((N_pix, N_pix))
        plt.imshow(im, cmap='bwr')
        plt.colorbar()
        plt.title(r'$\mathcal{N}(1, \sigma)$ with $\sigma=%.3f$' %sigma)
    return noisy_PSF


if __name__ == "__main__":

    path_zemax = os.path.join('POP', 'NYQUIST', 'FOC 0.15')

    # Random
    N_rand = 1000
    path_rand1 = os.path.join(path_zemax, 'RANDOM 0')
    zern_coefs_rand1 = np.load(os.path.join(path_rand1, 'rand_coef.npy'))
    PSFs_rand1 = load_files(path_rand1, N=N_rand, file_list=list_slices)

    path_rand2 = os.path.join(path_zemax, 'RANDOM 1')
    zern_coefs_rand2 = np.load(os.path.join(path_rand2, 'rand_coef.npy'))
    PSFs_rand2 = load_files(path_rand2, N=N_rand, file_list=list_slices)

    N_max = 2 * N_rand
    path_rand3 = os.path.join(path_zemax, 'RANDOM 2')
    zern_coefs_rand3 = np.load(os.path.join(path_rand3, 'rand_coef.npy'))
    PSFs_rand3 = load_files(path_rand3, N=N_max, file_list=list_slices)

    PSFs = np.concatenate((PSFs_rand1[0], PSFs_rand2[0], PSFs_rand3[0]), axis=0)
    PSFs_square = np.concatenate((PSFs_rand1[1], PSFs_rand2[1], PSFs_rand3[1]), axis=0)

    zern_coefs = np.concatenate((zern_coefs_rand1,
                                 zern_coefs_rand2, zern_coefs_rand3[:N_max]), axis=0)

    PEAK = np.max(PSFs_square[:, 0])

    PSFs /= PEAK
    PSFs_square /= PEAK

    PSFs_square, downPSFs_square, downPSFs_flat = downsample_slicer_pixels(PSFs_square)

    """ Photon Noise """
    factor = 25
    noisy_PSFs_square = photon_noise(downPSFs_square, factor)
    # Downsample again because the Poisson distribution splits the 2 pixels into separate values
    # and they should be the same even after the noise
    noisy_PSFs_square, downnoisy_PSFs_square, downnoisy_PSFs_flat = downsample_slicer_pixels(noisy_PSFs_square)

    plt.figure()
    plt.plot(np.max(downPSFs_square[:,0], axis=(1,2)))

    random_choice = rand_state.choice(2*N_rand + N_max, 2*N_rand + N_max - 20, replace=False)
    ext = 1.04
    slicer_extend = (-ext, ext, -ext, ext)
    width = 0.133
    i = 1
    for kk in random_choice[:5]:
        plt.figure()
        im_clean = np.concatenate((downPSFs_square[kk, 0], downPSFs_square[kk, 1]), axis=1)
        im_noisy = np.concatenate((downnoisy_PSFs_square[kk, 0], downnoisy_PSFs_square[kk, 1]), axis=1)
        im = np.concatenate((im_clean, im_noisy), axis=0)
        plt.imshow(im, extent=slicer_extend, cmap='viridis')
        plt.axhline(y=ext/2+width/2, color='white', linestyle='--', alpha=0.9)
        plt.axhline(y=ext/2-width / 2, color='white', linestyle='--', alpha=0.9)
        plt.axhline(y=ext/2+3*width/2, color='white', linestyle='--', alpha=0.7)
        plt.axhline(y=ext/2-3*width / 2, color='white', linestyle='--', alpha=0.7)
        plt.axhline(y=ext/2+5*width/2, color='white', linestyle='--', alpha=0.5)
        plt.axhline(y=ext/2-5*width / 2, color='white', linestyle='--', alpha=0.5)
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()

    plt.show()

    # ============================================================================== #
    #                                   PHOTON NOISE                                 #
    # ============================================================================== #

    """ (1) - Train with CLEAN PSFs, test with Photon noise """

    training, testing = generate_training_set(2*N_rand + N_max, 200, downPSFs_flat, zern_coefs, True)

    # low_training_noisy, low_coefs_noisy = train_with_noise(low_training[0], low_training[1], N_repeat=5)
    N_layer = (150, 100, 50)
    model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                         solver='adam', max_iter=N_iter, verbose=True,
                         batch_size='auto', shuffle=True, tol=1e-12,
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

    # Influence of photon noise
    N_phot = 10
    log_min, log_max = 1, 6
    f_list = np.logspace(log_min, log_max, N_phot)
    mu_rms_photon = np.zeros(N_phot)
    std_rms_photon = np.zeros(N_phot)
    for i, factor in enumerate(f_list):
        test_photon = photon_noise(testing[0], factor)
        test_photon_square = test_photon.reshape((test_photon.shape[0], 2, N_crop, N_crop))
        _a, _b, test_photon_flat = downsample_slicer_pixels(test_photon_square)

        guessed_photon = model.predict(X=test_photon_flat)
        _rms0, _rms = evaluate_wavefront_performance(N_zern, testing[1], guessed_photon,
                                                       zern_list=zern_list_low, show_predic=False)
        mu_rms_photon[i] = wave_nom * np.mean(_rms)
        std_rms_photon[i] = wave_nom * np.std(_rms)

    plt.figure()
    # Rescale by two because of the DUAL pixel situation
    plt.plot(2*f_list, mu_rms_photon, color='black')
    plt.fill_between(2*f_list, mu_rms_photon + std_rms_photon/2, mu_rms_photon - std_rms_photon/2,
                     facecolor='lightblue', alpha=0.7, label=r'RMS residual $\pm \sigma/2$')
    plt.axhline(y = wave_nom * np.mean(_rms0), color='black', linestyle='--', label='Initial RMS NCPA')
    plt.xscale('log')
    plt.xlim([10**log_min, 10**log_max])
    plt.ylim([0, 55])
    plt.xlabel('Photon count')
    plt.ylabel('RMS residual [nm]')
    plt.legend(loc=1)

    # ============================================================================== #

    """ (2) Resilience against Photon. Training with multiple noisy examples """

    training, testing = generate_training_set(2 * N_rand + N_max, 200, downPSFs_flat, zern_coefs, True)
    _clean, coeffs = training[0], training[1]

    N_noise = 3
    log_min, log_max = 2, 6
    f_list = np.logspace(log_min, log_max, N_phot)
    mu_rms_resil = np.zeros(N_phot)
    std_rms_resil = np.zeros(N_phot)
    for i, factor in enumerate(f_list):
        print('\n===========================')
        print(i)

        # Add some noise to the training
        noisy_list, noisy_coeffs = [], []
        for j in range(N_noise):
            noisy = photon_noise(_clean, factor)
            noisy_square = noisy.reshape((noisy.shape[0], 2, N_crop, N_crop))
            _a, _b, noisy_flat = downsample_slicer_pixels(noisy_square)
            noisy_list.append(noisy_flat)
            noisy_coeffs.append(coeffs)
        noisy_train = np.concatenate(noisy_list, axis=0)
        noisy_coeffs = np.concatenate(noisy_coeffs, axis=0)

        model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=False,
                             batch_size='auto', shuffle=True, tol=1e-12,
                             warm_start=True, alpha=1e-4, random_state=1234)
        model.fit(X=noisy_train, y=noisy_coeffs)

        # Noisy test
        test_photon = photon_noise(testing[0], factor)
        test_photon_square = test_photon.reshape((test_photon.shape[0], 2, N_crop, N_crop))
        _a, _b, test_photon_flat = downsample_slicer_pixels(test_photon_square)
        guessed_photon = model.predict(X=test_photon_flat)
        _rms0, _rms = evaluate_wavefront_performance(N_zern, testing[1], guessed_photon,
                                                       zern_list=zern_list_low, show_predic=False)
        mu_rms_resil[i] = wave_nom * np.mean(_rms)
        std_rms_resil[i] = wave_nom * np.std(_rms)

    plt.figure()
    plt.plot(2*f_list, mu_rms_photon, color='black', alpha=0.5)
    plt.fill_between(2*f_list, mu_rms_photon + std_rms_photon/2, mu_rms_photon - std_rms_photon/2,
                     facecolor='lightblue', alpha=0.7, label='Clean Training')
    plt.plot(2*f_list, mu_rms_resil, color='black', alpha=0.5)
    plt.fill_between(2*f_list, mu_rms_resil + std_rms_resil/2, mu_rms_resil - std_rms_resil/2,
                     facecolor='coral', alpha=0.7, label='Noisy Training')
    plt.axhline(y = wave_nom * np.mean(_rms0), color='black', linestyle='--', label='Initial RMS NCPA')
    plt.xscale('log')
    plt.xlim([2*10**log_min, 10**log_max])
    plt.ylim([0, 55])
    plt.xlabel('Photon count')
    plt.ylabel('RMS residual [nm]')
    plt.legend(title='RMS residual $\mu \pm \sigma/2$', loc=1)

    # ============================================================================== #
    #                                 FLAT FIELD ERRORS                              #
    # ============================================================================== #

    """ (1) Train on CLEAN PSFs """

    training, testing = generate_training_set(2*N_rand + N_max, 200, downPSFs_flat, zern_coefs, True)

    # low_training_noisy, low_coefs_noisy = train_with_noise(low_training[0], low_training[1], N_repeat=5)
    N_layer = (150, 100, 50)
    model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                         solver='adam', max_iter=N_iter, verbose=True,
                         batch_size='auto', shuffle=True, tol=1e-12,
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

    N_sigma = 10
    N_repeat = 20
    sigma_list = np.linspace(0, 0.15, N_sigma)[::-1]
    mu_rms_sigma = np.zeros(N_sigma)
    std_rms_sigma = np.zeros(N_sigma)
    for i, sigma in enumerate(sigma_list):
        print('\n-----------------------------')
        print(i)
        _a = []
        # Take the mean across many realizations of the Random Maps
        # Because some are 'unlucky' if they for instance reduce a lot the peak intensity
        # While other random flats do not affect the peak that much...
        for j in range(N_repeat):
            test_flat = flat_field(testing[0], sigma, False)

            guessed_flat = model.predict(X=test_flat)
            _rms0, _rms = evaluate_wavefront_performance(N_zern, testing[1], guessed_flat,
                                                           zern_list=zern_list_low, show_predic=False)
            # if i%2 == 0 and j==0:
            #     _rms0, _rms = evaluate_wavefront_performance(N_zern, testing[1], guessed_flat,
            #                                                  zern_list=zern_list_low, show_predic=True)
            #     print(sigma)
            #     plt.show()

            _a.append(_rms)
        a = np.array(_a)
        _rms = np.mean(a, axis=0)
        print(_rms.shape)
        mu_rms_sigma[i] = wave_nom * np.mean(_rms)
        std_rms_sigma[i] = wave_nom * np.std(a)

    plt.figure()
    plt.plot(sigma_list, mu_rms_sigma, color='black')
    plt.fill_between(sigma_list, mu_rms_sigma + std_rms_sigma/2, mu_rms_sigma - std_rms_sigma/2,
                     facecolor='lightgreen', alpha=0.7, label=r'RMS residual $\pm \sigma/2$')
    plt.axhline(y = wave_nom * np.mean(_rms0), color='black', linestyle='--', label='Initial RMS NCPA')
    # plt.xscale('log')
    plt.xlim([sigma_list.min(), sigma_list.max()])
    plt.ylim([0, 40])
    plt.xlim([0, 0.15])
    plt.xlabel(r'Flat field uncerainty $\sigma$ [ ]')
    plt.ylabel('RMS residual [nm]')
    plt.legend(loc=2)




