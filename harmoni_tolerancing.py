"""
====================================================================================================================
                    POP Machine Learning
====================================================================================================================

Analysis of the influence of uncertainties in the "as-built" instrument model. There will be random
misalignments of surfaces that will make the instrument differ from the Zemax model.

That will modify the PSFs slightly and thus it will have an impact on the performance of the Machine Learning method

Initial Analysis
- Train on POP simulations with perfect conditions
- Generate a test set for which the FIELD SPLITTER is slightly 'displaced' out of focus
    randomly selecting a displacement from a Uniform Distribution
- Investigate how the displacement affects the final RMS value

Future Analysis
- Enhance resilience by generating multiple TRAINING SETS with the SAME aberration
but different displacements
- Allow the network to ESTIMATE that displacement
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
        plt.scatter(range(N), initial_rms * wave_nom, c='blue', s=3, label='Initial')
        plt.scatter(range(N), residual_rms * wave_nom, c='red', s=3, label='Residual')
        plt.xlabel('Test PSF')
        plt.xlim([0, N])
        plt.ylim(bottom=0)
        plt.ylabel('RMS wavefront [nm]')
        plt.title(r'$\lambda=1.5$ $\mu$m (defocus: 0.20 waves)')
        plt.legend()

        N_ok = (np.argwhere(residual_rms * wave_nom < 100)).shape[0]
        # plt.figure()
        # plt.scatter(initial_rms * wave_nom, residual_rms * wave_nom, c='blue', s=8)
        # plt.axhline(y=100, linestyle='--')
        # plt.xlabel('Initial RMS [nm]')
        # plt.ylabel('Residual RMS [nm]')
        # plt.title('%d / %d cases with RMS < 100 nm' %(N_ok, N))
        # plt.ylim(bottom=0)
        #
        # plt.figure()
        # for k in range(N_zern):
        #     guess = guessed_coef[:, k]
        #     coef = test_coef[:, k]
        #     residual = coef - guess
        #     mu, s2 = np.mean(residual), (np.std(residual))
        #     label = zern_list[k] + r'  ($\mu$=%.3f, $\sigma$=%.2f)' %(mu, s2)
        #     plt.hist(residual, histtype='step', label=label)
        # plt.legend(title=r'Residual aberrations [waves]', loc=2)
        # plt.xlabel(r'Residual [waves]')
        # plt.xlim([-0.075, 0.075])

        for k in range(N_zern):
            guess = guessed_coef[:, k]
            coef = test_coef[:, k]

            colors = wave_nom * residual_rms
            colors -= colors.min()
            colors /= colors.max()
            colors = cm.rainbow(colors)

            plt.figure()
            ss = plt.scatter(coef, guess, c=colors, s=5)
            x = np.linspace(-0.15, 0.15, 10)
            # plt.colorbar(ss)
            plt.plot(x, x, color='black', linestyle='--')
            title = zern_list[k]
            plt.title(title)
            plt.xlabel('True Value [waves]')
            plt.ylabel('Predicted Value [waves]')
            plt.xlim([-0.075, 0.075])
            plt.ylim([-0.075, 0.075])


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


if __name__ == "__main__":

    # ============================================================================== #

    # Generate the Zernike coefficients and RANDOM displacements

    N_PSF = 2500
    a_min, a_max = -0.10, 0.10
    disp = 10
    rand_coef = np.random.uniform(a_min/2, a_max/2, size=(N_PSF, N_zern))
    random_disp = np.random.uniform(0, disp, size=(N_PSF, 1))
    coef = np.concatenate((rand_coef, random_disp), axis=-1)
    path_test = os.path.join('POP', 'NYQUIST')
    np.save(os.path.join(path_test, 'disp_coef'), coef)
    np.savetxt(os.path.join(path_test, 'disp_coef.txt'), coef, fmt='%.5f')

    # ============================================================================== #
    #                                   TRAINING SET                                 #
    # ============================================================================== #

    path_train = os.path.join('POP', 'NYQUIST', 'IRREGULARITIES', 'TRAIN')

    N_rand = 1000
    path_rand1 = os.path.join(path_train, 'RANDOM 0')
    zern_coefs_rand1 = np.load(os.path.join(path_rand1, 'rand_coef.npy'))
    PSFs_rand1 = load_files(path_rand1, N=N_rand, file_list=list_slices)

    path_rand2 = os.path.join(path_train, 'RANDOM 1')
    zern_coefs_rand2 = np.load(os.path.join(path_rand2, 'rand_coef.npy'))
    PSFs_rand2 = load_files(path_rand2, N=N_rand, file_list=list_slices)

    N_max = 2 * N_rand
    path_rand3 = os.path.join(path_train, 'RANDOM 2')
    zern_coefs_rand3 = np.load(os.path.join(path_rand3, 'rand_coef.npy'))
    PSFs_rand3 = load_files(path_rand3, N=N_max, file_list=list_slices)

    # Join the different Training Sets
    PSFs = np.concatenate((PSFs_rand1[0], PSFs_rand2[0], PSFs_rand3[0]), axis=0)
    PSFs_square = np.concatenate((PSFs_rand1[1], PSFs_rand2[1], PSFs_rand3[1]), axis=0)

    zern_coefs = np.concatenate((zern_coefs_rand1,
                                 zern_coefs_rand2, zern_coefs_rand3), axis=0)

    PEAK = np.max(PSFs_square[:, 0])

    # Rescale so that the peak of the PSF is approximately 1.0
    PSFs /= PEAK
    PSFs_square /= PEAK

    # Don't forget to downsample the pixels across the slicer width
    PSFs_square, downPSFs_square, downPSFs_flat = downsample_slicer_pixels(PSFs_square)

    # ============================================================================== #
    #                                TRAIN THE MODEL                                 #
    # ============================================================================== #

    training, testing = generate_training_set(2 * N_rand + N_max, 200, downPSFs_flat, zern_coefs, True)

    N_layer = (150, 100, 50)
    model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                     solver='adam', max_iter=N_iter, verbose=True,
                     batch_size='auto', shuffle=True, tol=1e-12,
                     warm_start=True, alpha=1e-4, random_state=1234)
    model.fit(X=training[0], y=training[1])

    guessed = model.predict(X=testing[0])
    print("\nModel guesses:")
    print(guessed[:5])
    print("\nTrue Values")
    print(testing[1][:5])

    print('\nFinal Performance:')
    rms0, rms = evaluate_wavefront_performance(N_zern, testing[1], guessed,
                                           zern_list=zern_list_low, show_predic=True)

    # ============================================================================== #
    #                           TEST SET - DISPLACEMENTS                             #
    # ============================================================================== #

    path_test = os.path.join('POP', 'NYQUIST', 'IRREGULARITIES', 'TEST')

    path_t1 = os.path.join(path_test, '3')
    zern_coefs_t1 = np.load(os.path.join(path_t1, 'disp_coef.npy'))
    PSFs_t1 = load_files(path_t1, N=N_PSF, file_list=list_slices)

    path_t2 = os.path.join(path_test, '1')
    zern_coefs_t2 = np.load(os.path.join(path_t2, 'disp_coef.npy'))
    PSFs_t2 = load_files(path_t2, N=N_PSF, file_list=list_slices)

    PSF_t = np.concatenate((PSFs_t1[0], PSFs_t2[0]), axis=0)
    PSF_t /= PEAK

    PSF_t_square = np.concatenate((PSFs_t1[1], PSFs_t2[1]), axis=0)
    PSF_t_square /= PEAK

    # Don't forget to downsample the TEST SET PSFs
    PSF_t_square, downPSF_t_square, downPSF_t_flat = downsample_slicer_pixels(PSF_t_square)

    coeffs_t = np.concatenate((zern_coefs_t1, zern_coefs_t2), axis=0)
    # Don't forget to remove the DISPLACEMENT coefficients
    zern_test = coeffs_t[:, :N_zern]
    disp_test = coeffs_t[:, -1]

    # Test the performance on the TEST set
    guessed_t = model.predict(X=downPSF_t_flat)
    print("\nLOW model guesses:")
    print(guessed_t[:5])
    print("\nTrue Values")
    print(zern_test[:5])

    _rms0, rms_test = evaluate_wavefront_performance(N_zern, zern_test, guessed_t,
                                                   zern_list=zern_list_low, show_predic=True)

    plt.figure()
    plt.scatter(disp_test, rms_test * wave_nom, s=3)
    plt.ylim([0, 60])
    plt.xlim([-25, 25])
    plt.xlabel('Displacement [mm]')
    plt.ylabel('RMS residual [nm]')
    plt.show()


    # ============================================================================== #
    #                                    ROBUST TRAINING                             #
    # ============================================================================== #

    # Recycle the initial training by saying "this is what NO displacemente looks like"
    train_set0 = downPSFs_flat
    zern_test0 = np.concatenate((zern_coefs,
                                 np.zeros((zern_coefs.shape[0], 1))), axis=-1)

    train_set1 = downPSF_t_flat
    zern_test1 = coeffs_t.copy()

    PSFs_robust = np.concatenate((train_set0, train_set1), axis=0)
    zern_robust = np.concatenate((zern_test0, zern_test1), axis=0)

    N_robust = PSFs_robust.shape[0]

    training_rob, testing_rob = generate_training_set(N_robust, 150, PSFs_robust, zern_robust, True)

    N_layer = (300, 200, 100)
    model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                         solver='adam', max_iter=N_iter, verbose=True,
                         batch_size='auto', shuffle=True, tol=1e-12,
                         warm_start=True, alpha=1e-4, random_state=1234)
    model.fit(X=training_rob[0], y=training_rob[1])

    guessed = model.predict(X=testing_rob[0])
    print("\nModel guesses:")
    print(guessed[:5])
    print("\nTrue Values")
    print(testing_rob[1][:5])

    print('\nFinal Performance:')
    rms0_rob, rms_rob = evaluate_wavefront_performance(N_zern, testing_rob[1][:, :N_zern], guessed[:, :N_zern],
                                               zern_list=zern_list_low, show_predic=True)

    disp_test = testing_rob[1][:, -1]
    disp_guess = guessed[:, -1]


