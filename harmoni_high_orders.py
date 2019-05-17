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

# Slices
list_slices = [17, 19, 21, 55, 57]
i_central = 19

# POP arrays - Nyquist sampled PSF
x_size = 2.08           # Physical size of arraxy at Image Plane
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
zern_list_low = ['Defocus', 'Astigmatism X', 'Astigmatism Y', 'Coma X', 'Coma Y']
zern_list_high = ['Trefoil X', 'Trefoil Y', 'Quatrefoil X', 'Quatrefoil Y']

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

def transform_zemax_to_noll(zemax_coef):
   """
   Rearranges the order of the coefficients from the Zemax convention to thew
   Zern library convention to evaluate the RMS and other performance metrics
   Zemax [Defocus, Astig X, Astig Y, Coma X, Coma Y, Trefoil X, Trefoil Y]
   Zern [Astig X, Defocus, Astig Y, Trefoil X, Coma X, Coma Y, Trefoil Y]
   """
   N, N_zern = zemax_coef.shape
   if N_zern == 4:
       # Case for HIGH orders [Tref X, Tref Y, Quatref X, Quatref Y]
       zern_coef = np.zeros((N, 7 + 5))
       zern_coef[:, 3] = zemax_coef[:, 0]   # Trefoil X
       zern_coef[:, 6] = zemax_coef[:, 1]   # Trefoil Y
       zern_coef[:, 7] = zemax_coef[:, 2]   # Quatrefoil X
       zern_coef[:,11] = zemax_coef[:, 3]   # Quatrefoil Y

   if N_zern == 5:
       # Case for LOW orders [Defocus, Astig X, Astig Y, Coma X, Coma Y]
       zern_coef = np.zeros((N, 7))
       zern_coef[:, 0] = zemax_coef[:, 1]  # Astig X
       zern_coef[:, 1] = zemax_coef[:, 0]  # Defocus
       zern_coef[:, 2] = zemax_coef[:, 2]  # Astig Y
       zern_coef[:, 3] = 0  # Trefoil X
       zern_coef[:, 4] = zemax_coef[:, 3]  # Coma X
       zern_coef[:, 5] = zemax_coef[:, 4]  # Coma Y
       zern_coef[:, 6] = 0  # Trefoil Y

   if N_zern == 4 + 5:
       # Case for Both HIGH and LOW orders
       # [Defocus, Astig X, Astig Y, Coma X, Coma Y] + [Tref X, Tref Y, Quatref X, Quatref Y]
       zern_coef = np.zeros((N, 7 + 5))
       zern_coef[:, 0] = zemax_coef[:, 1]   # Astig X
       zern_coef[:, 1] = zemax_coef[:, 0]   # Defocus
       zern_coef[:, 2] = zemax_coef[:, 2]   # Astig Y
       zern_coef[:, 3] = zemax_coef[:, 5]   # Trefoil X
       zern_coef[:, 4] = zemax_coef[:, 3]   # Coma X
       zern_coef[:, 5] = zemax_coef[:, 4]   # Coma Y
       zern_coef[:, 6] = zemax_coef[:, 6]   # Trefoil Y
       zern_coef[:, 7] = zemax_coef[:, 7]   # Quatrefoil X
       zern_coef[:,11] = zemax_coef[:, 8]   # Quatrefoil Y

   return zern_coef

def evaluate_wavefront_performance(N_zern, test_coef, guessed_coef, zern_list, show_predic=False):
    """
    Evaluates the performance of the ML method regarding the final
    RMS wavefront error. Compares the initial RMS NCPA and the residual
    after correction
    """

    # Transform the ordering to match the Zernike matrix
    new_test_coef = transform_zemax_to_noll(test_coef)
    new_guessed_coef = transform_zemax_to_noll(guessed_coef)

    x = np.linspace(-1, 1, 512, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho, theta = np.sqrt(xx**2 + yy**2), np.arctan2(xx, yy)
    pupil = rho <= 1.0
    rho, theta = rho[pupil], theta[pupil]
    zernike = zern.ZernikeNaive(mask=pupil)
    _phase = zernike(coef=np.zeros(new_test_coef.shape[1] + 3), rho=rho, theta=theta, normalize_noll=False, mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix[:,3:]   # remove the piston and tilts
    H_matrix = zern.invert_model_matrix(H_flat, pupil)

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

        for k in range(N_zern):
            guess = guessed_coef[:, k]
            coef = test_coef[:, k]

            colors = wave_nom * residual_rms
            colors -= colors.min()
            colors /= colors.max()
            colors = cm.rainbow(colors)

            plt.figure()
            ss = plt.scatter(coef, guess, c=colors, s=20)
            x = np.linspace(-0.15, 0.15, 10)
            # plt.colorbar(ss)
            plt.plot(x, x, color='black', linestyle='--')
            title = zern_list[k]
            plt.title(title)
            plt.xlabel('True Value [waves]')
            plt.ylabel('Predicted Value [waves]')
            plt.xlim([-0.15, 0.15])
            plt.ylim([-0.15, 0.15])

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


def load_files(path, file_list, N):
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

def create_rand_coef(a_max, N_PSFs, N_repeat, N_zern):
    """
    Creates Zernike coefficients randomly. For a total of N_PSFs examples.
    However, in order to cover a wider range of aberration intensities,
    we decrease the amplitude each N_repeat
    :param a_max: Maximum amplitude of the Zernike coefficients
    :param N_PSFs: Total number of random examples
    :param N_repeat: Number of subsets to divide N_PSFs.
    :param N_zern: Number of Zernike polynomials
    :return:
    """
    N_set = N_PSFs // N_repeat
    coef_list = []
    for i in range(N_repeat):
        amp = a_max / (1 + i)
        coef = np.random.uniform(-amp, amp, size=(N_set, N_zern))
        coef_list.append(coef)
    coeffs = np.concatenate(coef_list, axis=0)

    return coeffs

def train_with_noise(PSF, coefs, N_repeat):
    """
    Data Augmentation of the training set including noise
    to increase the robustness of the network
    :param PSF: clean PSFs from the nominal training set
    :param coefs: aberration coefficients
    :param N_repeat: number of repetitions
    :return:
    """
    PSF_list, coef_list = [], []
    for i in range(N_repeat):
        sigma = 0.1 * np.sqrt(3) / 2**i
        noise_map = np.random.uniform(1 - sigma, 1 + sigma, size=PSF.shape)
        noisy_training = PSF * noise_map
        PSF_list.append(noisy_training)
        coef_list.append(coefs)
    noisy_training = np.concatenate(PSF_list, axis=0)
    noisy_coefs = np.concatenate(coef_list, axis=0)

    return noisy_training, noisy_coefs

if __name__ == "__main__":

    path_files = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITHOUT AE/TRAIN')

    # os.path.join('POP', 'NYQUIST', 'HIGH ORDERS', 'TRAIN')

    # ============================================================================== #
    #                              GENERATE TRAINING SETS                            #
    # ============================================================================== #

    N_low, N_high = 5, 4
    N_PSF, N_repeat = 3000, 1
    a_max = 0.15
    n_test = N_PSF//10

    # LOW ORDERS network
    # coef_low = create_rand_coef(a_max, N_PSFs=N_PSF, N_repeat=N_repeat, N_zern=N_low)
    # np.save(os.path.join(path_files, 'LOW', 'coef'), coef_low)
    # np.savetxt(os.path.join(path_files, 'LOW', 'coef.txt'), coef_low, fmt='%.5f')
    #
    # # HIGH ORDERS network
    # coef_high = create_rand_coef(a_max, N_PSFs=N_PSF, N_repeat=N_repeat, N_zern=N_high)
    # np.save(os.path.join(path_files, 'HIGH', 'coef_high'), coef_high)
    # np.savetxt(os.path.join(path_files, 'HIGH', 'coef_high.txt'), coef_high, fmt='%.5f')

    # ============================================================================== #
    #                                LOAD TRAINING SETS                              #
    # ============================================================================== #

    path_low = os.path.join(path_files, 'LOW')
    zern_coefs_low = np.loadtxt(os.path.join(path_low, 'coef_low.txt'))
    PSFs_low = load_files(path_low, N=N_PSF, file_list=list_slices)

    PEAK = np.max(PSFs_low[1][:, 0])

    # Rescale so that the peak of the PSF is approximately 1.0
    PSFs_low[0] /= PEAK
    PSFs_low[1] /= PEAK

    # Don't forget to downsample the pixels across the slicer width
    _PSFs_low_sq, downPSFs_low, downPSFs_low_flat = downsample_slicer_pixels(PSFs_low[1])

    # Train the LOW order network
    low_training, low_testing = generate_training_set(N_PSF, n_test, downPSFs_low_flat, zern_coefs_low, True)

    low_training_noisy, low_coefs_noisy = train_with_noise(low_training[0], low_training[1], N_repeat=5)

    N_layer = (300, 200, 100)
    low_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=False, tol=1e-9,
                             warm_start=True, alpha=1e-2, random_state=1234)

    low_model.fit(X=low_training[0], y=low_training[1])
    # low_model.fit(X=low_training_noisy, y=low_coefs_noisy)

    low_guessed = low_model.predict(X=low_testing[0])
    print("\nLOW model guesses:")
    print(low_guessed[:5])
    print("\nTrue Values")
    print(low_testing[1][:5])

    r = low_testing[1] - low_guessed
    # print(r[:4])

    print('\n LOW order Model:')
    low_rms0, low_rms = evaluate_wavefront_performance(N_low, low_testing[1], low_guessed,
                                                       zern_list=zern_list_low, show_predic=True)
    print(wave_nom * np.std(low_rms))


    # ================================================================================================================ #

    path_high = os.path.join(path_files, 'HIGH')
    zern_coefs_high = np.loadtxt(os.path.join(path_high, 'coef_high.txt'))
    PSFs_high = load_files(path_high, N=N_PSF, file_list=list_slices)

    PSFs_high[0] /= PEAK
    PSFs_high[1] /= PEAK

    # Don't forget to downsample the pixels across the slicer width
    _PSFs_high_sq, downPSFs_high, downPSFs_high_flat = downsample_slicer_pixels(PSFs_high[1])

    # Train the HIGH order network
    high_training, high_testing = generate_training_set(N_PSF, n_test, downPSFs_high_flat, zern_coefs_high, True)

    high_training_noisy, high_coefs_noisy = train_with_noise(high_training[0], high_training[1], N_repeat=7)

    high_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9,
                             warm_start=False, alpha=1e-2, random_state=1234)

    # high_model.fit(X=high_training[0], y=high_training[1])
    high_model.fit(X=high_training_noisy, y=high_coefs_noisy)

    high_guessed = high_model.predict(X=high_testing[0])
    print("\nHIGH model guesses:")
    print(high_guessed[:5])
    print("\nTrue Values")
    print(high_testing[1][:5])

    print('\n HIGH order Model:')
    high_rms0, high_rms = evaluate_wavefront_performance(N_high, high_testing[1], high_guessed,
                                                       zern_list=zern_list_high, show_predic=False)

    # ============================================================================== #
    #                                TRAIN THE AUTOENCODER                           #
    # ============================================================================== #

    N_auto = 2500
    N_ext = N_auto - 50
    ae_coef = np.random.uniform(-a_max, a_max, size=(N_auto, N_low + N_high))
    # path_auto = os.path.join('POP', 'NYQUIST', 'HIGH ORDERS', 'AUTOENCODER')
    path_auto = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE')
    np.save(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef1'), ae_coef)
    np.savetxt(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef1.txt'), ae_coef, fmt='%.5f')

    # Subtract the LOW orders
    ae_low_coef, ae_high_coef = ae_coef[:, :N_low], ae_coef[:, N_low:]
    extra_zeros = np.zeros((N_auto, N_low))
    only_high = np.concatenate((extra_zeros, ae_high_coef), axis=1)
    np.save(os.path.join(path_auto, 'TRAINING_HIGH', 'autoencoder_high_coef2'), only_high)
    np.savetxt(os.path.join(path_auto, 'TRAINING_HIGH', 'autoencoder_high_coef2.txt'), only_high, fmt='%.5f')

    # Define the AUTOENCODER
    from keras.layers import Dense
    from keras.models import Sequential, Model, Input
    from keras import backend as K
    from numpy.linalg import norm as norm
    from keras.utils.vis_utils import plot_model

    input_dim = 2*N_crop**2
    encoding_dim = 32
    epochs = 2500
    batch = 32

    ### Autoencoder architecture
    K.clear_session()
    AE = Sequential()
    AE.add(Dense(16 * encoding_dim, input_shape=(input_dim, ), activation='relu'))
    AE.add(Dense(4 * encoding_dim, activation='relu'))
    AE.add(Dense(2 * encoding_dim, activation='relu'))
    AE.add(Dense(encoding_dim, activation='relu'))
    AE.add(Dense(2 * encoding_dim, activation='relu'))
    AE.add(Dense(4 * encoding_dim, activation='relu'))
    AE.add(Dense(input_dim, activation='sigmoid'))
    AE.summary()
    AE.compile(optimizer='adam', loss='binary_crossentropy')

    plot_model(AE, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    ### Train the AUTOENCODER
    PSFs_AE = load_files(os.path.join(path_auto, 'TRAINING_BOTH'), N=N_auto, file_list=list_slices)
    PSFs_AE[0] /= PEAK
    PSFs_AE[1] /= PEAK
    _PSFs_AE, downPSFs_AE, downPSFs_AE_flat = downsample_slicer_pixels(PSFs_AE[1])
    # ae_coef = np.loadtxt(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef.txt'))

    PSFs_AE_high = load_files(os.path.join(path_auto, 'TRAINING_HIGH'), N=N_auto, file_list=list_slices)
    PSFs_AE_high[0] /= PEAK
    PSFs_AE_high[1] /= PEAK
    _PSFs_AE_high, downPSFs_AE_high, downPSFs_AE_high_flat = downsample_slicer_pixels(PSFs_AE_high[1])
    # ae_coef_high = np.loadtxt(os.path.join(path_auto, 'TRAINING_HIGH', 'autoencoder_high_coef.txt'))

    train_noisy = downPSFs_AE_flat[:N_ext]
    train_clean = downPSFs_AE_high_flat[:N_ext]

    test_noisy = downPSFs_AE_flat[N_ext:]
    test_clean = downPSFs_AE_high_flat[N_ext:]

    AE.fit(train_noisy, train_clean,
           epochs=epochs, batch_size=batch, shuffle=True, verbose=2,
           validation_data=(test_noisy, test_clean))

    decoded = AE.predict(test_noisy)

    # Make sure the training has succeeded by checking the residuals
    residuals = np.mean(norm(np.abs(decoded - test_clean), axis=-1))
    total = np.mean(norm(np.abs(test_clean), axis=-1))
    print(residuals / total * 100)

    ### ENCODED images
    input_img = Input(shape=(input_dim,))
    encoded_layer1 = AE.layers[0]
    encoded_layer2 = AE.layers[1]
    encoded_layer3 = AE.layers[2]
    encoded_layer4 = AE.layers[3]

    encoder = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))

    encoder.summary()
    encoded_images = encoder.predict(train_noisy)

    N_samples = 50
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    im1 = ax1.imshow(encoded_images[:N_samples, :16], cmap='hot', origin='lower')
    ax1.set_title('Nominal PSF')
    ax1.set_ylabel('Sample')
    ax1.set_xlabel('Pixel Feature')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(encoded_images[:N_samples, 16:], cmap='hot', origin='lower')
    ax2.set_title('Defocused PSF')
    ax2.set_xlabel('Pixel Feature')
    plt.colorbar(im2)
    plt.show()


    from sklearn.decomposition import PCA
    from scipy.optimize import least_squares as lsq

    def features_training(num_images=1):
        """
        Function to analyse the features of the TRAINING set of the autoencoder
        """

        coefs = np.loadtxt(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef1.txt'))
        norm_coef = []
        losses_focus, peaks_focus, mins_focus = [], [], []
        losses_defocus, peaks_defocus, mins_defocus = [], [], []

        ### Light Loss - see how the Low Orders modify the total intensity
        for j in range(N_ext):

            low_orders = coefs[j, :N_low]
            norm_coef.append(np.linalg.norm(low_orders))

            input_focus = train_noisy[j, :N_crop**2].reshape((N_crop, N_crop))
            output_focus = train_clean[j, :N_crop**2].reshape((N_crop, N_crop))
            removed_features_focus = input_focus - output_focus
            loss_focus = np.sum(removed_features_focus)
            losses_focus.append(loss_focus)
            peaks_focus.append(np.max(removed_features_focus))
            mins_focus.append(np.min(removed_features_focus))

            input_defocus = train_noisy[j, N_crop**2:].reshape((N_crop, N_crop))
            output_defocus = train_clean[j, N_crop**2:].reshape((N_crop, N_crop))
            removed_features_defocus = input_defocus - output_defocus
            loss_defocus = np.sum(removed_features_defocus)
            losses_defocus.append(loss_defocus)
            peaks_defocus.append(np.max(removed_features_defocus))
            mins_defocus.append(np.min(removed_features_defocus))
        norm_coef = np.array(norm_coef)

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)
        # Focused PSF
        p_sort = np.argsort(peaks_focus)
        ax1.scatter(norm_coef[p_sort], np.sort(peaks_focus),
                    color=cm.bwr(np.linspace(0.5 + np.min(peaks_focus), 1, N_ext)), s=4, label='Maxima')
        m_sort = np.argsort(mins_focus)
        ax1.scatter(norm_coef[m_sort], np.sort(mins_focus),
                    color=cm.bwr(np.linspace(0, 0.5, N_ext)), s=4, label='Minima')
        loss_sort = np.argsort(losses_focus)
        ax1.legend(loc=2)
        leg = ax1.get_legend()
        leg.legendHandles[0].set_color('red')
        leg.legendHandles[1].set_color('blue')

        ax1.axhline(y=0.0, linestyle='--', color='black')
        ax1.set_title('Nominal PSF')
        ax1.set_ylabel(r'Light loss')
        ax1.set_ylim([-0.5, 0.5])

        ax3.scatter(norm_coef[loss_sort], np.sort(losses_focus), color='black', s=3, label='Total')
        ax3.legend(loc=2)
        ax3.axhline(y=0.0, linestyle='--', color='black')
        ax3.set_xlabel(r'Norm of low orders $\Vert a_{low} \Vert$')
        ax3.set_ylabel(r'Light loss')

        # Defocused PSF
        p_sort = np.argsort(losses_defocus)
        ax2.scatter(norm_coef[p_sort], np.sort(peaks_defocus),
                    color=cm.bwr(np.linspace(0.5 + np.min(peaks_defocus), 1, N_ext)), s=4, label='Maxima')
        m_sort = np.argsort(mins_defocus)
        ax2.scatter(norm_coef[m_sort], np.sort(mins_defocus),
                    color=cm.bwr(np.linspace(0, 0.5, N_ext)), s=4, label='Minima')
        loss_sort = np.argsort(losses_defocus)
        ax2.legend(loc=2)
        leg = ax2.get_legend()
        leg.legendHandles[0].set_color('red')
        leg.legendHandles[1].set_color('blue')

        ax2.axhline(y=0.0, linestyle='--', color='black')
        ax2.set_title('Defocused PSF')

        ax4.scatter(norm_coef[loss_sort], np.sort(losses_defocus), color='black', s=3, label='Total')
        ax4.legend(loc=2)
        ax4.axhline(y=0.0, linestyle='--', color='black')
        ax4.set_xlabel(r'Norm of low orders $\Vert a_{low} \Vert$')

        ### PCA analysis of the removed features
        # Focused PSF
        N_comp = N_low
        removed_features = train_noisy[:, :N_crop ** 2] - train_clean[:, :N_crop ** 2]
        pca = PCA(n_components=N_comp)
        pca.fit(X=removed_features)
        components = pca.components_.reshape((N_comp, N_crop, N_crop))
        variance_ratio = pca.explained_variance_ratio_
        total_variance = np.sum(variance_ratio)

        plt.figure()
        for i in range(N_comp):
            ax = plt.subplot(2, N_comp, i+1)
            plt.imshow(components[i], cmap='seismic', origin='lower')
            ax.set_title(r'PCA #%d [$\sigma^2_r=%.2f/%.2f$]' %(i+1, variance_ratio[i], total_variance))
            plt.colorbar(orientation="horizontal")
            cmin = min(components[i].min(), -components[i].max())
            plt.clim(cmin, -cmin)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        removed_features_defocus = train_noisy[:, N_crop ** 2:] - train_clean[:, N_crop ** 2:]
        pca_defocus = PCA(n_components=N_comp)
        pca_defocus.fit(X=removed_features_defocus)
        components_defocus = pca_defocus.components_.reshape((N_comp, N_crop, N_crop))
        variance_ratio_defocus = pca_defocus.explained_variance_ratio_
        total_variance_defocus = np.sum(variance_ratio_defocus)

        for i in range(N_comp):
            ax = plt.subplot(2, N_comp, i+1+N_comp)
            plt.imshow(components_defocus[i], cmap='seismic', origin='lower')
            ax.set_title(r'PCA #%d [$\sigma^2_r=%.2f/%.2f$]' %(i+1, variance_ratio_defocus[i], total_variance_defocus))
            plt.colorbar(orientation="horizontal")
            cmin = min(components_defocus[i].min(), -components_defocus[i].max())
            plt.clim(cmin, -cmin)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        ### Least Squares fit of the removed features
        def residuals_lsq(x, image_data, pca_components):
            model_image = np.dot(x, pca_components)
            return image_data - model_image

        random_images = np.random.randint(train_noisy.shape[0], size=num_images)
        for j in random_images:
            im = removed_features[j]
            res_lsq = lsq(fun=residuals_lsq, x0=np.zeros(N_comp), args=(im, pca.components_))
            x_fit = res_lsq['x']
            im_fit = (np.dot(x_fit, pca.components_)).reshape((N_crop, N_crop))
            im = im.reshape((N_crop, N_crop))
            vmin_im = min(im.min(), -im.max())
            vmin_fit = min(im_fit.min(), -im_fit.max())
            vmin = min(vmin_im, vmin_fit)

            error = np.sum(np.abs(im_fit - im)) / np.sum(np.abs(im))

            plt.figure()
            cmap = 'seismic'
            ax1 = plt.subplot(1, 3, 1)
            plt.imshow(im, cmap=cmap)
            plt.colorbar(orientation="horizontal")
            plt.clim(vmin, -vmin)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.set_title(r'Removed features: $PSF(\Phi_{low} + \Phi_{high}) - PSF(\Phi_{high})$')

            ax2 = plt.subplot(1, 3, 2)
            plt.imshow(im_fit, cmap=cmap)
            plt.colorbar(orientation="horizontal")
            plt.clim(vmin, -vmin)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.set_title(r'Least-Squares fit from PCA: $x_{lsq} \cdot PCA$')

            res = im_fit - im
            ax3 = plt.subplot(1, 3, 3)
            plt.imshow(res, cmap='bwr')
            plt.colorbar(orientation="horizontal")
            min_res = min(res.min(), -res.max())
            plt.clim(min_res, -min_res)
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
            ax3.set_title(r'Residuals ($\epsilon = %.2f$)' %error)

        random_images = np.random.choice(train_noisy.shape[0], size=48, replace=False)
        print(random_images)
        plt.figure()
        for i, img_j in enumerate(random_images):
            ax = plt.subplot(6, 8, i + 1)
            im = removed_features[img_j].reshape((N_crop, N_crop))
            plt.imshow(im, cmap='seismic')
            min_im = min(im.min(), -im.max())
            plt.clim(min_im, -min_im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

        return pca

    pca_features = features_training()


    ### --------------------------------------------------------------

    num_images = 5
    # np.random.seed(1234)
    random_images = np.random.randint(test_clean.shape[0], size=num_images)

    plt.figure()
    cmap = 'viridis'
    j_mid = 2
    for i, img_idx in enumerate(random_images):

        ax1 = plt.subplot(5, num_images, i + 1)
        im1 = test_clean[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im1, cmap=cmap)
        plt.colorbar()
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if i == j_mid:
            ax1.set_title('Ground Truth: Only HIGH orders')

        ax2 = plt.subplot(5, num_images, num_images + i + 1)
        im2 = test_noisy[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im2, cmap=cmap)
        plt.colorbar()
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        if i == j_mid:
            ax2.set_title('Input: Both LOW and HIGH orders')

        ax3 = plt.subplot(5, num_images, 2*num_images + i + 1)
        im3 = decoded[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im3, cmap=cmap)
        plt.colorbar()
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        if i == j_mid:
            ax3.set_title('Prediction: PSF after Autoencoder')

        ax4 = plt.subplot(5, num_images, 3*num_images + i + 1)
        res = (test_clean - decoded)
        im4 = res[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im4, cmap='bwr')
        plt.colorbar()
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        if i == j_mid:
            ax4.set_title('Residuals: Ground Truth - Prediction')

        ax5 = plt.subplot(5, num_images, 4 * num_images + i + 1)
        diff = (test_noisy - decoded)
        im5 = diff[img_idx, N_crop ** 2:].reshape((N_crop, N_crop))
        plt.imshow(im5, cmap='bwr')
        plt.colorbar()
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        if i == j_mid:
            ax5.set_title('Removed: Noisy - Decoded')
    plt.show()

    ### What FEATURES does the AUTOENCODER REMOVE?
    for i, img_idx in enumerate(random_images):

        ax1 = plt.subplot(3, num_images, i + 1)
        im1 = test_noisy[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im1, cmap=cmap)
        plt.colorbar(orientation="horizontal")
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if i == j_mid:
            ax1.set_title('Noisy Input: Both LOW and HIGH orders')

        ax2 = plt.subplot(3, num_images, num_images + i + 1)
        im2 = decoded[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im2, cmap=cmap)
        plt.colorbar(orientation="horizontal")
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        if i == j_mid:
            ax2.set_title('Clean Output: PSF after Autoencoder')

        ax3 = plt.subplot(3, num_images, 2*num_images + i + 1)
        diff = (test_noisy - decoded)
        im3 = diff[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im3, cmap='bwr')
        plt.colorbar(orientation="horizontal")
        cmax = max(im3.max(), -1*im3.min())
        plt.clim(-cmax, cmax)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        if i == j_mid:
            ax3.set_title('Removed features: Noisy - Clean')

    # ============================================================================== #
    #                                LOW ORDERS FEATURES                             #
    # ============================================================================== #
    path_feat = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE/ONLY_LOWS')
    PSFs_feat = load_files(path_feat, N=6, file_list=list_slices)

    # downPSFs_feat = PSFs_feat[1]
    _PSFs_feat, downPSFs_feat, downPSFs_feat_flat = downsample_slicer_pixels(PSFs_feat[1])
    peak_perf = np.max(downPSFs_feat[0, 0])
    downPSFs_feat /= peak_perf

    for i in range(5):
        res = downPSFs_feat[i+1,0] - downPSFs_feat[0,0]
        mm = min(res.min(), -res.max())
        plt.figure()
        plt.imshow(res, cmap='seismic')
        plt.colorbar()
        plt.clim(mm, -mm)
    plt.show()


    # ============================================================================== #
    #                                 GENERATE TEST SETS                             #
    # ============================================================================== #

    # The Test Set contains both LOW and HIGH order coefficients
    # path_test = os.path.join('POP', 'NYQUIST', 'HIGH ORDERS', 'TEST', '0')
    path_test = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITHOUT AE/TEST/0')
    N_test = 250
    # coef_test = create_rand_coef(a_max, N_PSFs=N_test, N_repeat=1, N_zern=N_low + N_high)
    # np.save(os.path.join(path_test, '0', 'coef_test'), coef_test)
    # np.savetxt(os.path.join(path_test, '0', 'coef_test.txt'), coef_test, fmt='%.5f')

    coef_test = np.loadtxt(os.path.join(path_test, 'coef_test.txt'))
    PSFs_test = load_files(path_test, N=N_test, file_list=list_slices)

    PSFs_test[0] /= PEAK
    PSFs_test[1] /= PEAK

    # Don't forget to downsample the pixels across the slicer width
    _PSFs_test, downPSFs_test, downPSFs_test_flat = downsample_slicer_pixels(PSFs_test[1])

    # ============================================================================== #
    #                                 ITERATIVE ALGORITHM                            #
    # ============================================================================== #

    rms_evolution = []
    iterations = 2
    extra_low = np.zeros((N_test, N_low))
    extra_high = np.zeros((N_test, N_high))

    # check the initial RMS
    rms00, rms_initial = evaluate_wavefront_performance(N_low + N_high, coef_test, np.zeros_like(coef_test),
                                                       zern_list=zern_list_low+zern_list_high, show_predic=False)

    rms_evolution.append(rms00)

    coef_low = coef_test
    downPSFs_low_flat = downPSFs_test_flat
    for k in range(iterations):
        low_guessed = low_model.predict(X=downPSFs_low_flat)
        print("\nLOW model guesses:")
        print(low_guessed[:5])
        print("\nTrue Values")
        print(coef_low[:5, :5])

        complete_low_guess = np.concatenate((low_guessed, extra_high), axis=1)
        rms0, low_rms = evaluate_wavefront_performance(N_low + N_high, coef_low, complete_low_guess,
                                                       zern_list=zern_list_low+zern_list_high, show_predic=False)

        rms_evolution.append(low_rms)

        remaining = coef_low - complete_low_guess
        print('\nRemaining aberrations after LOW correction')
        print(remaining[:5])

        # coef_path = os.path.join('POP', 'NYQUIST', 'HIGH ORDERS', 'TEST', '%dLOW' %(k+1))
        coef_path = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITHOUT AE/TEST/%dLOW' %(k+1))
        file_name = os.path.join(coef_path, 'remaining_iter%d_%d.txt' %(k+1, 1))
        np.savetxt(file_name, remaining, fmt='%.5f')

        # ============================================================================== #

        # coef_high = remaining.copy()
        coef_high = np.loadtxt(file_name)
        PSF_high = load_files(coef_path, N=N_test, file_list=list_slices)
        PSF_high[0] /= PEAK
        PSF_high[1] /= PEAK

        _PSF_high, downPSF_high, downPSF_high_flat = downsample_slicer_pixels(PSF_high[1])

        # USE THE AUTOENCODER to "clean" the PSF from LOW order features
        decoded_high = AE.predict(downPSF_high_flat)
        # decoded_high = downPSF_high_flat

        high_guessed = high_model.predict(X=decoded_high)
        print("\nHIGH model guesses:")
        print(high_guessed[:5])
        print("\nTrue Values")
        print(coef_high[:5, 5:])

        complete_high_guess = np.concatenate((extra_low, high_guessed), axis=1)
        rms1, high_rms = evaluate_wavefront_performance(N_low + N_high, coef_high, complete_high_guess,
                                                       zern_list=zern_list_low+zern_list_high, show_predic=False)

        rms_evolution.append(high_rms)

        remaining2 = remaining - complete_high_guess
        print('\nRemaining aberrations after HIGH correction')
        print(remaining2[:5])

        # coef_path = os.path.join('POP', 'NYQUIST', 'HIGH ORDERS', 'TEST', '%dHIGH' %(k+1))
        coef_path = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITHOUT AE/TEST/%dHIGH' % (k + 1))
        file_name = os.path.join(coef_path, 'remaining_iter%d_%d.txt' %(k+1, 2))
        np.savetxt(file_name, remaining2, fmt='%.5f')

        # ============================================================================== #
        # coef_low = remaining2.copy()
        coef_low = np.loadtxt(file_name)
        PSF_low = load_files(coef_path, N=N_test, file_list=list_slices)
        PSF_low[0] /= PEAK
        PSF_low[1] /= PEAK

        _PSFs_low, downPSFs_low, downPSFs_low_flat = downsample_slicer_pixels(PSF_low[1])

    RMS_AE = wave_nom * np.array(rms_evolution)
    mean_rms = np.mean(RMS_AE, axis=-1)
    labels = ['0', '1 LOW', '1 HIGH (AE)', '2 LOW', '2 HIGH (AE)']
    colors_AE = cm.coolwarm(np.linspace(0, 1, N_test))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(5), mean_rms)
    for i in range(5):
        plt.scatter(i * np.ones(N_test), np.sort(RMS_AE[i]), color=colors_AE, s=4)

    plt.xlabel('Iteration')
    plt.xticks(np.arange(5), labels)
    plt.ylabel('RMS [nm]')
    plt.ylim([0, 200])

    final_mean = mean_rms[-1]
    final_std = np.std(RMS_AE[-1])
    n_bins = 15
    plt.subplot(1, 2, 2)
    plt.hist(RMS_AE[-1], color='lightgreen', edgecolor='black')
    plt.xlim([0, 40])
    plt.xlabel('RMS [nm]')
    plt.show()

    # Print Comparison between initial and final values
    _rms, final_rms = evaluate_wavefront_performance(N_low + N_high, coef_test, coef_test - remaining,
                                                    zern_list=zern_list_low + zern_list_high, show_predic=True)

    # ============================================================================== #
    #                              SINGLE STEP - AUTOENCODER                         #
    # ============================================================================== #

    rms_single = []
    rms_single.append(rms00)

    path_single = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITHOUT AE/TEST_SINGLE_STEP/0')

    coef_test = np.loadtxt(os.path.join(path_single, 'coef_test.txt'))
    PSFs_test = load_files(path_single, N=N_test, file_list=list_slices)

    PSFs_test[0] /= PEAK
    PSFs_test[1] /= PEAK

    # Don't forget to downsample the pixels across the slicer width
    _PSFs_test, downPSFs_test, downPSFs_test_flat = downsample_slicer_pixels(PSFs_test[1])

    coef_single = coef_test
    downPSFs_low_flat = downPSFs_test_flat
    for k in range(iterations):

        ### LOW Newtork Guess
        low_guessed = low_model.predict(X=downPSFs_low_flat)
        print("\nLOW model guesses:")
        print(low_guessed[:5])
        print("\nTrue Values")
        print(coef_single[:5, :5])

        ### AUTOENCODER
        decoded_high = AE.predict(downPSFs_low_flat)

        ### HIGh Network Guess
        high_guessed = high_model.predict(X=decoded_high)
        print("\nHIGH model guesses:")
        print(high_guessed[:5])
        print("\nTrue Values")
        print(coef_single[:5, 5:])

        complete_guess = np.concatenate((low_guessed, high_guessed), axis=1)

        _rms, rms = evaluate_wavefront_performance(N_low + N_high, coef_single, complete_guess,
                                                       zern_list=zern_list_low+zern_list_high, show_predic=True)

        rms_single.append(rms)

        ### Save REMAINING for next iteration
        remaining = coef_single - complete_guess
        print('\nRemaining aberrations')
        print(remaining[:5])

        coef_path = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITHOUT AE/TEST_SINGLE_STEP/%d' %(k+1))
        file_name = os.path.join(coef_path, 'remaining_iter%d.txt' %(k+1))
        np.savetxt(file_name, remaining, fmt='%.5f')

        ### Load the files for the next iteration
        coef_single = np.loadtxt(file_name)
        PSF_low = load_files(coef_path, N=N_test, file_list=list_slices)
        PSF_low[0] /= PEAK
        PSF_low[1] /= PEAK

        _PSFs_low, downPSFs_low, downPSFs_low_flat = downsample_slicer_pixels(PSF_low[1])


    RMS_single = wave_nom * np.array(rms_single)
    mean_rms_single = np.mean(RMS_single, axis=-1)
    labels = ['0', '1 LOW', '1 HIGH (AE)', '2 LOW', '2 HIGH (AE)']
    colors_AE = cm.coolwarm(np.linspace(0, 1, N_test))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # plt.plot(np.arange(5), mean_rms)
    for i in range(5):
        if i == 0:
            plt.scatter(i * np.ones(N_test), np.sort(RMS_AE[i]), color=colors_AE, s=3, label='Standard Multi-Network')
        else:
            plt.scatter(i * np.ones(N_test), np.sort(RMS_AE[i]), color=colors_AE, s=3)

    for i in range(5):
        if i==2:
            plt.scatter(i * np.ones(N_test) + 0.075, np.sort(RMS_single[i//2]), color='black', s=3)
        if i==4:
            plt.scatter(i * np.ones(N_test) + 0.075, np.sort(RMS_single[i//2]), color='black', s=3, label='Single-Step Multi-Network')

    plt.xlabel('Iteration')
    plt.xticks(np.arange(5), labels)
    plt.ylabel('RMS [nm]')
    plt.ylim([0, 200])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(RMS_AE[-1], color='blue', histtype='step', label='Standard Multi-Network')
    plt.hist(RMS_single[-1], color='black', histtype='step', label='Single-Step Multi-Network')
    plt.xlim([0, 40])
    plt.xlabel('RMS [nm]')
    plt.legend()
    plt.show()


    # ============================================================================== #
    #                                         EXTRA                                  #
    # ============================================================================== #

    # INFLUENCE OF NUMBER OF EXAMPLES USED FOR TRAINING

    # For the case of 5000 examples with the same aberration intensity (the good one)
    n_examples = np.array([100, 250, 500, 1000, 2000, 3000, 4000, 5000])
    performance_5 = np.array([52.7, 42.3, 23.9, 18.3, 14.2, 12.0, 11.3, 10.7])
    spread_5 = np.array([25.21, 23.38, 10.86, 12.43, 7.17, 7.18, 7.36, 5.90])

    performance_4 = np.array([44.0, 30.4, 20.1, 15.5, 8.9, 13.7, 8.9, 8.0])
    spread_4 = np.array([19.73, 16.99, 12.03, 12.10, 6.47, 7.47, 6.25, 6.23])

    plt.figure()
    plt.errorbar(n_examples, performance_5, yerr=spread_5/2, fmt='o', color='indianred')
    plt.plot(n_examples, performance_5, color='maroon')
    plt.errorbar(n_examples, performance_4, yerr=spread_4/2, fmt='o', color='lightgreen')
    plt.plot(n_examples, performance_4, color='lightgreen')
    plt.xlabel(r'$N$ examples')
    plt.ylabel('Residual RMS [nm]')
    plt.xlim([0, 5015])
    plt.ylim([0, 70])
    plt.show()


    # For the case of 3 consecutive training sets with varying aberration intensity
    n_examples = np.array([100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000])
    perf = np.array([47.1, 32.8, 24.9, 21.2, 14.9, 14.4, 14.8, 12.5, 11.8, 14.2, 12.8, 9.9, 10.0])
    rr = np.array([19.49, 18.35, 11.25, 9.77, 7.15, 8.70, 8.79, 7.15, 8.1, 7.9, 8.8, 7.36, 6.7])
    improv = np.array([54.14, 62.76, 74.30, 78.52, 83.9, 82.7, 80.9, 83.96, 83.72, 79.48, 79.02, 83.04,82.9])

    plt.figure()
    plt.plot(n_examples, perf)
    plt.scatter(n_examples, perf)
    plt.xlabel(r'$N$ examples')
    plt.ylabel('Residual RMS [nm]')
    plt.xlim([0, 3000])

    a = 0.10
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(r'$N$ examples')
    ax1.set_ylabel('Residual RMS [nm]', color=color)
    ax1.plot(n_examples, perf, color=color)
    ax1.scatter(n_examples, perf, color=color)
    ax1.fill_between(np.arange(1000), y1=(1-a)*perf.min(), y2=(1+a)*perf.max(), facecolor='peachpuff', alpha=0.5)
    ax1.fill_between(np.arange(1000, 2000), y1=(1 - a) * perf.min(), y2=(1 + a) * perf.max(),
                     facecolor='lightsteelblue',
                     alpha=0.5)
    ax1.fill_between(np.arange(2000, 3000), y1=(1 - a) * perf.min(), y2=(1 + a) * perf.max(),
                     facecolor='palegreen',
                     alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([(1-a)*perf.min(), (1+a)*perf.max()])
    ax1.set_xlim([0, 3000])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Improvement [per cent]', color=color)  # we already handled the x-label with ax1
    ax2.plot(n_examples, improv, color=color)
    ax2.scatter(n_examples, improv, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    plt.figure()
    plt.plot(n_examples, perf)
    plt.fill_between(n_examples, perf + rr/2, perf - rr/2,
                     facecolor='lightblue', alpha=0.7, label=r'RMS residual $\pm \sigma/2$')
    plt.xlabel(r'$N$ examples')
    plt.ylabel('Residual RMS [nm]')
    plt.xlim([0, 1500])
    plt.show()