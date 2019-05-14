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


    # ============================================================================== #

    path_high = os.path.join(path_files, 'HIGH')
    zern_coefs_high = np.loadtxt(os.path.join(path_high, 'coef_high.txt'))
    PSFs_high = load_files(path_high, N=N_PSF, file_list=list_slices)

    PSFs_high[0] /= PEAK
    PSFs_high[1] /= PEAK

    # Don't forget to downsample the pixels across the slicer width
    _PSFs_high_sq, downPSFs_high, downPSFs_high_flat = downsample_slicer_pixels(PSFs_high[1])

    # Train the HIGH order network
    high_training, high_testing = generate_training_set(N_PSF, n_test, downPSFs_high_flat, zern_coefs_high, True)

    high_training_noisy, high_coefs_noisy = train_with_noise(high_training[0], high_training[1], N_repeat=5)

    high_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9,
                             warm_start=True, alpha=1e-2, random_state=1234)

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
    ae_coef = np.random.uniform(-a_max/4, a_max/4, size=(N_auto, N_low + N_high))
    path_auto = os.path.join('POP', 'NYQUIST', 'HIGH ORDERS', 'AUTOENCODER')
    # np.save(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef3'), ae_coef)
    # np.savetxt(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef3.txt'), ae_coef, fmt='%.5f')

    # Subtract the LOW orders
    ae_low_coef, ae_high_coef = ae_coef[:, :N_low], ae_coef[:, N_low:]
    extra_zeros = np.zeros((N_auto, N_low))
    only_high = np.concatenate((extra_zeros, ae_high_coef), axis=1)
    # np.save(os.path.join(path_auto, 'TRAINING_HIGH', 'autoencoder_high_coef3'), only_high)
    # np.savetxt(os.path.join(path_auto, 'TRAINING_HIGH', 'autoencoder_high_coef3.txt'), only_high, fmt='%.5f')

    # Define the AUTOENCODER
    from keras.layers import Dense
    from keras.models import Sequential
    from keras import backend as K
    from numpy.linalg import norm as norm

    input_dim = 2*N_crop**2
    encoding_dim = 32
    epochs = 2500
    batch = 32

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

    # Train the AUTOENCODER

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

    residuals = np.mean(norm(np.abs(decoded - test_clean), axis=-1))
    total = np.mean(norm(np.abs(test_clean), axis=-1))
    print(residuals / total * 100)

    num_images = 5
    # np.random.seed(1234)
    random_images = np.random.randint(test_clean.shape[0], size=num_images)

    plt.figure()
    cmap = 'viridis'
    j_mid = 2
    for i, img_idx in enumerate(random_images):

        ax1 = plt.subplot(4, num_images, i + 1)
        im1 = test_clean[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im1, cmap=cmap)
        plt.colorbar()
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if i == j_mid:
            ax1.set_title('Ground Truth: Only HIGH orders')

        ax2 = plt.subplot(4, num_images, num_images + i + 1)
        im2 = test_noisy[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im2, cmap=cmap)
        plt.colorbar()
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        if i == j_mid:
            ax2.set_title('Input: Both LOW and HIGH orders')

        ax3 = plt.subplot(4, num_images, 2*num_images + i + 1)
        im3 = decoded[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im3, cmap=cmap)
        plt.colorbar()
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        if i == j_mid:
            ax3.set_title('Prediction: PSF after Autoencoder')

        ax4 = plt.subplot(4, num_images, 3*num_images + i + 1)
        res = (test_clean - decoded)
        im4 = res[img_idx, N_crop**2:].reshape((N_crop, N_crop))
        plt.imshow(im4, cmap='bwr')
        plt.colorbar()
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        if i == j_mid:
            ax4.set_title('Residuals: Ground Truth - Prediction')
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
                                                       zern_list=zern_list_low+zern_list_high, show_predic=True)

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
        PSF_high = load_files(coef_path, N=N_PSF, file_list=list_slices)
        PSF_high[0] /= PEAK
        PSF_high[1] /= PEAK

        _PSF_high, downPSF_high, downPSF_high_flat = downsample_slicer_pixels(PSF_high[1])

        # USE THE AUTOENCODER to "clean" the PSF from LOW order features
        decoded_high = AE.predict(downPSF_high_flat)

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

        coef_path = os.path.join('POP', 'NYQUIST', 'HIGH ORDERS', 'TEST', '%dHIGH' %(k+1))
        file_name = os.path.join(coef_path, 'remaining_iter%d_%d.txt' %(k+1, 2))
        # np.savetxt(file_name, remaining2, fmt='%.5f')

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
    colors_AE = cm.jet(np.linspace(0, 1, N_test))

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
    plt.hist(RMS_AE[-1],  histtype='step', color='navy', label=r'Final RMS: %.1f $\pm$ %.1f nm' %(final_mean, final_std))
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