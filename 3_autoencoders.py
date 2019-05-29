## Same as harmoni_stacked_autoencoders_long but with 3 AE

"""
==========================================================
                    POP Machine Learning
==========================================================

Test with Stacked Autoencoders for High Order calibration

Denoising criterion as an unsupervised objective to guide
the learning of useful higher level representation

Train the Calibration Networks on the ENCODED data??
"""

import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import zern_core as zern
from pyzdde.zdde import readBeamFile
import matplotlib.cm as cm

from keras.layers import Dense
from keras.models import Sequential, Model, Input
from keras import backend as K
from numpy.linalg import norm as norm

from sklearn.neural_network import MLPRegressor

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

zern_list_low = ['Defocus', 'Astigmatism X', 'Astigmatism Y', 'Coma X', 'Coma Y']
zern_list_high = ['Trefoil X', 'Trefoil Y', 'Quatrefoil X', 'Quatrefoil Y']

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

class Autoencoder(object):

    input_dim = 2*N_crop**2
    encoding_dim = 32

    def __init__(self, noisy_dataset):

        self.features = noisy_dataset

    def load_clean_dataset(self, path_targets, N_PSF):

        features = load_files(path_targets, N=N_PSF, file_list=list_slices)
        features[0] /= PEAK
        features[1] /= PEAK
        _feat, down_feat, down_feat_flat = downsample_slicer_pixels(features[1])
        self.targets = down_feat_flat

    def train_autoencoder(self, N_train, epochs=2000):

        self.N_train = N_train
        train_noisy, train_clean = self.features[:N_train], self.targets[:N_train]
        test_noisy, test_clean = self.features[N_train:], self.targets[N_train:]

        AE = Sequential()
        AE.add(Dense(16 * encoding_dim, input_shape=(input_dim,), activation='relu'))
        AE.add(Dense(4 * encoding_dim, activation='relu'))
        AE.add(Dense(2 * encoding_dim, activation='relu'))
        AE.add(Dense(encoding_dim, activation='relu'))
        AE.add(Dense(2 * encoding_dim, activation='relu'))
        AE.add(Dense(4 * encoding_dim, activation='relu'))
        AE.add(Dense(input_dim, activation='sigmoid'))
        AE.summary()
        AE.compile(optimizer='adam', loss='mean_squared_error')

        AE.fit(train_noisy, train_clean,
                    epochs=epochs, batch_size=batch, shuffle=True, verbose=2,
                    validation_data=(test_noisy, test_clean))

        decoded = AE.predict(test_noisy)

        # Make sure the training has succeeded by checking the residuals
        residuals = np.mean(norm(np.abs(decoded - test_clean), axis=-1))
        total = np.mean(norm(np.abs(test_clean), axis=-1))
        print(residuals / total * 100)

        self.autoencoder_model = AE

        ### Define the ENCODER to access the CODE
        input_img = Input(shape=(self.input_dim,))
        encoded_layer1 = AE.layers[0]
        encoded_layer2 = AE.layers[1]
        encoded_layer3 = AE.layers[2]
        encoded_layer4 = AE.layers[3]
        encoder = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
        encoder.summary()

        self.encoder_model = encoder

        self.train = [train_noisy, train_clean]
        self.test = [test_noisy, test_clean]

    def train_calibration_model(self, coef, N_iter=5000):

        coef_train, coef_test = coef[:self.N_train], coef[self.N_train:]
        psf_train, psf_test = self.encoder_model.predict(self.train[0]), self.encoder_model.predict(self.test[0])

        ### MLP Regressor for HIGH orders (TRAINED ON ENCODED)

        N_layer = (200, 100, 50)
        calibration_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                                  solver='adam', max_iter=N_iter, verbose=True,
                                  batch_size='auto', shuffle=True, tol=1e-9,
                                  warm_start=False, alpha=1e-2, random_state=1234)

        calibration_model.fit(X=psf_train, y=coef_train)

        guessed = calibration_model.predict(X=psf_test)
        print("\nCalibration model guesses:")
        print(guessed[:5])
        print("\nTrue Values")
        print(coef_test[:5])

        self.calibration_model = calibration_model


if __name__ == "__main__":

    N_low, N_med, N_high = 5, 4, 2
    N_auto = 4000
    N_ext = N_auto - 100

    path_2nets = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG')
    path_3nets = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/3_NETS')

    ae_coef = np.loadtxt(os.path.join(path_3nets, 'TRAINING_ALL3', 'autoencoder_coef.txt'))
    ae_low_coef, ae_med_coef, ae_high_coef = ae_coef[:, :N_low], ae_coef[:, N_low:N_low+N_med], ae_coef[:, N_low+N_med+2:]

    """ (1) Load PSF data with ALL 3 aberrations """
    PSFs_AE = load_files(os.path.join(path_3nets, 'TRAINING_ALL3'), N=N_auto, file_list=list_slices)
    PEAK = np.max(PSFs_AE[1])
    PSFs_AE[0] /= PEAK
    PSFs_AE[1] /= PEAK
    _PSFs_AE, downPSFs_AE, downPSFs_AE_flat = downsample_slicer_pixels(PSFs_AE[1])


    """ (2) Load the datasets and train each Autoencoder """
    K.clear_session()

    AutoencoderHigh = Autoencoder(noisy_dataset=downPSFs_AE_flat)
    AutoencoderHigh.load_clean_dataset(os.path.join(path_3nets, 'TRAINING_HIGH_HIGH'), N_PSF=N_auto)
    AutoencoderHigh.train_autoencoder(N_train=N_ext)
    AutoencoderHigh.train_calibration_model(coef=ae_high_coef)

    AutoencoderMedium = Autoencoder(noisy_dataset=downPSFs_AE_flat)
    AutoencoderMedium.load_clean_dataset(os.path.join(path_2nets, 'TRAINING_HIGH'), N_PSF=N_auto)
    AutoencoderMedium.train_autoencoder(N_train=N_ext)
    AutoencoderMedium.train_calibration_model(coef=ae_med_coef)

    AutoencoderMedium = Autoencoder(noisy_dataset=downPSFs_AE_flat)
    AutoencoderMedium.load_clean_dataset(os.path.join(path_2nets, 'TRAINING_HIGH'), N_PSF=N_auto)
    AutoencoderMedium.train_autoencoder(N_train=N_ext)
    AutoencoderMedium.train_calibration_model(coef=ae_med_coef)


    """ (3) Test the performance """
    N_test = 250
    path_test = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/3_NETS/TEST/0')
    coef_test = np.loadtxt(os.path.join(path_test, 'test_coef.txt'))
    PSFs_test = load_files(path_test, N=N_test, file_list=list_slices)
    PSFs_test[0] /= PEAK
    PSFs_test[1] /= PEAK
    _PSFs_test, downPSFs_test, downPSFs_test_flat = downsample_slicer_pixels(PSFs_test[1])

