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


if __name__ == "__main__":

    #FIXME: look at correlations between the pixels of the encoded images and the norm of aberrations


    N_low, N_high = 5, 4

    aa = [0.25, 0.15, 0.10, 0.05]
    NN = [1000, 1000, 1000, 1000]
    N_auto = int(np.sum(NN))

    coef_list = [np.random.uniform(-a, a, size=(n, N_low + N_high)) for (a, n) in zip(aa, NN)]
    ae_coef = np.concatenate(coef_list, axis=0)

    path_auto = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG')
    # np.save(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef'), ae_coef)
    np.savetxt(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef.txt'), ae_coef, fmt='%.5f')

    ae_low_coef, ae_high_coef = ae_coef[:, :N_low], ae_coef[:, N_low:]
    only_high = np.concatenate((np.zeros((N_auto, N_low)), ae_high_coef), axis=1)
    only_low = np.concatenate((ae_low_coef, np.zeros((N_auto, N_high))), axis=1)
    np.savetxt(os.path.join(path_auto, 'TRAINING_LOW', 'autoencoder_coef.txt'), only_low, fmt='%.5f')
    np.savetxt(os.path.join(path_auto, 'TRAINING_HIGH', 'autoencoder_coef.txt'), only_high, fmt='%.5f')

    ### Load the TRAINING sets
    # NOISY: Both LOW and HIGH ("Features")
    PSFs_AE = load_files(os.path.join(path_auto, 'TRAINING_BOTH'), N=N_auto, file_list=list_slices)
    PEAK = np.max(PSFs_AE[1])
    PSFs_AE[0] /= PEAK
    PSFs_AE[1] /= PEAK
    _PSFs_AE, downPSFs_AE, downPSFs_AE_flat = downsample_slicer_pixels(PSFs_AE[1])

    # CLEAN: Only HIGH ("Targets")
    PSFs_AE_high = load_files(os.path.join(path_auto, 'TRAINING_HIGH'), N=N_auto, file_list=list_slices)
    PSFs_AE_high[0] /= PEAK
    PSFs_AE_high[1] /= PEAK
    _PSFs_AE_high, downPSFs_AE_high, downPSFs_AE_high_flat = downsample_slicer_pixels(PSFs_AE_high[1])

    N_ext = N_auto - 100
    train_noisy, train_clean = downPSFs_AE_flat[:N_ext], downPSFs_AE_high_flat[:N_ext]

    test_noisy, test_clean = downPSFs_AE_flat[N_ext:], downPSFs_AE_high_flat[N_ext:]

    ### Define the AUTOENCODER architecture
    input_dim = 2*N_crop**2
    encoding_dim = 32
    epochs = 2000
    batch = 32

    K.clear_session()
    AE_high = Sequential()
    AE_high.add(Dense(16 * encoding_dim, input_shape=(input_dim, ), activation='relu'))
    AE_high.add(Dense(4 * encoding_dim, activation='relu'))
    AE_high.add(Dense(2 * encoding_dim, activation='relu'))
    AE_high.add(Dense(encoding_dim, activation='relu'))
    AE_high.add(Dense(2 * encoding_dim, activation='relu'))
    AE_high.add(Dense(4 * encoding_dim, activation='relu'))
    AE_high.add(Dense(input_dim, activation='sigmoid'))
    AE_high.summary()
    AE_high.compile(optimizer='adam', loss='mean_squared_error')

    ### Run the TRAINING
    AE_high.fit(train_noisy, train_clean,
           epochs=epochs, batch_size=batch, shuffle=True, verbose=2,
           validation_data=(test_noisy, test_clean))

    decoded = AE_high.predict(test_noisy)

    # Make sure the training has succeeded by checking the residuals
    residuals = np.mean(norm(np.abs(decoded - test_clean), axis=-1))
    total = np.mean(norm(np.abs(test_clean), axis=-1))
    print(residuals / total * 100)

    input_img = Input(shape=(input_dim,))
    encoded_layer1, encoded_layer2 = AE_high.layers[0], AE_high.layers[1]
    encoded_layer3, encoded_layer4 = AE_high.layers[2], AE_high.layers[3]
    encoder_high = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
    encoder_high.summary()
    encoded_images = encoder_high.predict(train_noisy)

    ### Use the ENCODED data as training set
    high_coef_train, high_coef_test = ae_high_coef[:N_ext], ae_high_coef[N_ext:]
    high_psf_train, high_psf_test = encoded_images.copy(),  encoder_high.predict(test_noisy)

    ### MLP Regressor for HIGH orders (TRAINED ON ENCODED)
    N_layer = (200, 100, 50)
    N_iter = 5000
    high_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu', solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9, warm_start=True, alpha=1e-2, random_state=1234)

    high_model.fit(X=high_psf_train, y=high_coef_train)

    high_guessed = high_model.predict(X=high_psf_test)
    print("\nHIGH model guesses: \n", high_guessed[:5])
    print("\nTrue Values: \n", high_coef_test[:5])
    high_rms0, high_rms = evaluate_wavefront_performance(N_high, high_coef_test, high_guessed,
                                                       zern_list=zern_list_high, show_predic=False)

    # ================================================================================================================ #
    #                                       ANALYSIS OF THE ENCODER FEATURES                                           #
    # ================================================================================================================ #
    N_enc = 16
    enc_foc, enc_defoc = encoded_images[:, :N_enc], encoded_images[:, N_enc:]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(enc_foc[:25], cmap='hot')
    ax1.set_title('Nominal PSF')
    ax1.set_ylabel('Sample')
    ax1.set_xlabel('Pixel Feature')
    plt.colorbar(im1, ax=ax1)
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(enc_defoc[:25], cmap='hot')
    ax2.set_title('Defocused PSF')
    ax2.set_xlabel('Pixel Feature')
    plt.colorbar(im2)
    plt.show()

    # ================================================================================================================ #
    # FOCUSED

    low_orders, high_orders = ae_low_coef[:N_ext], ae_high_coef[:N_ext]
    j_non_zeros = [-1 if all(enc_foc[:,i]==0.0) else i for i in range(N_enc)]
    j_non_zeros = [x for x in j_non_zeros if x!= -1]

    j_non_zeros = [1, 6, 7, 8, 13, 15]
    N_non_zeros = len(j_non_zeros)

    plt.figure()
    colors_high = ['lightblue', 'crimson', 'black', 'lightgreen']
    for i in range(N_high):
        a_i = high_orders[:, i]
        norm_i = norm(high_orders, axis=1)
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_high, N_non_zeros, i*N_non_zeros + j + 1)
            i_sort = np.argsort(norm_i)
            sc = ax.scatter(a_i[i_sort], enc_foc[:, j_nz][i_sort], s=1, color=colors_high[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_high - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_high - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    colors_low = ['lightblue', 'crimson', 'black', 'lightgreen', 'orange']
    for i in range(N_low):
        a_i = low_orders[:, i]
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_low, N_non_zeros, i*N_non_zeros + j + 1)
            plt.scatter(a_i, enc_foc[:, j_nz], s=1, color=colors_low[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_low - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_low - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    from sklearn.decomposition import PCA

    N_comp = N_high
    pca = PCA(n_components=2)
    plt.figure()
    for i, j in enumerate([1, 6, 7, 8, 13, 15]):
        enc_high_ = np.concatenate((high_orders, enc_foc[:, j:j+1]), axis=1)
        pca.fit(X=enc_high_)
        components = pca.components_
        p_new = np.dot(enc_high_, components.T)
        ax = plt.subplot(2, 3, i + 1)
        # plt.scatter(p_new[:,0], p_new[:,1], s=2)
        plt.scatter(np.dot(high_orders, components[0,:N_high]), enc_foc[:, j:j+1], s=2)
        ax.set_title('Pixel %d' %j)
        ax.set_xlabel(r'PCA Aberration [$\lambda$]')
        if i == 0 or i==3:
            ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    x = np.linspace(0, 1, 1000)
    y = 1.25 * x + np.random.normal(0, 0.1, size=1000)
    z = np.array([x,x, y]).T
    pca = PCA(n_components=2)
    pca.fit(X=z)
    components = pca.components_


    # ================================================================================================================ #
    # DEFOCUSED

    j_non_zeros = [0, 1, 5, 6, 10, 15]
    N_non_zeros = len(j_non_zeros)

    plt.figure()
    colors_high = ['lightblue', 'crimson', 'black', 'lightgreen']
    for i in range(N_high):
        a_i = high_orders[:, i]
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_high, N_non_zeros, i*N_non_zeros + j + 1)
            plt.scatter(a_i, enc_defoc[:, j_nz], s=1, color=colors_high[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_high - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_high - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    colors_low = ['lightblue', 'crimson', 'black', 'lightgreen', 'orange']
    for i in range(N_low):
        a_i = low_orders[:, i]
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_low, N_non_zeros, i*N_non_zeros + j + 1)
            plt.scatter(a_i, enc_defoc[:, j_nz], s=1, color=colors_low[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_low - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_low - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    # ================================================================================================================ #
    #                                        ANALYSIS OF THE IMAGE FEATURES                                           #
    # ================================================================================================================ #
    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    def features_training(coefs, datasets):
        """
        Function to analyse the features of the TRAINING set of the autoencoder
        """

        norm_coef = []
        losses_focus, peaks_focus, mins_focus = [], [], []
        losses_defocus, peaks_defocus, mins_defocus = [], [], []

        # ============================================================================================================ #
        ### Light Loss - see how the Low Orders modify the total intensity
        for j in range(datasets[0].shape[0]):
            norm_coef.append(np.linalg.norm(coefs[j]))
            input_focus = datasets[0][j, :N_crop**2].reshape((N_crop, N_crop))
            output_focus = datasets[1][j, :N_crop**2].reshape((N_crop, N_crop))
            removed_features_focus = input_focus - output_focus
            loss_focus = np.sum(removed_features_focus)
            losses_focus.append(loss_focus)
            peaks_focus.append(np.max(removed_features_focus))
            mins_focus.append(np.min(removed_features_focus))

            input_defocus = datasets[0][j, N_crop**2:].reshape((N_crop, N_crop))
            output_defocus = datasets[1][j, N_crop**2:].reshape((N_crop, N_crop))
            removed_features_defocus = input_defocus - output_defocus
            loss_defocus = np.sum(removed_features_defocus)
            losses_defocus.append(loss_defocus)
            peaks_defocus.append(np.max(removed_features_defocus))
            mins_defocus.append(np.min(removed_features_defocus))
        norm_coef = np.array(norm_coef)

        print("\nStatistics for Focus:")
        print("Sum of MAX: ", np.mean(peaks_focus))
        print("Sum of MIN: ", np.mean(mins_focus))
        print("Sum of TOT: ", np.mean(losses_focus))
        print("\nStatistics for Defocus:")
        print("Sum of MAX: ", np.mean(peaks_defocus))
        print("Sum of MIN: ", np.mean(mins_defocus))
        print("Sum of TOT: ", np.mean(losses_defocus))

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # Focused PSF
        p_sort = np.argsort(peaks_focus)
        ax1.scatter(norm_coef[p_sort], np.sort(peaks_focus),
                    color=cm.bwr(np.linspace(0.5 + np.min(peaks_focus), 1, N_ext)), s=4, label='Maxima')
        m_sort = np.argsort(mins_focus)
        ax1.scatter(norm_coef[m_sort], np.sort(mins_focus),
                    color=cm.bwr(np.linspace(0, 0.5, N_ext)), s=4, label='Minima')
        loss_sort = np.argsort(losses_focus)
        ax1.legend(loc=3)
        leg = ax1.get_legend()
        leg.legendHandles[0].set_color('red')
        leg.legendHandles[1].set_color('blue')

        ax1.axhline(y=0.0, linestyle='--', color='black')
        ax1.set_title('Nominal PSF')
        ax1.set_ylabel(r'Light loss')
        ax1.set_ylim([-0.75, 0.75])         # For the High AE
        # ax1.set_ylim([-0.25, 0.25])         # For the Low AE
        ax1.set_xlim([0.0, 0.5])

        ax3.scatter(norm_coef[loss_sort], np.sort(losses_focus), color='black', s=3, label='Total')
        ax3.legend(loc=3)
        ax3.axhline(y=0.0, linestyle='--', color='black')
        ax3.set_xlabel(r'Norm of low orders $\Vert a_{low} \Vert$')
        ax3.set_ylabel(r'Light loss')
        ax3.set_ylim([-1.5, 0.25])              # For the High AE
        # ax3.set_ylim([-0.75, 0.25])              # For the Low AE
        ax3.set_xlim([0.0, 0.5])

        # Defocused PSF
        p_sort = np.argsort(losses_defocus)
        ax2.scatter(norm_coef[p_sort], np.sort(peaks_defocus),
                    color=cm.bwr(np.linspace(0.5 + np.min(peaks_defocus), 1, N_ext)), s=4, label='Maxima')
        m_sort = np.argsort(mins_defocus)
        ax2.scatter(norm_coef[m_sort], np.sort(mins_defocus),
                    color=cm.bwr(np.linspace(0, 0.5, N_ext)), s=4, label='Minima')
        loss_sort = np.argsort(losses_defocus)
        ax2.legend(loc=3)
        leg = ax2.get_legend()
        leg.legendHandles[0].set_color('red')
        leg.legendHandles[1].set_color('blue')

        ax2.axhline(y=0.0, linestyle='--', color='black')
        ax2.set_title('Defocused PSF')
        ax2.set_ylim([-0.75, 0.75])
        # ax2.set_ylim([-0.25, 0.25])
        ax2.set_xlim([0.0, 0.5])

        ax4.scatter(norm_coef[loss_sort], np.sort(losses_defocus), color='black', s=3, label='Total')
        ax4.legend(loc=3)
        ax4.axhline(y=0.0, linestyle='--', color='black')
        ax4.set_xlabel(r'Norm of low orders $\Vert a_{low} \Vert$')
        ax4.set_ylim([-1.5, 0.25])
        # ax4.set_ylim([-0.75, 0.25])
        ax4.set_xlim([0.0, 0.5])


        ### REMOVED FEATURES plots
        # Focused PSF
        # N_comp = coefs.shape[1]
        N_comp = 4
        removed_features = datasets[0][:, :N_crop**2] - datasets[1][:, :N_crop ** 2]
        pca = PCA(n_components=N_comp)
        pca.fit(X=removed_features)
        components = pca.components_.reshape((N_comp, N_crop, N_crop))
        variance_ratio = pca.explained_variance_ratio_
        total_variance = np.sum(variance_ratio)

        removed_features_defocus = datasets[0][:, N_crop ** 2:] - datasets[1][:, N_crop ** 2:]
        pca_defocus = PCA(n_components=N_comp)
        pca_defocus.fit(X=removed_features_defocus)
        components_defocus = pca_defocus.components_.reshape((N_comp, N_crop, N_crop))
        variance_ratio_defocus = pca_defocus.explained_variance_ratio_
        total_variance_defocus = np.sum(variance_ratio_defocus)

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

        for i in range(N_comp):
            ax = plt.subplot(2, N_comp, i+1+N_comp)
            plt.imshow(components_defocus[i], cmap='seismic', origin='lower')
            ax.set_title(r'PCA #%d [$\sigma^2_r=%.2f/%.2f$]' %(i+1, variance_ratio_defocus[i], total_variance_defocus))
            plt.colorbar(orientation="horizontal")
            cmin = min(components_defocus[i].min(), -components_defocus[i].max())
            plt.clim(cmin, -cmin)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        ### PCA for each of the Intensity ranges in the training set
        plt.figure()
        for k in range(4):
            if k != 3:
                removed_features = datasets[0][k*1000:(k+1)*1000, :N_crop ** 2] - datasets[1][k*1000:(k+1)*1000, :N_crop ** 2]
            if k == 3:
                removed_features = datasets[0][3000:, :N_crop ** 2] - datasets[1][3000:, :N_crop ** 2]
            pca = PCA(n_components=N_comp)
            pca.fit(X=removed_features)
            components = pca.components_.reshape((N_comp, N_crop, N_crop))
            variance_ratio = pca.explained_variance_ratio_
            total_variance = np.sum(variance_ratio)
            for i in range(N_comp):
                ax = plt.subplot(4, N_comp, k*N_comp + i + 1)
                plt.imshow(components[i], cmap='seismic', origin='lower')
                ax.set_title(r'PCA #%d [$\sigma^2_r=%.2f/%.2f$]' % (i + 1, variance_ratio[i], total_variance))
                # plt.colorbar(orientation="horizontal")
                cmin = min(components[i].min(), -components[i].max())
                plt.clim(cmin, -cmin)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:

                    ax.get_yaxis().set_visible(True)
                    ax.set_ylabel('Dataset %d' % k)

        from scipy.special import erf
        plt.figure()
        N_hist = 10
        colors = cm.coolwarm(np.linspace(0, 1, N_hist))
        for i in range(N_hist):
            tt = removed_features[i+100]
            tt = tt[np.argwhere(tt != 0.0)]
            # plot the cumulative histogram
            n, bins, patches = plt.hist(tt, density=True, histtype='step',color=colors[i],
                                       cumulative=True)

            patches[0].set_xy(patches[0].get_xy()[:-1])
        plt.axvline(0.0, color='black', linestyle='--')
        x = np.linspace(-0.25, 0.25, 500)
        mu, s = 0.0, 0.01
        z = (x - mu)/s
        fx = 0.5*(1 + erf(z/np.sqrt(2)))
        # plt.plot(x, fx, color='black', linestyle='-.')

        plt.ylim([0, 1])
        plt.xlabel('Pixel Value []')



        random_images = np.random.choice(1000, size=30, replace=False)
        print(random_images)
        plt.figure()
        mins = []
        for i, img_j in enumerate(random_images):
            ax = plt.subplot(5, 6, i + 1)
            im = removed_features[img_j].reshape((N_crop, N_crop))
            pp = ax.imshow(im, cmap='seismic')
            min_im = min(im.min(), -im.max())
            mins.append(np.abs(min_im))
            plt.colorbar(pp)
            pp.set_clim(min_im, -min_im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        print(sum(mins))

        plt.figure()
        mins = []
        for i, img_j in enumerate(random_images):
            ax = plt.subplot(5, 6, i + 1)
            im = removed_features[img_j+1000].reshape((N_crop, N_crop))
            pp = ax.imshow(im, cmap='seismic')
            min_im = min(im.min(), -im.max())
            mins.append(np.abs(min_im))
            plt.colorbar(pp)
            pp.set_clim(min_im, -min_im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        print(sum(mins))

        # plt.figure()
        # for i, img_j in enumerate(random_images):
        #     ax = plt.subplot(5, 6, i + 1)
        #     im = removed_features[img_j].reshape((N_crop, N_crop))
        #     im_fft = np.abs(fftshift(fft2(im)))**2
        #     pp = ax.imshow(im_fft)
        #     # plt.colorbar(pp)
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)

        plt.show()


    dataset_high = [train_noisy, train_clean]
    features_training(coefs=low_orders, datasets=dataset_high)

    # ================================================================================================================ #
    #                                                        ~~
    #                                              ~~ LOW ORDER NETWORK ~~                                             #
    #                                                        ~~
    # ================================================================================================================ #
    # CLEAN: Only LOW ("Targets")
    PSFs_AE_low = load_files(os.path.join(path_auto, 'TRAINING_LOW'), N=N_auto, file_list=list_slices)
    PSFs_AE_low[0] /= PEAK
    PSFs_AE_low[1] /= PEAK
    _PSFs_AE_low, downPSFs_AE_low, downPSFs_AE_low_flat = downsample_slicer_pixels(PSFs_AE_low[1])

    ### Separate PSFs into TRAINING and TESTING datasets
    train_noisy_low, test_noisy_low = downPSFs_AE_flat[:N_ext], downPSFs_AE_flat[N_ext:]
    train_clean_low, test_clean_low = downPSFs_AE_low_flat[:N_ext], downPSFs_AE_low_flat[N_ext:]

    AE_low = Sequential()
    AE_low.add(Dense(16 * encoding_dim, input_shape=(input_dim, ), activation='relu'))
    AE_low.add(Dense(4 * encoding_dim, activation='relu'))
    AE_low.add(Dense(2 * encoding_dim, activation='relu'))
    AE_low.add(Dense(encoding_dim, activation='relu'))
    AE_low.add(Dense(2 * encoding_dim, activation='relu'))
    AE_low.add(Dense(4 * encoding_dim, activation='relu'))
    AE_low.add(Dense(input_dim, activation='sigmoid'))
    AE_low.summary()
    AE_low.compile(optimizer='adam', loss='mean_squared_error')

    ### Run the TRAINING
    AE_low.fit(train_noisy_low, train_clean_low, epochs=epochs, batch_size=batch, shuffle=True, verbose=2,
           validation_data=(test_noisy_low, test_clean_low))

    decoded_low = AE_low.predict(test_noisy_low)

    # Make sure the training has succeeded by checking the residuals
    residuals = np.mean(norm(np.abs(decoded_low - test_clean_low), axis=-1))
    total = np.mean(norm(np.abs(test_clean_low), axis=-1))
    print(residuals / total * 100)

    ### Define the ENCODER to access the CODE
    input_img = Input(shape=(input_dim,))
    encoded_layer1, encoded_layer2 = AE_low.layers[0], AE_low.layers[1]
    encoded_layer3, encoded_layer4 = AE_low.layers[2], AE_low.layers[3]
    encoder_low = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
    encoder_low.summary()
    encoded_images_low = encoder_low.predict(train_noisy_low)

    ### Use the ENCODED data as training set
    low_coef_train, low_coef_test = ae_low_coef[:N_ext], ae_low_coef[N_ext:]
    low_psf_train, low_psf_test = encoded_images_low.copy(),  encoder_low.predict(test_noisy)

    ### MLP Regressor for HIGH orders (TRAINED ON ENCODED)
    low_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu', solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9, warm_start=True, alpha=1e-2, random_state=1234)

    low_model.fit(X=low_psf_train, y=low_coef_train)
    low_guessed = low_model.predict(X=low_psf_test)
    print("\nLOW model guesses: \n", low_guessed[:5])
    print("\nTrue Values \n", low_coef_test[:5])
    low_rms0, low_rms = evaluate_wavefront_performance(N_low, low_coef_test, low_guessed,
                                                       zern_list=zern_list_low, show_predic=False)


    N_enc = 16
    enc_foc, enc_defoc = encoded_images_low[:, :N_enc], encoded_images_low[:, N_enc:]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(enc_foc[:25], cmap='hot')
    ax1.set_title('Nominal PSF')
    ax1.set_ylabel('Sample')
    ax1.set_xlabel('Pixel Feature')
    plt.colorbar(im1, ax=ax1)
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(enc_defoc[:25], cmap='hot')
    ax2.set_title('Defocused PSF')
    ax2.set_xlabel('Pixel Feature')
    plt.colorbar(im2)
    plt.show()

    j_non_zeros = [-1 if all(enc_foc[:,i]==0.0) else i for i in range(N_enc)]
    j_non_zeros = [x for x in j_non_zeros if x!= -1]

    j_non_zeros.remove(0)
    j_non_zeros.remove(14)
    N_non_zeros = len(j_non_zeros)

    plt.figure()
    colors_high = ['lightblue', 'crimson', 'black', 'lightgreen']
    j_non_zeros = [1, 4, 9, 10, 11, 13]
    for i in range(N_high):
        a_i = high_orders[:, i]
        norm_i = norm(high_orders, axis=1)
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_high, 6, i*6 + j + 1)
            i_sort = np.argsort(norm_i)
            # colors = cm.seismic(np.linspace(0, 1, N_ext))
            sc = ax.scatter(a_i[i_sort], enc_foc[:, j_nz][i_sort], s=1, color=colors_high[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_high - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_high - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    colors_low = ['lightblue', 'crimson', 'black', 'lightgreen', 'orange']
    for i in range(N_low):
        a_i = low_orders[:, i]
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_low, 6, i*6 + j + 1)
            plt.scatter(a_i, enc_foc[:, j_nz], s=1, color=colors_low[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_low - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_low - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    pca = PCA(n_components=2)
    plt.figure()
    for i, j in enumerate(j_non_zeros):
        enc_low_ = np.concatenate((low_orders, enc_foc[:, j:j+1]), axis=1)
        pca.fit(X=enc_low_)
        components = pca.components_
        p_new = np.dot(enc_low_, components.T)
        ax = plt.subplot(2, 3, i + 1)
        # plt.scatter(p_new[:,0], p_new[:,1], s=2)
        plt.scatter(np.dot(low_orders, components[0,:N_low]), enc_foc[:, j:j+1], s=2)
        ax.set_title('Pixel %d' %j)
        if i < 3:
            ax.get_xaxis().set_visible(False)
        if i >= 3:
            ax.set_xlabel(r'PCA Aberration [$\lambda$]')
        if i == 0 or i==3:
            ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    dataset_low = [train_noisy_low, train_clean_low]
    features_training(coefs=high_orders, datasets=dataset_low)

    # ================================================================================================================ #
    #                                            TEST THE PERFORMANCE
    # ================================================================================================================ #
    N_test = 250
    path_test = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/0')
    coef_test = np.loadtxt(os.path.join(path_test, 'coef_test.txt'))
    PSFs_test = load_files(path_test, N=N_test, file_list=list_slices)
    PSFs_test[0] /= PEAK
    PSFs_test[1] /= PEAK
    _PSFs_test, downPSFs_test, downPSFs_test_flat = downsample_slicer_pixels(PSFs_test[1])

    rms_encoder = []
    # Initial RMS
    _r, _rms0 = evaluate_wavefront_performance(N_low + N_high, coef_test, np.zeros_like(coef_test),
                                                       zern_list=zern_list_low, show_predic=False)
    rms_encoder.append(_rms0)

    ### LOW orders
    encoded_test_low = encoder_low.predict(downPSFs_test_flat)
    low_orders = low_model.predict(X=encoded_test_low)
    print("\nTrue Coefficients")
    print(coef_test[:5, :N_low])
    print(low_orders[:5])
    l_rms0, low_orders_rms = evaluate_wavefront_performance(N_low, coef_test[:, :N_low], low_orders,
                                                       zern_list=zern_list_low, show_predic=False)

    ### HIGH orders
    encoded_test_high = encoder_high.predict(downPSFs_test_flat)
    high_orders = high_model.predict(X=encoded_test_high)
    print("\nTrue Coefficients")
    print(coef_test[:5, N_low:])
    print(high_orders[:5])
    h_rms0, high_orders_rms = evaluate_wavefront_performance(N_high, coef_test[:, N_low:], high_orders,
                                                       zern_list=zern_list_high, show_predic=False)

    all_orders = np.concatenate((low_orders, high_orders), axis=1)
    rr, all_orders_rms = evaluate_wavefront_performance(N_high + N_low, coef_test, all_orders,
                                                       zern_list=zern_list_high, show_predic=False)

    rms_encoder.append(all_orders_rms)

    remaining = coef_test - all_orders
    coef_path1 = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/1ALL')
    file_name = os.path.join(coef_path1, 'remaining_iter1.txt')
    np.savetxt(file_name, remaining, fmt='%.5f')

    # # # #  --------------------------------------------------------------------------------------------------  # # # #

    coef_test1 = np.loadtxt(os.path.join(coef_path1, 'remaining_iter1.txt'))
    PSFs_test1 = load_files(coef_path1, N=N_test, file_list=list_slices)
    PSFs_test1[0] /= PEAK
    PSFs_test1[1] /= PEAK
    _PSFs_test1, downPSFs_test1, downPSFs_test_flat1 = downsample_slicer_pixels(PSFs_test1[1])

    ### LOW orders
    encoded_test_low = encoder_low.predict(downPSFs_test_flat1)
    low_orders1 = low_model.predict(X=encoded_test_low)

    print("\nTrue Coefficients")
    print(coef_test1[:5, :N_low])
    print(low_orders1[:5])

    l_rms1, low_orders_rms1 = evaluate_wavefront_performance(N_low, coef_test1[:, :N_low], low_orders1,
                                                       zern_list=zern_list_low, show_predic=False)

    ### HIGH orders
    encoded_test_high = encoder_high.predict(downPSFs_test_flat1)
    high_orders1 = high_model.predict(X=encoded_test_high)

    print("\nTrue Coefficients")
    print(coef_test1[:5, N_low:])
    print(high_orders1[:5])
    h_rms1, high_orders_rms1 = evaluate_wavefront_performance(N_high, coef_test1[:, N_low:], high_orders1,
                                                       zern_list=zern_list_high, show_predic=False)

    all_orders = np.concatenate((low_orders1, high_orders1), axis=1)
    rr1, all_orders_rms1 = evaluate_wavefront_performance(N_high + N_low, coef_test1, all_orders,
                                                       zern_list=zern_list_high, show_predic=False)

    rms_encoder.append(all_orders_rms1)

    remaining1 = coef_test1 - all_orders

    coef_path2 = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/2ALL')
    file_name = os.path.join(coef_path2, 'remaining_iter2.txt')
    np.savetxt(file_name, remaining1, fmt='%.5f')

    # # # #  --------------------------------------------------------------------------------------------------  # # # #

    coef_test2 = np.loadtxt(os.path.join(coef_path2, 'remaining_iter2.txt'))
    PSFs_test2 = load_files(coef_path2, N=N_test, file_list=list_slices)
    PSFs_test2[0] /= PEAK
    PSFs_test2[1] /= PEAK
    _PSFs_test2, downPSFs_test2, downPSFs_test_flat2 = downsample_slicer_pixels(PSFs_test2[1])

    ### LOW orders
    encoded_test_low = encoder_low.predict(downPSFs_test_flat2)
    low_orders2 = low_model.predict(X=encoded_test_low)
    l_rms2, low_orders_rms2 = evaluate_wavefront_performance(N_low, coef_test2[:, :N_low], low_orders2,
                                                       zern_list=zern_list_low, show_predic=False)
    ### HIGH orders
    encoded_test_high = encoder_high.predict(downPSFs_test_flat2)
    high_orders2 = high_model.predict(X=encoded_test_high)
    h_rms2, high_orders_rms2 = evaluate_wavefront_performance(N_high, coef_test2[:, N_low:], high_orders2,
                                                       zern_list=zern_list_high, show_predic=False)

    all_orders = np.concatenate((low_orders2, high_orders2), axis=1)
    rr2, all_orders_rms2 = evaluate_wavefront_performance(N_high + N_low, coef_test2, all_orders,
                                                       zern_list=zern_list_high, show_predic=False)
    rms_encoder.append(all_orders_rms2)

    remaining2 = coef_test2 - all_orders
    # coef_path1 = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/1ALL')
    # file_name = os.path.join(coef_path1, 'remaining_iter1.txt')
    # np.savetxt(file_name, remaining, fmt='%.5f')

    # ================================================================================================================ #
    def hide_ticks(axes, visible=False):
        plt.setp(axes.get_xticklabels(), visible=visible)
        plt.setp(axes.get_yticklabels(), visible=visible)
        if not visible:
            axes.tick_params(axis=u'both', which=u'both', length=0)
        elif visible:
            axes.tick_params(axis=u'both', which=u'both', length=3)
    def plot_cross():
        plt.axvline(0.0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(0.0, color='black', linestyle='--', alpha=0.5)

    ### Final Results - Analysis of aberrations
    final_aberr = remaining2.copy().T
    cov_matrix = np.cov(final_aberr)
    N_t = N_low
    lims = [1.5*np.round(max(-1.2*x.min(), 1.2*x.max()), 2) for x in final_aberr]
    plt.figure()
    for j in range(N_t):
        c_j = final_aberr[j]
        for i in np.arange(j+1, N_t):
            c_i = final_aberr[i]
            k = (i-1)*(N_t) + j + 1
            print(i+1, j+1)
            ax = plt.subplot(N_t-1, N_t, k)
            x_label, y_label = zern_list_low[j] + r' [$\lambda$]', zern_list_low[i] + r' [$\lambda$]'
            x_lim, y_lim = lims[j], lims[i]

            corr = np.corrcoef(np.array([c_i, c_j]))[0, 1]
            plt.scatter(c_j, c_i, s=3, label=r'$\rho$=%.2f' %corr)

            plot_cross()
            ax.set_xlim([-0.075, 0.075])
            ax.set_ylim([-0.075, 0.075])
            hide_ticks(ax)
            # ax.legend()

            if i == N_low-1:
                ax.set_xlabel(x_label)
                if j == 0:
                    hide_ticks(ax, True)
            if j == 0:
                ax.set_ylabel(y_label)

    ijk = [(1, 0, 2), (2, 0, 3), (3, 0, 4),
           (2, 1, 8), (3, 1, 9),
           (3, 2, 14)]
    for (i, j, k) in ijk:

        ax = plt.subplot(N_t - 1, N_t, k)
        c_i, c_j = final_aberr[i+N_low], final_aberr[j+N_low]
        corr = np.corrcoef(np.array([c_i, c_j]))[0, 1]
        plt.scatter(c_i, c_j, s=3, color='crimson', label=r'$\rho$=%.2f' %corr)
        xlabel = zern_list_high[i] + r' [$\lambda$]'
        ylabel = zern_list_high[j] + r' [$\lambda$]'
        plot_cross()
        hide_ticks(ax)
        # ax.legend()
        # ax.set_xlim([-0.025, 0.025])
        # ax.set_ylim([-0.025, 0.025])
        ax.set_xlim([-0.075, 0.075])
        ax.set_ylim([-0.075, 0.075])
        if j == 0:
            ax.set_title(xlabel)
            # if i == 3:
            #     hide_ticks(ax, True)
            #     ax.yaxis.tick_right()
            #     ax.xaxis.tick_top()
        if i == 3:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(ylabel, fontsize='large')
            # ax.set_ylabel(ylabel,

    plt.show()

    N_t = N_high
    for j in range(N_t):
        c_j = final_aberr[j+N_low]
        for i in np.arange(j+1, N_t):
            c_i = final_aberr[i+N_low]
            k = (i-1)*(N_t) + j + 1
            print(i+1, j+1)
            ax = plt.subplot(N_t-1, N_t, k)
            x_label, y_label = zern_list_high[j] + r' [$\lambda$]', zern_list_high[i] + r' [$\lambda$]'
            x_lim, y_lim = lims[j], lims[i]

            plt.scatter(c_j, c_i, s=3, color='crimson')
            plt.axvline(0.0, color='black', linestyle='--', alpha=0.5)
            plt.axhline(0.0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlim([-0.025, 0.025])
            ax.set_ylim([-0.025, 0.025])

            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis=u'both', which=u'both', length=0)

            if i == N_t-1:
                ax.set_xlabel(x_label)
                if j == 0:
                    plt.setp(ax.get_xticklabels(), visible=True)
                    plt.setp(ax.get_yticklabels(), visible=True)
                    ax.tick_params(axis=u'both', which=u'both', length=3)
            if j == 0:
                ax.set_ylabel(y_label)
    plt.show()

    plt.figure(figsize=(6, 6))
    for i in range(N_low):
        c_i = final_aberr[i]
        for j in range(N_high):
            c_j = final_aberr[j + N_low]
            k = i*N_high + j + 1
            print(k)
            ax = plt.subplot(N_low, N_high, k)
            x_label, y_label = zern_list_low[i] + r' [$\lambda$]', zern_list_high[j] + r' [$\lambda$]'
            plt.scatter(c_j, c_i, s=3)
            plt.axvline(0.0, color='black', linestyle='--', alpha=0.5)
            plt.axhline(0.0, color='black', linestyle='--', alpha=0.5)
            # ax.set_xlim([-0.04, 0.04])
            ax.set_xlim([-0.1, 0.1])
            ax.set_ylim([-0.1, 0.1])
            # ax.set_ylim([-0.05, 0.05])

            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis=u'both', which=u'both', length=0)

            if i == N_low -1:
                ax.set_xlabel(y_label)
                if j == 0:
                    plt.setp(ax.get_xticklabels(), visible=True)
                    plt.setp(ax.get_yticklabels(), visible=True)
                    ax.tick_params(axis=u'both', which=u'both', length=3)
            if j == 0:
                ax.set_ylabel(x_label)
    plt.show()

    plt.figure()
    for i in range(N_low):
        ax = plt.subplot(1, N_low, i+1)
        ax.hist(final_aberr[i], histtype='step')
        xmin = min(final_aberr[i].min(), -final_aberr[i].max())
        ax.set_xlim([1.1*xmin, -1.1*xmin])
        plt.axvline(0.0, color='black', linestyle='--', alpha=0.5)
    plt.show()

    initial_mean_abs = wave_nom * np.mean(np.abs(coef_test.T), axis=1)
    final_mean_abs = wave_nom * np.mean(np.abs(final_aberr), axis=1)

    initial_mean = wave_nom * np.mean(coef_test.T, axis=1)
    final_mean = wave_nom * np.mean(final_aberr, axis=1)
    final_median = wave_nom * np.median(final_aberr, axis=1)
    final_std = wave_nom * np.std(final_aberr, axis=1)

    def shuffle_data(array):
        N_samples = array.shape[0]
        i = [j for j in range(N_samples)]
        np.random.shuffle(i)
        shuffled = array.copy()
        shuffled = shuffled[i]
        return shuffled

    AE_MI = Sequential()
    AE_MI.add(Dense(16 * encoding_dim, input_shape=(input_dim, ), activation='relu'))
    AE_MI.add(Dense(4 * encoding_dim, activation='relu'))
    AE_MI.add(Dense(2 * encoding_dim, activation='relu'))
    AE_MI.add(Dense(encoding_dim, activation='relu'))
    AE_MI.add(Dense(2 * encoding_dim, activation='relu'))
    AE_MI.add(Dense(4 * encoding_dim, activation='relu'))
    AE_MI.add(Dense(input_dim, activation='sigmoid'))
    AE_MI.summary()
    AE_MI.compile(optimizer='adam', loss='mean_squared_error')

    ### Run the TRAINING
    AE_MI.fit(train_noisy, train_clean,
           epochs=2, batch_size=batch, shuffle=True, verbose=2,
           validation_data=(test_noisy, test_clean))

    input_img = Input(shape=(input_dim,))
    encoded_layer1, encoded_layer2 = AE_MI.layers[0], AE_MI.layers[1]
    encoded_layer3, encoded_layer4 = AE_MI.layers[2], AE_MI.layers[3]
    encoder_MI = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
    encoder_MI.summary()


    x = encoder_MI.predict(train_noisy)[:,:16]
    y = train_noisy[:, :N_crop**2]


    mi = mutual_information((x, y), k=10)

    x0 = encoder_high.predict(train_clean)[:, :16]
    k_n = 5
    mi0 = entropy(x, k_n) - entropy(np.hstack((x, y)), k_n)
    print(mi0)

    epoch = [10, 20, 50]
    for epp in epoch:
        AE_MI.fit(train_noisy, train_clean,
                  epochs=epp, batch_size=batch, shuffle=True, verbose=0,
                  validation_data=(test_noisy, test_clean))
        x = encoder_MI.predict(train_clean)[:, :16]

        mi = entropy(x, k_n) - entropy(np.hstack((x, y)), k_n)
        print(mi)



    # ================================================================================================================ #
    #      COMPARISON WITH DECODED IMAGE TRAINING
    # ================================================================================================================ #

    rms_autoencoder = [_rms0]

    ### HIGH ORDER MODEL
    train_high_decoded = train_clean
    train_high_coef = ae_high_coef[:N_ext]

    test_high_decoded = AE_high.predict(downPSFs_test_flat)
    test_high_coef = coef_test[:, N_low:]

    high_model_encoded = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9,
                             warm_start=True, alpha=1e-2, random_state=1234)

    high_model_encoded.fit(X=train_high_decoded, y=train_high_coef)

    high_guessed_encoded = high_model_encoded.predict(X=test_high_decoded)
    print("\nHIGH model guesses:")
    print(high_guessed_encoded[:5])
    print("\nTrue Values")
    print(test_high_coef[:5])

    print('\n HIGH order Model:')
    high_rms0_enc, high_rms_enc = evaluate_wavefront_performance(N_high, test_high_coef, high_guessed_encoded,
                                                       zern_list=zern_list_high, show_predic=False)

    ### LOW ORDER MODEL
    train_low_decoded = train_clean_low
    train_low_coef = ae_low_coef[:N_ext]

    test_low_decoded = AE_low.predict(downPSFs_test_flat)
    test_low_coef = coef_test[:, :N_low]

    low_model_encoded = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                             solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9,
                             warm_start=True, alpha=1e-2, random_state=1234)

    low_model_encoded.fit(X=train_low_decoded, y=train_low_coef)

    low_guessed_encoded = low_model_encoded.predict(X=test_low_decoded)
    print("\nLOW model guesses:")
    print(low_guessed_encoded[:5])
    print("\nTrue Values")
    print(test_low_coef[:5])

    print('\n HIGH order Model:')
    low_rms0_enc, low_rms_enc = evaluate_wavefront_performance(N_low, test_low_coef, low_guessed_encoded,
                                                       zern_list=zern_list_low, show_predic=False)

    ### Combined
    both_encoded = np.concatenate((low_guessed_encoded, high_guessed_encoded), axis=1)
    pp, both_encoded_rms = evaluate_wavefront_performance(N_high + N_low, coef_test, both_encoded,
                                                       zern_list=zern_list_high, show_predic=False)

    rms_autoencoder.append(both_encoded_rms)

    remaining_encoded = coef_test - both_encoded

    coef_path_enc1 = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/1ALL_ENCODED')
    file_name = os.path.join(coef_path_enc1, 'remaining_iter1.txt')
    np.savetxt(file_name, remaining_encoded, fmt='%.5f')

    # # # #

    coef_test_enc1 = np.loadtxt(os.path.join(coef_path_enc1, 'remaining_iter1.txt'))
    PSFs_test_enc1 = load_files(coef_path_enc1, N=N_test, file_list=list_slices)
    PSFs_test_enc1[0] /= PEAK
    PSFs_test_enc1[1] /= PEAK
    _PSFs_test_enc1, downPSFs_test_enc1, downPSFs_test_enc_flat1 = downsample_slicer_pixels(PSFs_test_enc1[1])

    low_guessed_encoded = low_model_encoded.predict(X=AE_low.predict(downPSFs_test_enc_flat1))
    high_guessed_encoded = high_model_encoded.predict(X=AE_high.predict(downPSFs_test_enc_flat1))
    both_encoded = np.concatenate((low_guessed_encoded, high_guessed_encoded), axis=1)
    pp1, both_encoded_rms1 = evaluate_wavefront_performance(N_high + N_low, coef_test_enc1, both_encoded,
                                                       zern_list=zern_list_high, show_predic=False)

    rms_autoencoder.append(both_encoded_rms1)
    remaining_encoded1 = remaining_encoded - both_encoded

    coef_path_enc2 = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/2ALL_ENCODED')
    file_name = os.path.join(coef_path_enc2, 'remaining_iter2.txt')
    np.savetxt(file_name, remaining_encoded1, fmt='%.5f')

    # # # #

    coef_test_enc2 = np.loadtxt(os.path.join(coef_path_enc2, 'remaining_iter2.txt'))
    PSFs_test_enc2 = load_files(coef_path_enc2, N=N_test, file_list=list_slices)
    PSFs_test_enc2[0] /= PEAK
    PSFs_test_enc2[1] /= PEAK
    _PSFs_test_enc2, downPSFs_test_enc2, downPSFs_test_enc_flat2 = downsample_slicer_pixels(PSFs_test_enc2[1])

    low_guessed_encoded = low_model_encoded.predict(X=AE_low.predict(downPSFs_test_enc_flat2))
    high_guessed_encoded = high_model_encoded.predict(X=AE_high.predict(downPSFs_test_enc_flat2))
    both_encoded = np.concatenate((low_guessed_encoded, high_guessed_encoded), axis=1)
    pp2, both_encoded_rms2 = evaluate_wavefront_performance(N_high + N_low, coef_test_enc2, both_encoded,
                                                       zern_list=zern_list_high, show_predic=False)
    rms_autoencoder.append(both_encoded_rms2)


    # ================================================================================================================ #

    ### Plot results
    n = len(rms_encoder)
    rms_encoder_arr = wave_nom * np.array(rms_encoder)
    rms_autoencoder_arr = wave_nom * np.array(rms_autoencoder)
    colors = cm.coolwarm(np.linspace(0, 1, N_test))

    plt.figure()
    plt.subplot(1, 2, 1)
    i = 0
    plt.scatter(i * np.ones(N_test) + 0.025, np.sort(rms_autoencoder_arr[i]), color='coral', s=4, label=r'Reconstructed $x$')
    plt.scatter(i * np.ones(N_test) - 0.025, np.sort(rms_encoder_arr[i]), color='blue', s=4, label=r'Encoded $h$')
    for i in np.arange(1, n):
        plt.scatter(i * np.ones(N_test) + 0.025, np.sort(rms_autoencoder_arr[i]), color='coral', s=4)
        plt.scatter(i*np.ones(N_test) - 0.025, np.sort(rms_encoder_arr[i]), color='blue', s=4)

    plt.legend(title='Architecture')
    plt.ylim([0, 350])
    plt.ylabel('RMS [nm]')
    plt.xlabel('Iteration')

    med_aut = np.median(rms_autoencoder_arr[-1])
    med_enc = np.median(rms_encoder_arr[-1])
    plt.subplot(1, 2, 2)
    plt.hist(rms_autoencoder_arr[-1], histtype='step', color='coral', label=r'Reconstructed $x$')
    plt.axvline(x=med_aut, linestyle='--', color='coral', label='Median: %.1f nm' %med_aut)
    plt.hist(rms_encoder_arr[-1], histtype='step', color='blue', label=r'Encoded $h$')
    plt.axvline(x=med_enc, linestyle='--', color='blue', label='Median: %.1f nm' %med_enc)
    # plt.legend()
    plt.xlabel('Final RMS [nm]')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 0, 3, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title='Architecture')
    plt.show()

    rel_enc = [(before - after) / before for (before, after) in zip(rms_encoder_arr[0], rms_encoder_arr[-1])]
    rel_enc = np.mean(rel_enc)

    rel_autoenc = [(before - after) / before for (before, after) in zip(rms_autoencoder_arr[0], rms_autoencoder_arr[-1])]
    rel_autoenc = np.mean(rel_autoenc)

    # ================================================================================================================ #
    #                                                        ~~
    #                                          ~~ CONTRACTIVE AUTOENCODERS ~~                                          #
    #                                                        ~~
    # ================================================================================================================ #

    train_noisy, train_clean = downPSFs_AE_flat[:N_ext], downPSFs_AE_high_flat[:N_ext]
    test_noisy, test_clean = downPSFs_AE_flat[N_ext:], downPSFs_AE_high_flat[N_ext:]

    lam = 1e-5

    CAE_high = Sequential()
    CAE_high.add(Dense(16 * encoding_dim, input_shape=(input_dim, ), activation='relu'))
    CAE_high.add(Dense(4 * encoding_dim, activation='relu'))
    CAE_high.add(Dense(2 * encoding_dim, activation='relu'))
    CAE_high.add(Dense(encoding_dim, activation='relu', name='encoded'))
    CAE_high.add(Dense(2 * encoding_dim, activation='relu'))
    CAE_high.add(Dense(4 * encoding_dim, activation='relu'))
    CAE_high.add(Dense(input_dim, activation='sigmoid'))
    CAE_high.summary()

    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=CAE_high.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = CAE_high.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

        return mse + contractive

    CAE_high.compile(optimizer='adam', loss=contractive_loss)

    ### Run the TRAINING
    CAE_high.fit(train_noisy, train_clean,
           epochs=2000, batch_size=batch, shuffle=True, verbose=2,
           validation_data=(test_noisy, test_clean))

    decoded = CAE_high.predict(test_noisy)

    # Make sure the training has succeeded by checking the residuals
    residuals = np.mean(norm(np.abs(decoded - test_clean), axis=-1))
    total = np.mean(norm(np.abs(test_clean), axis=-1))
    print(residuals / total * 100)

    input_img = Input(shape=(input_dim,))
    encoded_layer1, encoded_layer2 = CAE_high.layers[0], CAE_high.layers[1]
    encoded_layer3, encoded_layer4 = CAE_high.layers[2], CAE_high.layers[3]
    Cencoder_high = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
    Cencoder_high.summary()
    encoded_images = Cencoder_high.predict(train_noisy)

    ### Use the ENCODED data as training set
    high_coef_train, high_coef_test = ae_high_coef[:N_ext], ae_high_coef[N_ext:]
    high_psf_train, high_psf_test = encoded_images.copy(),  Cencoder_high.predict(test_noisy)

    ### MLP Regressor for HIGH orders (TRAINED ON ENCODED)
    Chigh_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu', solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9, warm_start=True, alpha=1e-2, random_state=1234)

    Chigh_model.fit(X=high_psf_train, y=high_coef_train)

    high_guessed = Chigh_model.predict(X=high_psf_test)
    print("\nHIGH model guesses: \n", high_guessed[:5])
    print("\nTrue Values: \n", high_coef_test[:5])
    high_rms0, high_rms = evaluate_wavefront_performance(N_high, high_coef_test, high_guessed,
                                                       zern_list=zern_list_high, show_predic=False)

    # ================================================================================================================ #
    #                                       ANALYSIS OF THE ENCODER FEATURES                                           #
    # ================================================================================================================ #
    N_enc = 16
    enc_foc, enc_defoc = encoded_images[:, :N_enc], encoded_images[:, N_enc:]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(enc_foc[:25], cmap='hot')
    ax1.set_title('Nominal PSF')
    ax1.set_ylabel('Sample')
    ax1.set_xlabel('Pixel Feature')
    plt.colorbar(im1, ax=ax1)
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(enc_defoc[:25], cmap='hot')
    ax2.set_title('Defocused PSF')
    ax2.set_xlabel('Pixel Feature')
    plt.colorbar(im2)
    plt.show()

    # ================================================================================================================ #
    # FOCUSED

    low_orders, high_orders = ae_low_coef[:N_ext], ae_high_coef[:N_ext]
    j_non_zeros = [-1 if all(enc_foc[:,i]<=0.4) else i for i in range(N_enc)]
    j_non_zeros = [x for x in j_non_zeros if x!= -1]
    j_non_zeros = [1, 2, 4, 7, 8, 13]

    N_non_zeros = len(j_non_zeros)

    plt.figure()
    colors_high = ['lightblue', 'crimson', 'black', 'lightgreen']
    for i in range(N_high):
        a_i = high_orders[:, i]
        norm_i = norm(high_orders, axis=1)
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_high, N_non_zeros, i*N_non_zeros + j + 1)
            i_sort = np.argsort(norm_i)
            sc = ax.scatter(a_i[i_sort], enc_foc[:, j_nz][i_sort], s=1, color=colors_high[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_high - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_high - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    colors_low = ['lightblue', 'crimson', 'black', 'lightgreen', 'orange']
    for i in range(N_low):
        a_i = low_orders[:, i]
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_low, N_non_zeros, i*N_non_zeros + j + 1)
            plt.scatter(a_i, enc_foc[:, j_nz], s=1, color=colors_low[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_low - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_low - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    N_comp = N_high
    pca = PCA(n_components=2)
    plt.figure()
    for i, j in enumerate(j_non_zeros):
        enc_high_ = np.concatenate((high_orders, enc_foc[:, j:j+1]), axis=1)
        pca.fit(X=enc_high_)
        components = pca.components_
        p_new = np.dot(enc_high_, components.T)
        ax = plt.subplot(2, 3, i + 1)
        # plt.scatter(p_new[:,0], p_new[:,1], s=2)
        plt.scatter(np.dot(high_orders, components[0,:N_high]), enc_foc[:, j:j+1], s=2)
        ax.set_title('Pixel %d' %j)
        ax.set_xlabel(r'PCA Aberration [$\lambda$]')
        if i == 0 or i==3:
            ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    # ================================================================================================================ #
    #                                                        ~~
    #                                          ~~ CONTRACTIVE AUTOENCODERS ~~                                          #
    #                                                        ~~
    # ================================================================================================================ #

    ### Separate PSFs into TRAINING and TESTING datasets
    train_noisy_low, test_noisy_low = downPSFs_AE_flat[:N_ext], downPSFs_AE_flat[N_ext:]
    train_clean_low, test_clean_low = downPSFs_AE_low_flat[:N_ext], downPSFs_AE_low_flat[N_ext:]

    CAE_low = Sequential()
    CAE_low.add(Dense(16 * encoding_dim, input_shape=(input_dim, ), activation='relu'))
    CAE_low.add(Dense(4 * encoding_dim, activation='relu', name='h2'))
    CAE_low.add(Dense(2 * encoding_dim, activation='relu', name='h1'))
    CAE_low.add(Dense(encoding_dim, activation='relu', name='h'))
    CAE_low.add(Dense(2 * encoding_dim, activation='relu'))
    CAE_low.add(Dense(4 * encoding_dim, activation='relu'))
    CAE_low.add(Dense(input_dim, activation='sigmoid'))
    CAE_low.summary()

    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=CAE_low.get_layer('h').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = CAE_low.get_layer('h').output
        dh = h * (1 - h)  # N_batch x N_hidden
        # dh = K.one(h)

        W1 = K.variable(value=CAE_low.get_layer('h1').get_weights()[0])  # N x N_hidden
        W1 = K.transpose(W1)  # N_hidden x N
        h1 = CAE_low.get_layer('h1').output
        # dh1 = K.hard_sigmoid(h1)
        dh1 = h1 * (1 - h1)
        # print("h:", h.shape)
        # print("W:", W.shape)
        #
        # print("h1:", h1.shape)
        # print("W1:", W1.shape)
        #
        # WW1 = (K.dot(W, W1))*2
        # print("W.W1:", WW1.shape)


        d = K.expand_dims(dh) * W
        d1 = K.expand_dims(dh1) * W1
        dh_total = K.dot(d, d1)

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        # contractive = lam * K.sum((dh)** 2 * K.sum(np.dot(W, W1)** 2, axis=1), axis=1)
        contractive = lam * K.sum((dh_total)** 2)

        return mse + contractive

    CAE_low.compile(optimizer='adam', loss=contractive_loss)

    ### Run the TRAINING
    CAE_low.fit(train_noisy_low, train_clean_low,
           epochs=20, batch_size=batch, shuffle=True, verbose=2,
           validation_data=(test_noisy_low, test_clean_low))

    decoded = CAE_low.predict(test_noisy_low)

    # Make sure the training has succeeded by checking the residuals
    residuals = np.mean(norm(np.abs(decoded - test_clean_low), axis=-1))
    total = np.mean(norm(np.abs(test_clean_low), axis=-1))
    print(residuals / total * 100)

    input_img = Input(shape=(input_dim,))
    encoded_layer1, encoded_layer2 = CAE_low.layers[0], CAE_low.layers[1]
    encoded_layer3, encoded_layer4 = CAE_low.layers[2], CAE_low.layers[3]
    Cencoder_low = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
    Cencoder_low.summary()
    encoded_images = Cencoder_low.predict(train_noisy_low)

    ### Use the ENCODED data as training set
    low_coef_train, low_coef_test = ae_low_coef[:N_ext], ae_low_coef[N_ext:]
    low_psf_train, low_psf_test = encoded_images.copy(),  Cencoder_low.predict(test_noisy_low)

    ### MLP Regressor for low orders (TRAINED ON ENCODED)
    Clow_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu', solver='adam', max_iter=N_iter, verbose=True,
                             batch_size='auto', shuffle=True, tol=1e-9, warm_start=True, alpha=1e-2, random_state=1234)

    Clow_model.fit(X=low_psf_train, y=low_coef_train)

    low_guessed = Clow_model.predict(X=low_psf_test)
    print("\nlow model guesses: \n", low_guessed[:5])
    print("\nTrue Values: \n", low_coef_test[:5])
    low_rms0, low_rms = evaluate_wavefront_performance(N_low, low_coef_test, low_guessed,
                                                       zern_list=zern_list_low, show_predic=False)

    # ================================================================================================================ #

    N_test = 250
    path_test = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/0')
    coef_test = np.loadtxt(os.path.join(path_test, 'coef_test.txt'))
    PSFs_test = load_files(path_test, N=N_test, file_list=list_slices)
    PSFs_test[0] /= PEAK
    PSFs_test[1] /= PEAK
    _PSFs_test, downPSFs_test, downPSFs_test_flat = downsample_slicer_pixels(PSFs_test[1])

    rms_Cencoder = []
    # Initial RMS
    _r, _rms0 = evaluate_wavefront_performance(N_low + N_high, coef_test, np.zeros_like(coef_test),
                                                       zern_list=zern_list_low, show_predic=False)
    rms_Cencoder.append(_rms0)

    ### LOW orders
    encoded_test_low = Cencoder_low.predict(downPSFs_test_flat)
    low_orders = Clow_model.predict(X=encoded_test_low)
    print("\nTrue Coefficients")
    print(coef_test[:5, :N_low])
    print(low_orders[:5])
    l_rms0, low_orders_rms = evaluate_wavefront_performance(N_low, coef_test[:, :N_low], low_orders,
                                                       zern_list=zern_list_low, show_predic=False)

    ### HIGH orders
    encoded_test_high = Cencoder_high.predict(downPSFs_test_flat)
    high_orders = Chigh_model.predict(X=encoded_test_high)
    print("\nTrue Coefficients")
    print(coef_test[:5, N_low:])
    print(high_orders[:5])
    h_rms0, high_orders_rms = evaluate_wavefront_performance(N_high, coef_test[:, N_low:], high_orders,
                                                       zern_list=zern_list_high, show_predic=False)

    all_orders = np.concatenate((low_orders, high_orders), axis=1)
    rr, all_orders_rms = evaluate_wavefront_performance(N_high + N_low, coef_test, all_orders,
                                                       zern_list=zern_list_high, show_predic=False)

    rms_Cencoder.append(all_orders_rms)

    remainingC = coef_test - all_orders
    coef_path1 = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE LONG/TEST/1ALL_CAE')
    file_name = os.path.join(coef_path1, 'remaining_iter1.txt')
    np.savetxt(file_name, remainingC, fmt='%.5f')

    # ================================================================================================================ #

    ### Plot results
    n = len(rms_Cencoder)
    rms_Cencoder_arr = wave_nom * np.array(rms_Cencoder)
    rms_autoencoder_arr = wave_nom * np.array(rms_autoencoder)
    colors = cm.coolwarm(np.linspace(0, 1, N_test))

    plt.figure()
    plt.subplot(1, 2, 1)
    i = 0
    plt.scatter(i * np.ones(N_test) + 0.025, np.sort(rms_autoencoder_arr[i]), color='coral', s=4, label=r'Reconstructed $x$')
    plt.scatter(i * np.ones(N_test) - 0.025, np.sort(rms_Cencoder_arr[i]), color='blue', s=4, label=r'Encoded $h$')
    for i in np.arange(1, n):
        plt.scatter(i * np.ones(N_test) + 0.025, np.sort(rms_autoencoder_arr[i]), color='coral', s=4)
        plt.scatter(i*np.ones(N_test) - 0.025, np.sort(rms_Cencoder_arr[i]), color='blue', s=4)

    plt.legend(title='Architecture')
    plt.ylim([0, 350])
    plt.ylabel('RMS [nm]')
    plt.xlabel('Iteration')

    # ================================================================================================================ #
    #                                       ANALYSIS OF THE ENCODER FEATURES                                           #
    # ================================================================================================================ #
    N_enc = 16
    enc_foc, enc_defoc = encoded_images[:, :N_enc], encoded_images[:, N_enc:]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(enc_foc[:25], cmap='hot')
    ax1.set_title('Nominal PSF')
    ax1.set_ylabel('Sample')
    ax1.set_xlabel('Pixel Feature')
    plt.colorbar(im1, ax=ax1)
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(enc_defoc[:25], cmap='hot')
    ax2.set_title('Defocused PSF')
    ax2.set_xlabel('Pixel Feature')
    plt.colorbar(im2)
    plt.show()

    # ================================================================================================================ #
    # FOCUSED

    low_orders, high_orders = ae_low_coef[:N_ext], ae_high_coef[:N_ext]
    j_non_zeros = [-1 if all(enc_defoc[:,i]<=0.25) else i for i in range(N_enc)]
    j_non_zeros = [x for x in j_non_zeros if x!= -1]
    j_non_zeros.remove(7)

    # j_non_zeros = [1, 2, 6, 8, 11, 13]
    N_non_zeros = len(j_non_zeros)

    plt.figure()
    colors_high = ['lightblue', 'crimson', 'black', 'lightgreen']
    for i in range(N_high):
        a_i = high_orders[:, i]
        norm_i = norm(high_orders, axis=1)
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_high, N_non_zeros, i*N_non_zeros + j + 1)
            i_sort = np.argsort(norm_i)
            sc = ax.scatter(a_i[i_sort], enc_defoc[:, j_nz][i_sort], s=1, color=colors_high[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_high - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_high - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    colors_low = ['lightblue', 'crimson', 'black', 'lightgreen', 'orange']
    for i in range(N_low):
        a_i = low_orders[:, i]
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_low, N_non_zeros, i*N_non_zeros + j + 1)
            plt.scatter(a_i, enc_defoc[:, j_nz], s=1, color=colors_low[i])
            if i == 0:
                ax.set_title('Pixel %d' %j_nz)
            if i != N_low - 1:
                ax.get_xaxis().set_visible(False)
            if i == N_low - 1:
                ax.set_xlabel(r'Aberration [$\lambda$]')
            if j == 0:
                ax.set_ylabel('Pixel Value [ ]')
    plt.show()

    pca = PCA(n_components=2)
    plt.figure()
    for i, j in enumerate(j_non_zeros):
        enc_low_ = np.concatenate((low_orders, enc_defoc[:, j:j+1]), axis=1)
        pca.fit(X=enc_low_)
        components = pca.components_
        p_new = np.dot(enc_low_, components.T)
        ax = plt.subplot(2, N_non_zeros//2+1, i + 1)
        # plt.scatter(p_new[:,0], p_new[:,1], s=2)
        plt.scatter(np.dot(low_orders, components[0,:N_low]), enc_defoc[:, j:j+1], s=2)
        ax.set_title('Pixel %d' %j)
        if i < 3:
            ax.get_xaxis().set_visible(False)
        if i >= 3:
            ax.set_xlabel(r'PCA Aberration [$\lambda$]')
        if i == 0 or i==3:
            ax.set_ylabel('Pixel Value [ ]')
    plt.show()
