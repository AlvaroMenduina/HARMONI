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
from numpy.random import RandomState
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
    from keras.layers import Dense
    from keras.models import Sequential, Model, Input
    from keras import backend as K
    from numpy.linalg import norm as norm

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
        if i < 3:
            ax.get_xaxis().set_visible(False)
        if i >= 3:
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

    j_non_zeros = [0, 1, 3, 4, 5, 6, 10, 12, 15]
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
    encoded_images_low = encoder_low.predict(train_noisy)

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
    for i in range(N_high):
        a_i = high_orders[:, i]
        norm_i = norm(high_orders, axis=1)
        for j, j_nz in enumerate(j_non_zeros):
            ax = plt.subplot(N_high, N_non_zeros, i*N_non_zeros + j + 1)
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

    pca = PCA(n_components=2)
    plt.figure()
    for i, j in enumerate(j_non_zeros):
        enc_low_ = np.concatenate((low_orders, enc_foc[:, j:j+1]), axis=1)
        pca.fit(X=enc_low_)
        components = pca.components_
        p_new = np.dot(enc_low_, components.T)
        ax = plt.subplot(2, 5, i + 1)
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
    file_name = os.path.join(coef_path1, 'remaining_iter%d.txt' % (k + 1))
    np.savetxt(file_name, remaining, fmt='%.5f')

    # Extra stuff
    ###====
    #
    # from sklearn.decomposition import PCA
    #
    # N_comp = N_high
    # pca = PCA(n_components=2)
    # pca.fit(X=enc_foc)
    # components = pca.components_.reshape((N_comp, 4, 4))
    # variance_ratio = pca.explained_variance_ratio_
    # total_variance = np.sum(variance_ratio)
    #
    # import itertools
    # kk = [k for k in range(2000)]
    # Jac = []
    # pairs_k = []
    # pairs_a = {0: [], 1: [], 2: [], 3: []}
    # pairs_pix = {0: [], 1: [], 2: [], 3: []}
    # d_pi_d_aj = {0: [], 1: [], 2: [], 3: []}
    # for (k1, k2) in list(itertools.combinations(kk, 2)):
    #     pix1, pix2 = enc_foc[k1], enc_foc[k2]
    #     delta_pix = pix2 - pix1
    #
    #     for j in range(N_high):     # Loop over the Aberrations
    #         h1 = high_orders[k1, j]
    #         h2 = high_orders[k2, j]
    #         dh = h2 - h1
    #
    #         if 1e-6 < np.abs(dh) < 1e-3:    # If dh small enough
    #             pairs_k.append([k1, k2])
    #             pairs_a[j].append(h1)
    #             # print("\nDelta Aberr: ", j, dh)
    #
    #             delta_p = []
    #             p_pix = []
    #             for i in [1, 5, 6, 7, 8, 13, 15]:       # Loop over the pixels
    #                 delta_p.append(delta_pix[i]/ dh)
    #                 p_pix.append(pix1[i])
    #             d_pi_d_aj[j].append(delta_p)
    #             pairs_pix[j].append(p_pix)
    #
    # for j in range(N_high):
    #     pix = np.array(pairs_pix[j])
    #     der = np.array(d_pi_d_aj[j])
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(pairs_a[j], pix[:, 3], der[:, 3], s=2)
    #     # plt.scatter(pix[:, 5], der[:, 5], s=2)
    #     ax.set_zlim([-1000, 1000])
    # plt.show()
    #
    # pairs_k = []
    # pairs_a = {0: [], 1: [], 2: [], 3: []}
    # pairs_pix = {0: [], 1: [], 2: [], 3: []}
    # d_pi_d_aj = {0: [], 1: [], 2: [], 3: []}
    # for (k1, k2) in list(itertools.combinations(kk, 2)):
    #     pix1, pix2 = enc_foc[k1], enc_foc[k2]
    #     delta_pix = pix2 - pix1
    #
    #     for i in [1, 5, 6, 7, 8, 13, 15]:  # Loop over the pixels
    #         delta = delta_pix[i]
    #
    #         if np.abs(delta)