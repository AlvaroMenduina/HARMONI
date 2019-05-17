"""
==========================================================
                    POP Machine Learning
==========================================================

Separate analysis of the AUTOENCODER for high-order NCPA calibration
"""

import os
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import zern_core as zern
from pyzdde.zdde import readBeamFile
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

    path_files = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITHOUT AE/TRAIN')

    N_low, N_high = 5, 4
    N_PSF = 3000

    # ================================================================================================================ #
    #                                              LOAD LOW ORDER TRAINING SETS                                        #
    # ================================================================================================================ #

    path_low = os.path.join(path_files, 'LOW')
    zern_coefs_low = np.loadtxt(os.path.join(path_low, 'coef_low.txt'))
    PSFs_low = load_files(path_low, N=N_PSF, file_list=list_slices)

    PEAK = np.max(PSFs_low[1][:, 0])

    # ================================================================================================================ #
    #                                          TRAINING SET FOR THE AUTOENCODER                                        #
    # ================================================================================================================ #

    ### Zernike Coefficients for the Zemax macros
    N_auto = 2500
    N_ext = N_auto - 50
    path_auto = os.path.abspath('H:/POP/NYQUIST/HIGH ORDERS/WITH AE')
    ae_coefs = np.loadtxt(os.path.join(path_auto, 'TRAINING_BOTH', 'autoencoder_coef1.txt'))

    # Subtract the LOW orders
    ae_low_coef, ae_high_coef = path_auto[:, :N_low], path_auto[:, N_low:]
    extra_zeros = np.zeros((N_auto, N_low))
    only_high = np.concatenate((extra_zeros, ae_high_coef), axis=1)

    ### Load the TRAINING sets
    # NOISY: Both LOW and HIGH ("Features")
    PSFs_AE = load_files(os.path.join(path_auto, 'TRAINING_BOTH'), N=N_auto, file_list=list_slices)
    PSFs_AE[0] /= PEAK
    PSFs_AE[1] /= PEAK
    _PSFs_AE, downPSFs_AE, downPSFs_AE_flat = downsample_slicer_pixels(PSFs_AE[1])

    # CLEAN: Only HIGH ("Targets")
    PSFs_AE_high = load_files(os.path.join(path_auto, 'TRAINING_HIGH'), N=N_auto, file_list=list_slices)
    PSFs_AE_high[0] /= PEAK
    PSFs_AE_high[1] /= PEAK
    _PSFs_AE_high, downPSFs_AE_high, downPSFs_AE_high_flat = downsample_slicer_pixels(PSFs_AE_high[1])

    ### Separate PSFs into TRAINING and TESTING datasets
    train_noisy = downPSFs_AE_flat[:N_ext]
    train_clean = downPSFs_AE_high_flat[:N_ext]

    test_noisy = downPSFs_AE_flat[N_ext:]
    test_clean = downPSFs_AE_high_flat[N_ext:]

    ### Define the AUTOENCODER architecture
    from keras.layers import Dense
    from keras.models import Sequential, Model, Input
    from keras import backend as K
    from numpy.linalg import norm as norm
    from keras.utils.vis_utils import plot_model

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

    ### Run the TRAINING
    AE.fit(train_noisy, train_clean,
           epochs=epochs, batch_size=batch, shuffle=True, verbose=2,
           validation_data=(test_noisy, test_clean))

    decoded = AE.predict(test_noisy)

    # Make sure the training has succeeded by checking the residuals
    residuals = np.mean(norm(np.abs(decoded - test_clean), axis=-1))
    total = np.mean(norm(np.abs(test_clean), axis=-1))
    print(residuals / total * 100)

    # ================================================================================================================ #
    #                                             ANALYSING THE ENCODED IMAGES                                         #
    # ================================================================================================================ #

    ### Define the ENCODER to access the CODE
    input_img = Input(shape=(input_dim,))
    encoded_layer1 = AE.layers[0]
    encoded_layer2 = AE.layers[1]
    encoded_layer3 = AE.layers[2]
    encoded_layer4 = AE.layers[3]
    encoder = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
    encoder.summary()
    encoded_images = encoder.predict(train_noisy)

    # Show the CODE images
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

    # ================================================================================================================ #
    #                                  ANALYSING THE FEATURES IN THE TRAINING SET                                      #
    # ================================================================================================================ #

    from sklearn.decomposition import PCA
    from scipy.optimize import least_squares as lsq

    def features_training(num_images=1):
        """
        Function to analyse the features of the TRAINING set of the autoencoder
        """

        norm_coef = []
        losses_focus, peaks_focus, mins_focus = [], [], []
        losses_defocus, peaks_defocus, mins_defocus = [], [], []

        # ============================================================================================================ #
        ### Light Loss - see how the Low Orders modify the total intensity
        for j in range(N_ext):

            low_orders = ae_coefs[j, :N_low]
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

        # ============================================================================================================ #

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

        # ============================================================================================================ #
        ### Compare the removed features to the true difference
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

        # ============================================================================================================ #
        ### Show some examples of the true features

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
