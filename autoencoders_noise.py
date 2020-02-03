"""

Could we use Autoencoders to remove noise effects from PSF images

"""

import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import zern_core as zern
from scipy.optimize import least_squares
import time

from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D
from keras.models import Model, Sequential, Input
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from numpy.linalg import norm as norm

# PARAMETERS
Z = 1.5                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 32                    # Pixels to crop the PSF
N_PIX = 512                 # Pixels for the Fourier arrays

# SPAXEL SCALE
WAVE = 1.5                 # 1.5 microns
ELT_DIAM = 39
MILIARCSECS_IN_A_RAD = 206265000

def rho_spaxel_scale(spaxel_scale=4.0, wavelength=1.0):
    """
    Compute the aperture radius necessary to have a
    certain SPAXEL SCALE [in mas] at a certain WAVELENGTH [in microns]

    That would be the aperture radius in an array ranging from [-1, 1] in physical length
    For example, if rho = 0.5, then the necessary aperture is a circle of half the size of the array

    We can use the inverse of that to get the "oversize" in physical units in our arrays to match a given scale
    :param spaxel_scale: [mas]
    :param wavelength: [microns]
    :return:
    """

    scale_rad = spaxel_scale / MILIARCSECS_IN_A_RAD
    rho = scale_rad * ELT_DIAM / (wavelength * 1e-6)
    return rho


def check_spaxel_scale(rho_aper, wavelength):
    """
    Checks the spaxel scale at a certain wavelength, for a given aperture radius
    defined for a [-1, 1] physical array
    :param rho_aper: radius of the aperture, relative to an array of size [-1, 1]
    :param wavelength: wavelength of interest (the PSF grows in size with wavelength, changing the spaxel scale)
    :return:
    """

    SPAXEL_RAD = rho_aper * wavelength / ELT_DIAM * 1e-6
    SPAXEL_MAS = SPAXEL_RAD * MILIARCSECS_IN_A_RAD
    print('%.2f mas spaxels at %.2f microns' %(SPAXEL_MAS, wavelength))


# SPAXEL SCALE
SPAXEL_MAS = 4.0    # [mas] spaxel scale at wave0
RHO_APER = rho_spaxel_scale(spaxel_scale=SPAXEL_MAS, wavelength=WAVE)
RHO_OBSC = 0.3 * RHO_APER  # Central obscuration (30% of ELT)

def invert_mask(x, mask):
    """
    Takes a vector X which is the result of masking a 2D with the Mask
    and reconstructs the 2D array
    Useful when you need to evaluate a Zernike Surface and most of the array is Masked
    """
    N = mask.shape[0]
    ij = np.argwhere(mask==True)
    i, j = ij[:,0], ij[:,1]
    result = np.zeros((N, N))
    result[i,j] = x
    return result

def invert_mask_datacube(x, mask):
    """
    Takes a vector X which is the result of masking a 2D with the Mask
    and reconstructs the 2D array
    Useful when you need to evaluate a Zernike Surface and most of the array is Masked
    """
    M = x.shape[-1]
    N = mask.shape[0]
    ij = np.argwhere(mask==True)
    i, j = ij[:,0], ij[:,1]
    result = np.zeros((M, N, N)).astype(np.float32)
    for k in range(M):
        result[k,i,j] = x[:,k]
    return result


def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True):
    """
    Computes the (Xc, Yc) coordinates of actuator centres
    inside a circle of rho_aper, assuming there are N_actuators
    along the [-1, 1] line

    :param N_actuators: Number of actuators along the [-1, 1] line
    :param rho_aper: Relative size of the aperture wrt [-1, 1]
    :param rho_obsc: Relative size of the obscuration
    :param radial: if True, we add actuators at the boundaries RHO_APER, RHO_OBSC
    :return: [act (list of actuator centres), delta (actuator separation)], max_freq (max spatial frequency we sense)
    """

    x0 = np.linspace(-1., 1., N_actuators, endpoint=True)
    delta = x0[1] - x0[0]
    N_in_D = 2*RHO_APER/delta
    print('%.2f actuators in D' %N_in_D)
    max_freq = N_in_D / 2                   # Max spatial frequency we can sense
    xx, yy = np.meshgrid(x0, x0)
    x_f = xx.flatten()
    y_f = yy.flatten()

    act = []    # List of actuator centres (Xc, Yc)
    for x_c, y_c in zip(x_f, y_f):
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        if r < (rho_aper - delta/2) and r > (rho_obsc + delta/2):   # Leave some margin close to the boundary
            act.append([x_c, y_c])

    if radial:  # Add actuators at the boundaries, keeping a constant angular distance
        for r in [rho_aper, rho_obsc]:
            N_radial = int(np.floor(2*np.pi*r/delta))
            d_theta = 2*np.pi / N_radial
            theta = np.linspace(0, 2*np.pi - d_theta, N_radial)
            # Super important to do 2Pi - d_theta to avoid placing 2 actuators in the same spot... Degeneracy
            for t in theta:
                act.append([r*np.cos(t), r*np.sin(t)])

    total_act = len(act)
    print('Total Actuators: ', total_act)
    return [act, delta], max_freq

def plot_actuators(centers):
    """
    Plot the actuators given their centre positions [(Xc, Yc), ..., ]
    :param centers: act from actuator_centres
    :return:
    """
    N_act = len(centers[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circ1 = Circle((0,0), RHO_APER, linestyle='--', fill=None)
    circ2 = Circle((0,0), RHO_OBSC, linestyle='--', fill=None)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    for c in centers[0]:
        ax.scatter(c[0], c[1], color='red', s=20)
    ax.set_aspect('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('%d actuators' %N_act)

def actuator_matrix(centres, alpha_pc=1, rho_aper=RHO_APER, rho_obsc=RHO_OBSC):
    """
    Computes the matrix containing the Influence Function of each actuator
    Returns a matrix of size [N_PIX, N_PIX, N_actuators] where each [N_PIX, N_PIX, k] slice
    represents the effect of "poking" one actuator

    Current model: Gaussian function

    :param centres: [act, delta] list from actuator_centres containing the centres and the spacing
    :param alpha: scaling factor to control the tail of the Gaussian and avoid overlap / crosstalk
    :param rho_aper:
    :param rho_obsc:
    :return:
    """
    alpha = 1 / np.sqrt(np.log(100 / alpha_pc))
    #TODO: Update the Model to something other than a Gaussian

    cent, delta = centres
    N_act = len(cent)
    matrix = np.empty((N_PIX, N_PIX, N_act))
    x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x0, x0)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    pupil = (rho <= rho_aper) & (rho >= rho_obsc)

    for k in range(N_act):
        xc, yc = cent[k][0], cent[k][1]
        r2 = (xx - xc) ** 2 + (yy - yc) ** 2
        matrix[:, :, k] = pupil * np.exp(-r2 / (alpha * delta) ** 2)

    mat_flat = matrix[pupil]

    return matrix, pupil, mat_flat

# ==================================================================================================================== #
#                                          Point Spread Function
# ==================================================================================================================== #

class PointSpreadFunctionFast(object):
    """
    Faster version of the PSF that uses a single FFT operation to generate multiple images
    """
    minPix, maxPix = (N_PIX + 1 - pix) // 2, (N_PIX + 1 + pix) // 2

    def __init__(self, matrices, amplitude=False):

        self.N_act = matrices[0].shape[-1]
        self.RBF_mat = matrices[0].copy()
        self.pupil_mask = matrices[1].copy()
        self.RBF_flat = matrices[2].copy()
        self.PEAK = self.peak_PSF()

    def generate_amplitude(self):
        centers, MAX_FREQ = actuator_centres(N_actuators=10)
        # plot_actuators(centers)
        return actuator_matrix(centers)[0]

    def amplitude_error(self, strength, N_cases):
        self.amplitude_matrix = self.generate_amplitude()

        N_amp = self.amplitude_matrix.shape[-1]
        new_pupil = np.empty((N_cases, N_PIX, N_PIX))
        RMS = []
        for k in range(N_cases):
            coef = strength * np.random.uniform(low=-1, high=1, size=N_amp)
            amp_map = np.dot(self.amplitude_matrix, coef)
            non_zero_map = amp_map[self.pupil_mask]
            amp0 = np.mean(non_zero_map)
            amp_map -= amp0
            rms = np.std(amp_map[self.pupil_mask])
            RMS.append(rms)
            new_pupil[k] = self.pupil_mask * (self.pupil_mask + amp_map)
            if k % 100 == 0:
                img = new_pupil[k]
                cmax = max(-np.min(img[self.pupil_mask] - 1), np.max(img[self.pupil_mask] - 1))
                plt.figure()
                plt.imshow(img, cmap='bwr')
                plt.clim(1 - cmax, 1 + cmax)
                plt.title(r'RMS deviations %.2f per cent' %(100*RMS[k]))
                plt.colorbar()
        self.RMS_amp = 100*np.mean(RMS)
        print("\nAmplitude errors with an average of %.3f per cent RMS" %(100*np.mean(RMS)))

        return new_pupil

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """
        im, strehl = self.compute_PSF(np.zeros((1, self.N_act)))
        return strehl

    def compute_PSF(self, coef, crop=True, amplitude=None):
        """
        Compute the PSF and the Strehl ratio
        """

        phase_flat = np.dot(self.RBF_flat, coef.T)
        phase_datacube = invert_mask_datacube(phase_flat, self.pupil_mask)
        if amplitude is None:
            pupil_function = self.pupil_mask[np.newaxis, :, :] * np.exp(1j * phase_datacube)
        elif amplitude is not None:
            N_cases = phase_datacube.shape[0]
            new_pupil = self.amplitude_error(amplitude, N_cases)
            pupil_function = new_pupil * np.exp(1j * phase_datacube)

        # print("\nSize of Complex Pupil Function array: %.3f Gbytes" %(pupil_function.nbytes / 1e9))

        image = (np.abs(fftshift(fft2(pupil_function), axes=(1,2))))**2

        try:
            image /= self.PEAK

        except AttributeError:
            # If self.PEAK is not defined, self.compute_PSF will compute the peak
            pass

        strehl = np.max(image, axis=(1, 2))

        if crop:
            image = image[:, self.minPix:self.maxPix, self.minPix:self.maxPix]
        else:
            pass
        # print("\nSize of Image : %.3f Gbytes" % (image.nbytes / 1e9))
        return image, strehl


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    """ (1) Define the ACTUATOR model for the WAVEFRONT """

    N_actuators = 25
    centers, MAX_FREQ = actuator_centres(N_actuators)
    N_act = len(centers[0])
    plot_actuators(centers)

    alpha_pc = 20
    rbf_mat = actuator_matrix(centers, alpha_pc=alpha_pc)        # Actuator matrix

    PSF = PointSpreadFunctionFast(rbf_mat)

    p = PSF.compute_PSF(np.zeros((1, N_act)))
    plt.figure()
    plt.imshow(np.log10(p[0][0]))
    plt.show()

    """ Generate a training set of random PSF with aberrations """

    def generate_PSF(PSF_model, N_samples, batch_size=1000):
        coef_rand = np.random.uniform(low=-Z, high=Z, size=(N_samples, N_act))
        N_batches = N_samples // batch_size
        batches = []
        print("\nGenerating a dataset of %d PSF examples" % N_samples)
        print("Task divided in %d batches of %d images" % (N_batches, batch_size))
        for k in range(N_batches):
            print("Batch #%d" % (k + 1))
            data = np.empty((batch_size, pix, pix, 1))
            nom_c = coef_rand[k * batch_size:(k + 1) * batch_size]
            img_nominal = PSF_model.compute_PSF(nom_c)
            data[:, :, :, 0] = img_nominal[0]
            batches.append(data)
        dataset = np.concatenate(batches, axis=0)

        return dataset, batches, coef_rand
    N_samples = 20000
    PSF_img, PSF_batches, PSF_coef = generate_PSF(PSF, N_samples)

    def photon_noise(PSF_array, low, high, copies=5):
        """
        Rescales the PSF according to Poisson noise
        to simulate photon noise:

        Noisy_PSF = 1/factor * Poisson {factor * PSF}
        :param PSF_array: Raw PSFs with Peaks of around 1.0
        :param factor: controls the amount of photon noise, while keeping the peaks close to 1.0
        :return:

        """
        print("Adding Photon Noise")
        N_samples = PSF_array.shape[0]
        pix = PSF_array.shape[1]
        copy_array = np.zeros((N_samples * copies, pix, pix, 1))
        noisy_PSF = np.zeros((N_samples * copies, pix, pix, 1))
        for k in range(N_samples):
            a_copy = PSF_array[k].copy()
            for j in range(copies):
                copy_array[copies*k + j] = a_copy
                factor = np.random.uniform(low=low, high=high)
                noisy_PSF[copies*k + j] = (np.random.poisson(lam=a_copy * factor)) / (factor)
        return copy_array, noisy_PSF


    # photon_factor = 50
    # PSF_img_copy, PSF_img_photon = photon_noise(PSF_img, photon_factor, N_copies)
    #
    # k = 10
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1 = plt.subplot(1, 3, 1)
    # img1 = ax1.imshow(PSF_img_copy[k, :,:,0])
    # ax1.get_xaxis().set_visible(False)
    # ax1.get_yaxis().set_visible(False)
    # ax1.set_title('Nominal PSF')
    #
    # ax2 = plt.subplot(1, 3, 2)
    # img2 = ax2.imshow(PSF_img_photon[k, :,:,0])
    # ax2.get_xaxis().set_visible(False)
    # ax2.get_yaxis().set_visible(False)
    # ax2.set_title(r'Photon Noise')
    #
    # res_photon = PSF_img_copy - PSF_img_photon
    # ax3 = plt.subplot(1, 3, 3)
    # img3 = ax3.imshow(res_photon[k, :,:,0])
    # ax3.get_xaxis().set_visible(False)
    # ax3.get_yaxis().set_visible(False)
    # ax3.set_title(r'Residual Photon Noise')
    # plt.show()

    """ (3) Autoencoder Model """
    K.clear_session()
    N_channels = 1          # 1 Nominal, 1 Defocused PSF
    input_shape = (pix, pix, 1,)

    k_size = 3
    model = models.Sequential()
    model.add(Conv2D(64, kernel_size=(k_size, k_size), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (k_size, k_size), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(8, (k_size, k_size), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # ### Encoded
    # model.add(Conv2D(8, (k_size, k_size), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (k_size, k_size), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (k_size, k_size), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (k_size, k_size), activation='sigmoid', padding='same'))
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy')
    # model.compile(optimizer='adam', loss='mean_squared_error')

    # N_total = N_copies * N_samples
    # N_train = N_total * 80 // 100
    # train_noisy, train_clean = PSF_img_photon[:N_train], PSF_img_copy[:N_train]
    # test_noisy, test_clean = PSF_img_photon[N_train:], PSF_img_copy[N_train:]

    N_copies = 15
    N_loops = 15
    epochs = 5
    N_batches = len(PSF_batches)
    ### Run the TRAINING

    low, high = 50, 250

    for i in range(N_loops):
        print("Loop: %d / %d" % (i+1, N_loops))
        for k in range(N_batches - 1):
            print("Batch: %d / %d" % (k + 1, N_batches))
            train_clean, train_noisy = photon_noise(PSF_batches[k], low=low, high=high, copies=N_copies)
            test_clean, test_noisy = photon_noise(PSF_batches[-1], low=low, high=high, copies=2)
            model.fit(train_noisy, train_clean, epochs=epochs, shuffle=True,
                      verbose=1, validation_data=(test_noisy, test_clean))

            clean_img = model.predict(test_noisy)
            residual = test_clean - clean_img
            RMS = np.std(residual, axis=(1, 2))
            mean_RMS = np.mean(RMS)
            print(mean_RMS)

    clean_img = model.predict(test_noisy)
    residual = test_clean - clean_img
    RMS = np.std(residual, axis=(1,2))
    mean_RMS = np.mean(RMS)
    print(mean_RMS)

    ### Define the ENCODER to access the CODE
    input_img = Input(shape=(input_shape))
    encoded_layer1 = model.layers[0]
    encoded_layer2 = model.layers[1]
    encoded_layer3 = model.layers[2]
    encoded_layer4 = model.layers[3]
    encoder = Model(input_img, encoded_layer4(encoded_layer3(encoded_layer2(encoded_layer1(input_img)))))
    encoder.summary()
    encoded_images = encoder.predict(train_noisy)

    filters_encoded = encoded_images.shape[-1]
    n_rows = int(np.sqrt(filters_encoded))
    n_cols = filters_encoded//n_rows
    j_img = 1*N_copies
    cmap = 'Reds'
    for j_img in range(3):
        plt.figure()
        plt.imshow(train_noisy[j_img*N_copies,:,:,0],cmap)

        fig, axes = plt.subplots(n_rows, n_cols)
        for i_row in range(n_rows):
            for j_col in range(n_cols):
                k = n_cols*i_row + j_col
                ax1 = plt.subplot(n_rows,n_cols, k+1)
                img1 = ax1.imshow(encoded_images[j_img*N_copies,:,:,k], cmap=cmap)
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                # plt.colorbar(img1, ax=ax1, orientation='horizontal')
    plt.show()

    cmap_PSF = 'hot'
    for k in range(5):
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        ax1 = plt.subplot(2, 3, 1)
        img1 = ax1.imshow(test_clean[N_copies*k, :, :, 0], cmap=cmap_PSF)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title('Clean Target')
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(2, 3, 2)
        img2 = ax2.imshow(test_noisy[N_copies*k, :, :, 0], cmap=cmap_PSF)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(r'Noisy Input')
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

        photon_error = test_clean[N_copies*k, :, :, 0] - test_noisy[N_copies*k, :, :, 0]
        rms_photon = np.std(photon_error)
        cmin = min(np.min(photon_error), -np.max(photon_error))
        ax3 = plt.subplot(2, 3, 3)
        img3 = ax3.imshow(photon_error, cmap='seismic')
        img3.set_clim(cmin, -cmin)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(r'Photon Error [Clean - Noisy] %.2e' % rms_photon)
        plt.colorbar(img3, ax=ax3, orientation='horizontal')


        ax5 = plt.subplot(2, 3, 5)
        img5 = ax5.imshow(clean_img[N_copies*k, :, :, 0], cmap=cmap_PSF)
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        ax5.set_title(r'Reconstructed Output')
        # img5.set_clim(cmin, -cmin)
        plt.colorbar(img5, ax=ax5, orientation='horizontal')

        res_error = test_clean - clean_img
        rms_res = np.std(res_error)
        ax6 = plt.subplot(2, 3, 6)
        img6 = ax6.imshow(res_error[N_copies*k, :, :, 0], cmap='seismic')
        ax6.get_xaxis().set_visible(False)
        ax6.get_yaxis().set_visible(False)
        ax6.set_title(r'Reconstructed Error [Clean - Recon] %.2e' % rms_res)
        img6.set_clim(cmin, -cmin)
        plt.colorbar(img6, ax=ax6, orientation='horizontal')

    plt.show()



    ### Things to check

    # Which Loss Function to use? Binary Crossentropy or MSE?
    # Impact of Depth (N layers). Worse for 8x8 code
    # Impact of N filters
    # Impact of the number of Copies
    # Plot validation over time, not the Binary Crossentropy

    # Visualize the ENCODED
    # For the same PSF but different instances of Noise, see if the Encoded is the same!!
    # What happens if you flip them? Use the Clean as Input,




