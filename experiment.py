"""
==========================================================
                    Unsupervised - Experiment
==========================================================

#FIXME Complete the description
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

N_zern = 5
Z = 1.
N_features = 25*25

class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    ### Parameters
    rho_aper = 0.25         # Size of the aperture relative to 1.0
    N_pix = 128
    pix = 25                # Number of pixels for the Zoom of the PSF
    minPix, maxPix = (N_pix + 1 - pix) // 2, (N_pix + 1 + pix) // 2

    def __init__(self, N_zern):

        ### Zernike Wavefront
        x = np.linspace(-1, 1, self.N_pix, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        self.pupil = rho <= self.rho_aper
        rho, theta = rho[self.pupil], theta[self.pupil]
        zernike = zern.ZernikeNaive(mask=self.pupil)
        _phase = zernike(coef=np.zeros(N_zern + 3), rho=rho/self.rho_aper, theta=theta, normalize_noll=False,
                         mode='Jacobi', print_option='Silent')
        H_flat = zernike.model_matrix[:, 3:]  # remove the piston and tilts
        self.H_matrix = zern.invert_model_matrix(H_flat, self.pupil)

        # Update the number of aberrations to match the dimensions of H
        self.H_matrix = self.H_matrix[:,:,:N_zern]
        self.N_zern = self.H_matrix.shape[-1]

        self.PEAK = self.peak_PSF()

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """

        im, strehl = self.compute_PSF(np.zeros(self.N_zern))

        return strehl

    def compute_PSF(self, zern_coef):
        """
        Compute the PSF and the Strehl ratio
        """

        # if zern_coef.shape != self.N_zern:      # Zero-pad to match dimensions
        #     extra_zeros = np.zeros((self.N_zern - zern_coef.shape[0]))
        #     zern_coef = np.concatenate([zern_coef, extra_zeros])

        phase = np.dot(self.H_matrix, zern_coef)
        pupil_function = self.pupil * np.exp(1j * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2

        try:
            image /= self.PEAK

        except AttributeError:
            # If self.PEAK is not defined, self.compute_PSF will compute the peak
            pass

        strehl = np.max(image)

        image = image[self.minPix:self.maxPix, self.minPix:self.maxPix]
        return image, strehl

    def plot_PSF(self, zern_coef, i):
        """
        Plot an image of the PSF
        :param zern_coef:
        :param i: iteration (for labelling purposes)
        """

        PSF, strehl = self.compute_PSF(zern_coef)

        plt.figure()
        plt.imshow(PSF)
        plt.title('Iter: %d Strehl: %.3f' %(i, strehl))
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

def generate_training_set(PSF_model, N_samples=1000):

    _im, s = PSF_model.compute_PSF(np.zeros(N_zern))
    N_features = _im.flatten().shape[0]

    training = np.zeros((N_samples, N_features))
    coefs = np.zeros((N_samples, N_zern))
    for i in range(N_samples):
        rand_coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
        coefs[i] = rand_coef
        ims, _s = PSF_model.compute_PSF(rand_coef)
        training[i] = ims.flatten()

    return training, coefs


if __name__ == "__main__":

    coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
    print(coef)

    PSF = PointSpreadFunction(N_zern)
    PSF.plot_PSF(coef, i=0)
    plt.show()

    from keras.backend.tensorflow_backend import tf

    model = Sequential()
    model.add(Dense(N_features//125, input_shape=(N_features,), activation='relu'))
    # model.add(Dense(N_features//5, input_shape=(N_features,), activation='relu'))
    # model.add(Dense(N_features//25, input_shape=(N_features,), activation='relu'))
    # model.add(Dense(N_features//125, input_shape=(N_features,), activation='relu'))
    model.summary()

    H = PSF.H_matrix.copy()
    pup = PSF.pupil.copy()
    peak = PSF.PEAK.copy()
    def strehl_loss(y_pred, y_true):
        print(y_pred.shape)

        phase = np.dot(H, y_pred)
        print(phase.shape)
        cos_x = pup * K.cos(phase)
        sin_x = pup * K.sin(phase)

        F_cos = tf.fft2d(cos_x)
        F_sin = tf.fft2d(sin_x)

        # pupil_function = pup * K.exp(1j * phase)
        # image = (K.abs(tf.fft2d(pupil_function)))**2
        # strehl = K.max(image)/peak

        return 1.

    model.compile(optimizer='adam', loss=strehl_loss)
    #
    # train_images, train_coefs = generate_training_set(PSF)
    #
    # model.fit(x=train_images, epochs=100, shuffle=True, verbose=2)



