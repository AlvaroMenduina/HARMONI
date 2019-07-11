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

N_zern = 3
Z = 1.25
pix = 12
N_features = pix**2

class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    ### Parameters
    rho_aper = 0.40         # Size of the aperture relative to 1.0
    N_pix = 128
                   # Number of pixels for the Zoom of the PSF
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

def generate_training_set(PSF_model, N_samples=1500):

    _im, s = PSF_model.compute_PSF(np.zeros(N_zern))
    extra_coefs = generate_sampling(2, N_zern, 0.2, -0.1)
    N_features = _im.flatten().shape[0] * (extra_coefs.shape[0] + 1)

    training = np.zeros((N_samples, N_features))
    coefs = np.zeros((N_samples, N_zern))


    for i in range(N_samples):
        rand_coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
        coefs[i] = rand_coef
        ims, _s = PSF_model.compute_PSF(rand_coef)
        nom_im = [ims.flatten()]
        # print(ims.flatten().shape)
        for c in extra_coefs:
            ims, _s = PSF_model.compute_PSF(rand_coef + c)
            nom_im.append(ims.flatten())
            # print(len(nom_im))
        training[i] = np.concatenate(nom_im)

    return training, coefs

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


if __name__ == "__main__":
    extra_coefs = generate_sampling(2, N_zern, 0.2, -0.1)
    coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
    print(coef)

    PSF = PointSpreadFunction(N_zern)
    PSF.plot_PSF(coef, i=0)
    plt.show()

    train_images, train_coefs = generate_training_set(PSF)

    from keras.backend.tensorflow_backend import tf

    model = Sequential()
    model.add(Dense(1000, input_shape=(train_images.shape[1],), activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(N_zern, activation='relu'))
    model.summary()

    H = PSF.H_matrix.copy().T
    H = np.rollaxis(H, 1, 0)
    pup = PSF.pupil.copy()
    peak = PSF.PEAK.copy()


    pupt = tf.constant(pup, dtype=tf.float32)
    ht = tf.constant(H, dtype=tf.float32)

    def loss(y_pred, y_true):
        phase = K.dot(y_pred, ht)
        print(y_pred)
        cos_x = pupt * K.cos(phase)
        sin_x = pupt * K.sin(phase)
        complex_phase = tf.complex(cos_x, sin_x)
        image = (K.abs(tf.fft2d(complex_phase)))**2
        print(image.shape)
        strehl = K.max(image, axis=(1, 2)) / peak
        print(strehl.shape)

        return 0*y_true - K.mean(strehl)

    model.compile(optimizer='adam', loss=loss)

    model.fit(x=train_images, y=train_coefs, epochs=5000, shuffle=True, verbose=1)

    guess = model.predict(train_images[:10])



