"""
==========================================================
                    Unsupervised Calibration
==========================================================

Experiments: see if we can infer the underlying aberrations in an UNSUPERVISED way

We have a nominal PSF with aberrations Phi_0, that we don't know
We take several PSF images with different Deformable Mirror corrections
And we use a CNN to find the correction that minimizes the difference between
the PSF(Phi_0 + Correction) and a perfect PSF

"""

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import zern_core as zern

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.models import Sequential
from keras import backend as K
from numpy.linalg import norm as norm

# PARAMETERS
N_zern = 3                  # Number of aberrations to consider
Z = 1.25                    # Strength of the aberrations
pix = 25                    # Pixels to crop the PSF

class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    ### Parameters
    rho_aper = 0.25         # Size of the aperture relative to 1.0
    N_pix = 128             # Number of pixels for the FFT computations
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

        # Save only the part of the model matrix that we need
        self.H_matrix = self.H_matrix[:,:,:N_zern]
        # FIXME: Watch out! The way we form H with the Zernike pyramid means that we can end up using the aberrations we
        # don't want. FIX this is in the future
        self.N_zern = self.H_matrix.shape[-1]

        self.PEAK = self.peak_PSF()

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """

        im, strehl = self.compute_PSF(np.zeros(self.N_zern))

        return strehl

    def compute_PSF(self, zern_coef, crop=True):
        """
        Compute the PSF and the Strehl ratio
        """
        phase = np.dot(self.H_matrix, zern_coef)
        pupil_function = self.pupil * np.exp(1j * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2

        try:
            image /= self.PEAK

        except AttributeError:
            # If self.PEAK is not defined, self.compute_PSF will compute the peak
            pass

        strehl = np.max(image)

        if crop:
            image = image[self.minPix:self.maxPix, self.minPix:self.maxPix]
        else:
            pass
        return image, strehl

    def plot_PSF(self, zern_coef):
        """
        Plot an image of the PSF
        """
        PSF, strehl = self.compute_PSF(zern_coef)

        plt.figure()
        plt.imshow(PSF)
        plt.title('Strehl: %.3f' %strehl)
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

def generate_training_set(PSF_model, N_samples=1500):

    _im, s = PSF_model.compute_PSF(np.zeros(N_zern))
    N_pix = _im.shape[0]
    extra_coefs = generate_sampling(2, N_zern, 0.2, -0.10)
    N_channels = (extra_coefs.shape[0] + 1)

    training = np.zeros((N_samples, N_pix, N_pix, N_channels))
    perfect = np.zeros((N_samples, 128, 128))
    coefs = np.zeros((N_samples, N_zern))

    im_perfect, _s = PSF_model.compute_PSF(np.zeros(N_zern), crop=False)
    for i in range(N_samples):
        rand_coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
        coefs[i] = rand_coef
        im0, _s = PSF_model.compute_PSF(rand_coef)
        # nom_im = [ims.flatten()]
        nom_im = [im0]
        # print(ims.flatten().shape)
        for c in extra_coefs:
            ims, _s = PSF_model.compute_PSF(rand_coef + c)
            # nom_im.append(ims.flatten())
            nom_im.append(ims - im0)
            # print(len(nom_im))
            # print(nom_im[1].shape)
        training[i] = np.moveaxis(np.array(nom_im), 0, -1)
        perfect[i] = fftshift(im_perfect)


    return training, coefs, perfect

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
    extra_coefs = generate_sampling(2, N_zern, 0.3, -0.15)
    coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
    print(coef)

    PSF = PointSpreadFunction(N_zern)
    PSF.plot_PSF(coef, i=0)
    plt.show()

    train_images, train_coefs, perfect = generate_training_set(PSF, N_samples=500)
    input_shape = (pix, pix, extra_coefs.shape[0]+1, )

    from keras.backend.tensorflow_backend import tf
    num_classes = N_zern

    # MLP
    # model = Sequential()
    # model.add(Dense(1000, input_shape=input_shape, activation='relu'))
    # model.add(Dense(500, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(N_zern, activation='relu'))
    # model.summary()

    # CNN
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes))
    # model.add(Activation('linear'))
    model.summary()

    H = PSF.H_matrix.copy().T
    H = np.rollaxis(H, 1, 0)
    pup = PSF.pupil.copy()
    peak = PSF.PEAK.copy()


    pupt = tf.constant(pup, dtype=tf.float32)
    ht = tf.constant(H, dtype=tf.float32)
    coef_t = tf.constant(train_coefs[:-10], dtype=tf.float32)
    perfect_t = tf.constant(perfect[:-10], dtype=tf.float32)

    def loss(y_true, y_pred):
        phase = K.dot(coef_t + y_pred, ht)
        print(phase.shape)

        cos_x = pupt * K.cos(phase)
        sin_x = pupt * K.sin(phase)
        complex_phase = tf.complex(cos_x, sin_x)
        image = (K.abs(tf.fft2d(complex_phase)))**2 / peak
        print(image.shape)

        res = K.mean(K.sum((image - perfect_t)**2))


        # strehl = K.max(image, axis=(1, 2)) / peak
        # print(strehl.shape)
        # res = -K.mean(strehl)

        return res

    model.compile(optimizer='adam', loss=loss)

    model.fit(x=train_images[:-10], y=train_coefs[:-10], epochs=50, batch_size=490, shuffle=False, verbose=1)
    guess = model.predict(train_images[-10:])
    # print(guess)
    print(train_coefs[-10:])

    print(train_coefs[-10:] + guess)

    PSF.plot_PSF(train_coefs[-1], i=0)
    PSF.plot_PSF(train_coefs[-1] + guess[-1], i=0)
    plt.show()
    #
    # # Super Interesting stuff
    # # How to get the gradients in Keras
    # def get_weight_grad(model, inputs, outputs):
    #     """ Gets gradient of model for given inputs and outputs for all weights"""
    #     grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    #     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    #     f = K.function(symb_inputs, grads)
    #     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    #     output_grad = f(x + y + sample_weight)
    #     return output_grad
    #
    #
    # def get_layer_output_grad(model, inputs, outputs, layer=-1):
    #     """ Gets gradient a layer output for given inputs and outputs"""
    #     grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    #     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    #     f = K.function(symb_inputs, grads)
    #     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    #     output_grad = f(x + y + sample_weight)
    #     return output_grad
    #
    #
    # weight_grads = get_weight_grad(model, train_images[:10], train_coefs[:10])
    # output_grad = get_layer_output_grad(model, train_images[:10], train_coefs[:10])



