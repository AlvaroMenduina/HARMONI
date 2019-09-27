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

import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import zern_core as zern

import keras
from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from numpy.linalg import norm as norm

# PARAMETERS
Z = 1.25                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 30                    # Pixels to crop the PSF
N_PIX = 256                 # Pixels for the Fourier arrays
RHO_APER = 0.5              # Size of the aperture relative to the physical size of the Fourier arrays
RHO_OBSC = 0.15             # Central obscuration



def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC):
    """
    Computes the (Xc, Yc) coordinates of actuator centres
    inside a circle of rho_aper, assuming there are N_actuators
    along the [-1, 1] line

    :param N_actuators:
    :param rho_aper:
    :return:
    """

    x0 = np.linspace(-1., 1., N_actuators, endpoint=True)
    delta = x0[1] - x0[0]
    xx, yy = np.meshgrid(x0, x0)
    x_f = xx.flatten()
    y_f = yy.flatten()

    act = []
    for x_c, y_c in zip(x_f, y_f):
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        if r < 0.95 * rho_aper and r > 1.1 * rho_obsc:
            act.append([x_c, y_c])
    total_act = len(act)
    print('Total Actuators: ', total_act)
    return act, delta

def plot_actuators(centers):
    N_act = len(centers[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circ1 = Circle((0,0), RHO_APER, linestyle='--', fill=None)
    circ2 = Circle((0,0), RHO_OBSC, linestyle='--', fill=None)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    for c in centers[0]:
        ax.scatter(c[0], c[1], color='red')
    ax.set_aspect('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('%d actuators' %N_act)

def rbf_matrix(centres, rho_aper=RHO_APER, rho_obsc=RHO_OBSC):

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
        matrix[:, :, k] = pupil * np.exp(-r2 / (1 * delta) ** 2)

    mat_flat = matrix[pupil]

    return matrix, pupil, mat_flat

class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    N_pix = N_PIX             # Number of pixels for the FFT computations
    minPix, maxPix = (N_pix + 1 - pix) // 2, (N_pix + 1 + pix) // 2

    def __init__(self, matrices):

        self.N_act = matrices[0].shape[-1]
        self.RBF_mat = matrices[0].copy()
        self.pupil_mask = matrices[1].copy()
        self.RBF_flat = matrices[2].copy()
        self.defocus = np.zeros_like(matrices[1])

        self.PEAK = self.peak_PSF()

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """
        im, strehl = self.compute_PSF(np.zeros(self.N_act))
        return strehl

    def compute_PSF(self, coef, crop=True):
        """
        Compute the PSF and the Strehl ratio
        """

        phase = np.dot(self.RBF_mat, coef) + self.defocus

        pupil_function = self.pupil_mask * np.exp(1j * phase)
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

    def plot_PSF(self, coef):
        """
        Plot an image of the PSF
        """
        PSF, strehl = self.compute_PSF(coef)

        plt.figure()
        plt.imshow(PSF)
        plt.title('Strehl: %.3f' %strehl)
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

# ==================================================================================================================== #

def generate_training_set(PSF_model, N_samples=1500, dm_stroke=0.10, sampling="simple", N_cases=2):

    N_act = PSF.N_act

    # Generate extra coefficients for the Deformable Mirror corrections
    if sampling == "simple":        # Scales as 2 * N_zern
        extra_coefs = simple_sampling(N_act, dm_stroke)

    elif sampling == "complete":        # Scales as 2 ^ N_zern
        extra_coefs = generate_sampling(2, N_act, 2*dm_stroke, -dm_stroke)

    elif sampling == "random":          # Random displacements for 2*N_cases
        stroke = 0.5*Z
        extra_coefs = np.random.uniform(low=-stroke, high=stroke, size=(N_cases, N_act))

    else:
        raise Exception

    N_channels = 1 + extra_coefs.shape[0]

    # Perfect PSF (128x128) - For the Loss function
    im_perfect, _s = PSF_model.compute_PSF(np.zeros(N_act))
    perfect = np.zeros((N_samples, pix, pix))           # Store the PSF N_sample times

    # Training set contains (25x25)-images of: Nominal PSF + PSFs with corrections
    training = np.zeros((N_samples, pix, pix, N_channels))
    # Store the Phi_0 coefficients for later
    coefs = np.zeros((N_samples, N_act))

    for i in range(N_samples):

        if i%100 == 0:
            print(i)

        rand_coef = np.random.uniform(low=-Z, high=Z, size=N_act)
        coefs[i] = rand_coef
        im0, _s = PSF_model.compute_PSF(rand_coef)
        # Store the images in a least and then turn it into array
        nom_im = [im0]

        for c in extra_coefs:

            ims, _s = PSF_model.compute_PSF(rand_coef + c)
            # Difference between NOMINAL and CORRECTED
            # nom_im.append(ims - im0)
            nom_im.append(ims)

        training[i] = np.moveaxis(np.array(nom_im), 0, -1)
        # NOTE: Tensorflow does not have FFTSHIFT operation. So we have to fftshift the Perfect PSF
        # back to the weird un-shifted format.
        perfect[i] = fftshift(im_perfect)

    return training, coefs, perfect, extra_coefs

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

def simple_sampling(N_zern, dm_stroke):
    """
    Extra coefficients in the form:
    [-x, 0, ..., 0]
    [+x, 0, ..., 0]
    [0, -x, 0, ..., 0]
    [0, +x, 0, ..., 0]
            ...

    The previous sampling scheme scales as sampling ^ N_zern. In contrast,
    this scales as 2 * N_zern
    """
    coefs = np.empty((2* N_zern, N_zern))
    for i in range(N_zern):
        dummy = np.zeros((2, N_zern))
        dummy[0, i] = dm_stroke
        dummy[1, i] = -dm_stroke

        coefs[2*i:2*i+2] = dummy
    return coefs

if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    N_actuators = 25
    centers = actuator_centres(N_actuators)
    # centers = radial_grid(N_radial=5)
    N_act = len(centers[0])
    plot_actuators(centers)

    rbf_mat = rbf_matrix(centers)        # Actuator matrix

    c_act = np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(rbf_mat[0], c_act)
    p0 = min(phase0.min(), -phase0.max())

    plt.figure()
    plt.imshow(phase0, extent=(-1,1,-1,1), cmap='bwr')
    plt.colorbar()
    plt.clim(p0, -p0)
    for c in centers[0]:
        plt.scatter(c[0], c[1], color='black', s=4)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

    PSF = PointSpreadFunction(rbf_mat)

    N_cases = 3
    N_classes = N_act
    N_train, N_test = 500, 50
    N_samples = N_train + N_test
    sampling = "random"

    _images, _coefs, _perfect, _extra = generate_training_set(PSF, N_samples, sampling=sampling, N_cases=N_cases)
    train_images, train_coefs, perfect_psf = _images[:N_train], _coefs[:N_train], _perfect[:N_train]
    test_images, test_coefs = _images[N_train:], _coefs[N_train:]
    dummy = np.zeros_like(train_coefs)

    k = 0
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(train_images[k, :, :, 0], cmap='hot')
    # ax1.set_title(r'$PSF(\Phi_0)$')
    # plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(train_images[k, :, :, 1], cmap='hot')
    # ax2.set_title(r'$PSF(\Phi_0 + \Delta_1) - PSF(\Phi_0)$')
    # plt.colorbar(im2, ax=ax2)

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(train_images[k, :, :, 2], cmap='hot')
    # ax3.set_title(r'$PSF(\Phi_0 - \Delta_1) - PSF(\Phi_0)$')
    # plt.colorbar(im3, ax=ax3)

    N_channels = train_images.shape[-1]
    input_shape = (pix, pix, N_channels,)

    plt.show()

    model = models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(N_classes))
    model.summary()

    # Some bits for the Loss function definition
    H = PSF.RBF_mat.copy().T
    # Roll it to match the Theano convention for dot product that TF uses
    H = np.rollaxis(H, 1, 0)  # Model Matrix to compute the Phase with Zernikes
    pup = PSF.pupil_mask.copy()  # Pupil Mask
    peak = PSF.PEAK.copy()  # Peak to normalize the FFT calculations

    # Transform them to TensorFlow
    pupt = tf.constant(pup, dtype=tf.float32)
    ht = tf.constant(H, dtype=tf.float32)
    coef_t = tf.constant(train_coefs, dtype=tf.float32)
    perfect_t = tf.constant(perfect_psf, dtype=tf.float32)


    def loss(y_true, y_pred):
        """
        Custom Keras Loss function
        :param y_true: unused because we want it to be unsupervised
        :param y_pred: predicted corrections for the PSF
        :return:

        Notes: Keras doesn't like dealing with Complex numbers so we separate the pupil function
        P = P_mask * exp( 1i * Phase)
        into Real and Imaginary parts using Euler's formula and then join them back into a Complex64
        because Tensorflow expects it that way for the Fourier Transform
        """

        # Phase includes the unknown Phi_0 (coef_t) and the Predictions
        phase = K.dot(coef_t + y_pred, ht)

        cos_x, sin_x = pupt * K.cos(phase), pupt * K.sin(phase)
        complex_phase = tf.complex(cos_x, sin_x)
        image = (K.abs(tf.fft2d(complex_phase))) ** 2 / peak
        print(image.shape)

        Q1 = image[:, :pix//2, :pix//2] - perfect_t[:, :pix//2, :pix//2]
        Q2 = image[:, N_PIX-pix//2:, :pix//2] - perfect_t[:, pix//2:, :pix//2]
        Q3 = image[:, :pix//2, N_PIX-pix//2:] - perfect_t[:, :pix//2, pix//2:]
        Q4 = image[:, N_PIX-pix//2:, N_PIX-pix//2:] - perfect_t[:, pix//2:, pix//2:]
        print(Q1.shape)

        # Compute the Difference between the PSF after applying a correction and a Perfect PSF
        res = K.mean(K.sum(Q1**2 + Q2**2 + Q3**2 + Q4**2))

        # We can train it to maximize the Strehl ratio on
        # strehl = K.max(image, axis=(1, 2)) / peak
        # print(strehl.shape)
        # res = -K.mean(strehl)

        return res


    model.compile(optimizer='adam', loss=loss)
    train_history = model.fit(x=train_images, y=dummy, epochs=50, batch_size=N_train, shuffle=False, verbose=1)
    # NOTE: we force the batch_size to be the whole Training set because otherwise we would need to match
    # the chosen coefficients from the batch to those of the coef_t tensor. Can't be bothered...

    loss_hist = train_history.history['loss']

    guess_coef = model.predict(test_images)
    residual = test_coefs + guess_coef








    loss_array, strehl_array = [], []

    for N_cases in [1, 2, 3]:

        N_cases = 3
        N_classes = N_zern
        N_train, N_test = 5000, 200
        N_samples = N_train + N_test
        sampling = "random"

        # Generate the Training and Test sets
        _images, _coefs, _perfect, _extra = generate_training_set(PSF, N_samples=N_samples, sampling=sampling, N_cases=N_cases)
        train_images, train_coefs, perfect_psf = _images[:N_train], _coefs[:N_train], _perfect[:N_train]
        test_images, test_coefs = _images[N_train:], _coefs[N_train:]
        dummy = np.zeros_like(train_coefs)

        ### Convolutional Neural Networks
        N_channels = train_images.shape[-1]
        input_shape = (pix, pix, N_channels,)

        # k = 0
        # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1 = plt.subplot(1, 3, 1)
        # im1 = ax1.imshow(train_images[k, :, :, 0], cmap='hot')
        # ax1.set_title(r'$PSF(\Phi_0)$')
        # plt.colorbar(im1, ax=ax1)
        #
        # ax2 = plt.subplot(1, 3, 2)
        # im2 = ax2.imshow(train_images[k, :, :, 1], cmap='bwr')
        # ax2.set_title(r'$PSF(\Phi_0 + \Delta_1) - PSF(\Phi_0)$')
        # plt.colorbar(im2, ax=ax2)
        #
        # ax3 = plt.subplot(1, 3, 3)
        # im3 = ax3.imshow(train_images[k, :, :, 2], cmap='bwr')
        # ax3.set_title(r'$PSF(\Phi_0 - \Delta_1) - PSF(\Phi_0)$')
        # plt.colorbar(im3, ax=ax3)
        #
        # plt.show()

        # CNN Model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.5))
        model.add(Flatten())
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        model.add(Dense(N_classes))
        # model.add(Activation('linear'))
        model.summary()


        # Some bits for the Loss function definition
        H = PSF.H_matrix.copy().T
        # Roll it to match the Theano convention for dot product that TF uses
        H = np.rollaxis(H, 1, 0)            # Model Matrix to compute the Phase with Zernikes
        pup = PSF.pupil.copy()              # Pupil Mask
        peak = PSF.PEAK.copy()              # Peak to normalize the FFT calculations

        # Transform them to TensorFlow
        pupt = tf.constant(pup, dtype=tf.float32)
        ht = tf.constant(H, dtype=tf.float32)
        coef_t = tf.constant(train_coefs, dtype=tf.float32)
        perfect_t = tf.constant(perfect_psf, dtype=tf.float32)


        def loss(y_true, y_pred):
            """
            Custom Keras Loss function
            :param y_true: unused because we want it to be unsupervised
            :param y_pred: predicted corrections for the PSF
            :return:

            Notes: Keras doesn't like dealing with Complex numbers so we separate the pupil function

            P = P_mask * exp( 1i * Phase)

            into Real and Imaginary parts using Euler's formula and then join them back into a Complex64
            because Tensorflow expects it that way for the Fourier Transform
            """

            # Phase includes the unknown Phi_0 (coef_t) and the Predictions
            phase = K.dot(coef_t + y_pred, ht)

            cos_x, sin_x = pupt * K.cos(phase), pupt * K.sin(phase)
            complex_phase = tf.complex(cos_x, sin_x)
            image = (K.abs(tf.fft2d(complex_phase)))**2 / peak

            # Compute the Difference between the PSF after applying a correction and a Perfect PSF
            res = K.mean(K.sum((image - perfect_t)**2))

            # We can train it to maximize the Strehl ratio on
            # strehl = K.max(image, axis=(1, 2)) / peak
            # print(strehl.shape)
            # res = -K.mean(strehl)

            return res

        def compute_strehl(PSF_images, true_coef):
            """
            For a given set of PSF images and their associated aberration coefficients,
            it computes the predicted correction, and the Strehl ratio after that correction is applied
            :param PSF_images:
            :param true_coef:
            :return:
            """
            guess_coef = model.predict(PSF_images)
            strehls = []
            for g, c in zip(guess_coef, true_coef):
                # print(g)
                # print(c)
                _im, s = PSF.compute_PSF(g + c)
                strehls.append(s)
            return np.array(strehls)

        def post_analysis(test_images, test_coefs, N_show=5, path='Z50'):
            """

            :param test_images:
            :param test_coefs:
            :param N_show:
            :return:
            """
            # initial_strehls = compute_strehl(test_images, np.zeros_like(test_coefs))
            initial_strehls = np.max(test_images[:,:,:,0], axis=(1,2))
            final_strehls = compute_strehl(test_images, test_coefs)

            plt.figure()    # Show a comparison of Strehl ratios
            plt.hist(initial_strehls, histtype='step', label='Before')
            plt.hist(final_strehls, histtype='step', label='After')
            plt.xlabel('Strehl Ratio')
            plt.legend(title='Stage')
            plt.savefig(os.path.join('Experiment', path, 'Strehl_Hist'))

            for k in range(N_show):     # Show the PSF comparison (Before / After)

                guess_coef = model.predict(test_images)
                final_PSF, _s = PSF.compute_PSF(guess_coef[k] + test_coefs[k])

                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1 = plt.subplot(1, 2, 1)
                im1 = ax1.imshow(test_images[k, :, :, 0], cmap='hot')
                ax1.set_title('Before %.3f' %initial_strehls[k])
                plt.colorbar(im1, ax=ax1)

                ax2 = plt.subplot(1, 2, 2)
                im2 = ax2.imshow(final_PSF, cmap='hot')
                ax2.set_title('After %.3f' %final_strehls[k])
                plt.colorbar(im2, ax=ax2)

                plt.savefig(os.path.join('Experiment', path, '%d' %k))



        model.compile(optimizer='adam', loss=loss)
        train_history = model.fit(x=train_images, y=dummy, epochs=1500, batch_size=N_train, shuffle=False, verbose=1)
        # NOTE: we force the batch_size to be the whole Training set because otherwise we would need to match
        # the chosen coefficients from the batch to those of the coef_t tensor. Can't be bothered...

        loss_hist = train_history.history['loss']

        loss_array.append(loss_hist)
        strehl_array.append(compute_strehl(test_images, test_coefs))

    LL = [loss_array[:-2], loss_array[-2], loss_array[-1]]
    plt.figure()
    for i, l in enumerate(LL):
        plt.semilogy(l, label=i+1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Number of cases')
    plt.show()

    plt.figure()
    for i, s in enumerate(strehl_array):
        plt.hist(s, label=i+1, histtype='step')
    plt.legend()
    plt.show()

    strehls = np.array(strehl_array)
    s_mean = np.mean(strehls, axis=1)

    # ===============================================================================================================  #

    plt.figure()
    plt.semilogy(l_PSF, label='PSF')
    plt.semilogy(l_Diff, label='Diff')
    plt.legend(title='Approach')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0, 250])
    plt.show()



    # Check predictions
    guess = model.predict(test_images)
    print(guess[-1, :5])
    print("\nTrue Values:")
    print(test_coefs[-1, :5])

    # Plot a comparison Before & After
    PSF.plot_PSF(test_coefs[-1])
    PSF.plot_PSF(test_coefs[-1] + guess[-1])
    plt.show()

    # Some convergence results
    loss_hist = train_history.history['loss']
    plt.figure()
    plt.semilogy(loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    N_cases = [1, 2, 3]

    losses.append(loss_hist)
    strehls.append(compute_strehl())

    loss_array = [losses[:-2], losses[-1]]
    strehl_array = [strehls[:-2], strehls[-1]]

    """ Visualizing the Layers """


    layer_outputs = [layer.output for layer in model.layers]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    img_tensor = test_images[:1]
    M = img_tensor.shape[-1]
    for k in range(M):
        plt.figure()
        plt.imshow(img_tensor[0,:,:,k], cmap='hot')
        plt.colorbar()
        plt.title('Channel %d' %k)

    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)


    layer_names = []
    for layer in model.layers[:3]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        print(size)
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :,col * images_per_row + row]
                # channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                # channel_image /= channel_image.std()
                # channel_image *= 64
                # channel_image += 128
                # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='hot')

    ## _________________________________________________________________________________________________ ##

    # Trying to see what the CNN does.
    # Let's

    # extra_coefs = simple_sampling(N_zern, dm_stroke=0.1)
    # extra_coefs = random_sampling(N_zern, dm_stroke=0.1, N_cases=3)
    extra_coefs = _extra
    coef0 = test_coefs[1]
    im0, _s = PSF.compute_PSF(coef0)
    nom_im = [im0]
    for c in extra_coefs:
        ims, _s = PSF.compute_PSF(coef0 + c)
        nom_im.append(ims - im0)
    im0 = np.moveaxis(np.array(nom_im), 0, -1)
    im0 = im0[np.newaxis, :, :, :]

    coef1 = coef0.copy()
    coef1[0] += 0.25         # Poke one of the Zernikes
    im1, _s = PSF.compute_PSF(coef1)
    nom_im = [im1]
    for c in extra_coefs:
        ims, _s = PSF.compute_PSF(coef1 + c)
        nom_im.append(ims - im1)
    im1 = np.moveaxis(np.array(nom_im), 0, -1)
    im1 = im1[np.newaxis, :, :, :]

    j_channel = 0
    i1, i2 = im0[0, :, :, j_channel], im1[0, :, :, j_channel]
    i3 = i2 - i1
    mins = min(i3.min(), -i3.max())
    maxs = max(i1.max(), i2.max())

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = plt.subplot(1, 3, 1)
    img1 = ax1.imshow(i1, cmap='bwr')
    ax1.set_title('Nominal Image')
    # img1.set_clim(0, maxs)
    img1.set_clim(-maxs, maxs)
    plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 3, 2)
    img2 = ax2.imshow(i2, cmap='bwr')
    ax2.set_title('Poked Image')
    # img2.set_clim(0, maxs)
    img2.set_clim(-maxs, maxs)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')

    ax3 = plt.subplot(1, 3, 3)
    img3 = ax3.imshow(i3, cmap='bwr')
    ax3.set_title('Residual')
    img3.set_clim(mins, -mins)
    c3 = plt.colorbar(img3, ax=ax3, orientation='horizontal')

    plt.savefig(os.path.join('Experiment','PokedComparison_channel%d') %j_channel)


    layer_outputs = [layer.output for layer in model.layers]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations0 = activation_model.predict(im0)
    activations1 = activation_model.predict(im1)
    diff_activ = [a - b for (a,b) in zip(activations0, activations1)]

    k = 1
    ran = activations0[-k].shape[-1]
    plt.figure()
    # plt.scatter(range(ran), coef0)
    plt.scatter(range(ran), activations0[-k][0], label='Nominal')
    # plt.scatter(range(ran), coef0 + activations0[-k][0], color='black')
    plt.scatter(range(ran), activations1[-k][0], label=r'Poked $Z[0]$')
    # plt.scatter(0.2, -coef0[0], color='black')
    # plt.scatter(0.2, -coef1[0], color='red')
    plt.xlabel('Zernike coefficient')
    plt.ylabel(r'Value [$\lambda$]')
    plt.legend(title='Network predictions')
    plt.savefig(os.path.join('Experiment', 'Poked_predicitons'))
    plt.show()

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations0):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :,col * images_per_row + row]
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(np.log10(display_grid), aspect='auto', cmap='binary')
        plt.savefig(os.path.join('Experiment', layer_name + '_nom_LOG'))

    for layer_name, layer_activation in zip(layer_names, activations1):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :,col * images_per_row + row]
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='binary')
        plt.savefig(os.path.join('Experiment', layer_name + '_poke'))


    for layer_name, layer_activation in zip(layer_names, diff_activ):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :,col * images_per_row + row]
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(np.log10(np.abs(display_grid)), cmap='binary')
        # m = min(display_grid.min(), -display_grid.max())
        # plt.clim(m, -m)
        plt.savefig(os.path.join('Experiment', layer_name + '_diff_LOG'))

        # plt.colorbar(orientation='horizontal')
    plt.show()


    ## _________________________________________________________________________________________________ ##

    # Checking how the FILTERS look like

    conv = model.layers[0]
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        v = sess.run(conv.weights[0])
        print(v[:,:,0,0])
        i = 1
        for k in range(5):
            plt.figure()
            plt.imshow(v[:,:,k,i])
    plt.show()

        ## Further analysis

    """
    - Range of Z intensities over which we can operate. Maybe too small is difficult. Tiny features
    - Instead of 2^N_zern use 2*N_zern. 1 correction per aberration (+-)
    - Impact of strength of DM stroke. Too small, probably impossible to calibrate
    - Impact of underlying aberrations we do not know. Noise
    
    - Instead of using 2*N_zern + 1, why not N images with random commands?
    
    - Whether to use Dropout or not. Dropout makes it worse
    - CNN output visualization to understand it
    - Influence of Number of channels
    
    - Recurrent NN??
    """


    # MLP
    # model = Sequential()
    # model.add(Dense(1000, input_shape=input_shape, activation='relu'))
    # model.add(Dense(500, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(N_zern, activation='relu'))
    # model.summary()

    # # Validation tensors
    # val_coef = tf.constant(test_coefs, dtype=tf.float32)
    # val_perfect = tf.constant(perfect_psf[:N_test], dtype=tf.float32)
    #
    # class CustomCallback(keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         self.losses = []
    #
    #
    #     def on_train_end(self, logs={}):
    #         return
    #
    #     def on_epoch_begin(self, epoch, logs={}):
    #         return
    #
    #     def on_epoch_end(self, epoch, logs={}):
    #         self.losses.append(logs.get('loss'))
    #         y_pred = self.model.predict(self.validation_data[0])
    #         print(y_pred)
    #         return
    #
    #     def on_batch_begin(self, batch, logs={}):
    #         return
    #
    #     def on_batch_end(self, batch, logs={}):
    #         return

    # custom_callback = CustomCallback()


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



