"""

Improved version of experiment.py

Date: 20th January 2020

"""

import os
import numpy as np
from numpy.fft import fft2, fftshift
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.backend.tensorflow_backend import tf

# PARAMETERS
pix = 30                    # Pixels to crop the PSF
N_PIX = 256

WAVE = 1.5      # microns
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

def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True):
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
    N_in_D = 2*rho_aper/delta
    print('%.2f actuators in D' %N_in_D)
    xx, yy = np.meshgrid(x0, x0)
    x_f = xx.flatten()
    y_f = yy.flatten()

    act = []
    for x_c, y_c in zip(x_f, y_f):
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        if r < rho_aper - delta/2 and r > rho_obsc + delta/2:
            act.append([x_c, y_c])

    if radial:
        for r in [rho_aper, rho_obsc]:
            N_radial = int(np.floor(2*np.pi*r/delta))
            d_theta = 2*np.pi / N_radial
            theta = np.linspace(0, 2*np.pi - d_theta, N_radial)
            # print(theta)
            for t in theta:
                act.append([r*np.cos(t), r*np.sin(t)])

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
    # alpha_pc is the height of the Gaussian at a distance of 1 Delta in %
    alpha = 1/np.sqrt(np.log(100/alpha_pc))

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

        pupil_function = self.pupil_mask * np.exp(2*np.pi * 1j * phase)
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


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    N_actuators = 20
    centers = actuator_centres(N_actuators, radial=True)
    # centers = radial_grid(N_radial=5)
    N_act = len(centers[0])
    plot_actuators(centers)

    actuator_matrices = actuator_matrix(centers, alpha_pc=30)        # Actuator matrix

    c_act = np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(actuator_matrices[0], c_act)
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

    PSF = PointSpreadFunction(actuator_matrices)

    class NoiseEffects(object):

        """
        A variety of Noise Effects to add to PSF images
        """

        def __init__(self):

            pass

        def add_readout_noise(self, PSF_images, RMS_READ):
            """
            Add Readout Noise in the form of additive Gaussian noise
            at a given RMS_READ equal to 1 / SNR, the Signal To Noise ratio

            Assuming the perfect PSF has a peak of 1.0
            :param PSF_images: datacube of [N_samples, pix, pix, 2] nominal and defocus PSF images
            :param RMS_READ: 1/ SNR
            :return:
            """

            N_samples, pix, N_chan = PSF_images.shape[0], PSF_images.shape[1], PSF_images.shape[-1]
            PSF_images_noisy = np.zeros_like(PSF_images)
            print("Adding Readout Noise with RMS: %.3f" % RMS_READ)
            for k in range(N_samples):
                for j in range(N_chan):
                    read_out = np.random.normal(loc=0, scale=RMS_READ, size=(pix, pix))
                    PSF_images_noisy[k, :, :, j] = PSF_images[k, :, :, j] + read_out

            return PSF_images_noisy

        def add_flat_field(self, PSF_images, flat_delta):
            """
            Add Flat Field uncertainties in the form of a multiplicative map
            of Uniform Noise [1 - delta, 1 + delta]

            Such that if delta is 10 %, the pixel sensitivity is known to +- 10 %
            :param PSF_images: datacube of [N_samples, pix, pix, 2] nominal and defocus PSF images
            :param flat_delta: range of the Flat Field uncertainty [1- delta, 1 + delta]
            :return:
            """

            N_samples, pix, N_chan = PSF_images.shape[0], PSF_images.shape[1], PSF_images.shape[-1]
            print(r"Adding Flat Field errors [1 - $\delta$, 1 + $\delta$]: $\delta$=%.3f" % flat_delta)
            # sigma_uniform = flat_delta / np.sqrt(3)

            for k in range(N_samples):
                for j in range(N_chan):
                    flat_field = np.random.uniform(low=1-flat_delta, high=1 + flat_delta, size=(pix, pix))
                    PSF_images[k, :, :, j] *= flat_field
            return PSF_images

        def add_bad_pixels(self, PSF_images, p_bad=0.20, max_bad_pixels=3, BAD_PIX=1.0):

            N_samples, pix, N_chan = PSF_images.shape[0], PSF_images.shape[1], PSF_images.shape[-1]

            # Randomly select which PSF images [nom, foc] will have bad_pixels
            which_images_nom = np.random.binomial(n=1, p=p_bad, size=N_samples)
            index_nom = np.argwhere(which_images_nom == 1)
            which_images_foc = np.random.binomial(n=1, p=p_bad, size=N_samples)
            index_foc = np.argwhere(which_images_foc == 1)

            # Nominal Channel
            for i_nom in index_nom:
                how_many = np.random.randint(low=1, high=max_bad_pixels)        # How many bad pixels?
                for j in range(how_many):                                       # Which (x, y) pixels?
                    x_bad, y_bad = np.random.choice(pix, size=1)[0], np.random.choice(pix, size=1)[0]
                    PSF_images[i_nom[0], x_bad, y_bad, 0] = BAD_PIX

            # Defocus Channel
            for i_foc in index_foc:
                how_many = np.random.randint(low=1, high=max_bad_pixels)        # How many bad pixels?
                for j in range(how_many):                                       # Which (x, y) pixels?
                    x_bad, y_bad = np.random.choice(pix, size=1)[0], np.random.choice(pix, size=1)[0]
                    PSF_images[i_foc[0], x_bad, y_bad, 1] = BAD_PIX

            return PSF_images

    class Calibration(object):

        def __init__(self, PSF_model, diversity_strength=1./(2*np.pi), noise_effects=NoiseEffects()):

            self.PSF_model = PSF_model
            self.diversity_strength = diversity_strength
            self.noise_effects = noise_effects

        def generate_datasets(self, N_batches, N_train, N_test, coef_strength, rescale=0.35):
            """
            Generate the [clean] datasets necessary to train a calibration model
            We split the training in N_batches of size N_train PSF images

            The function generates a total of (N_batches * N_train + N_test) PSF images with random aberrations

            [perfect_PSFs] is a datacube of fftshifted PSF images with no aberrations.
            We need them to evaluate the loss function.

            :param N_batches: How many batches of size N_train to create
            :param N_train: Number of PSF images within a batch
            :param N_test: Number of PSF images to test the performance
            :param coef_strength: Strength of the aberration coefficients, used to modulate the RMS
            :param rescale: Rescale the Training Coefficients between [1] and [rescale] to cover a wider range of RMS
            :return:
            """

            N_act = self.PSF_model.N_act
            N_samples = N_batches * N_train + N_test
            N_channels = 2

            # Perfect PSF (N_train N_PIX, N_PIX) - For the Loss function
            im_perfect, _s = self.PSF_model.compute_PSF(np.zeros(N_act), crop=False)
            perfect_PSFs = np.stack(N_train * [fftshift(im_perfect)])
            # NOTE: Tensorflow does not have FFTSHIFT operation. So we have to fftshift the Perfect PSF
            # back to the weird un-shifted format.

            # Generate a Phase Diversity coef if it doesn't exist yet
            try:
                coef_diversity = self.diversity_strength * np.load('diversity.npy')
            except FileNotFoundError:
                diversity = np.random.uniform(low=-1, high=1, size=N_act)
                np.save('diversity', diversity)
                coef_diversity = self.diversity_strength * np.load('diversity.npy')

            # Generate random aberration coefficients
            coefs = coef_strength * np.random.uniform(low=-1, high=1, size=(N_samples, N_act))
            # Rescale the coefficients to account for different Strehl ratios
            rescale_train = np.linspace(1.0, rescale, N_batches * N_train)
            rescale_test = np.linspace(1.0, 0.5, N_test)
            rescale_coef = np.concatenate([rescale_train, rescale_test])
            coefs *= rescale_coef[:, np.newaxis]

            dataset = np.zeros((N_samples, pix, pix, N_channels))

            # Loop over the samples, generating nominal and defocus images
            print("\nGenerating %d PSF images" % N_samples)
            for i in range(N_samples):
                if i % 100 == 0:
                    print(i)
                im0, _s = self.PSF_model.compute_PSF(coefs[i])       # Nominal
                images = [im0]
                imfoc, _s = self.PSF_model.compute_PSF(coefs[i] + coef_diversity)
                images.append(imfoc)
                dataset[i] = np.moveaxis(np.array(images), 0, -1)

            # Separate the batches
            train_batches = [dataset[i*N_train:(i+1)*N_train] for i in range(N_batches)]
            coef_batches = [coefs[i * N_train:(i + 1) * N_train] for i in range(N_batches)]
            test_images, test_coefs = dataset[N_batches * N_train:], coefs[N_batches * N_train:]
            print("Finished")
            return train_batches, coef_batches, test_images, test_coefs, perfect_PSFs

        def update_PSF(self, coefs):
            """
            Updates the PSF images after calibration. We use this to run the calibration
            for several iterations until it converges
            :param coefs: residual coefficients
            :return:
            """

            print("\nUpdating the PSF images")
            N_channels = 2
            N_samples = coefs.shape[0]
            coef_diversity = self.diversity_strength * np.load('diversity.npy')
            dataset = np.zeros((N_samples, pix, pix, N_channels))
            for i in range(N_samples):
                if i % 100 == 0:
                    print(i)
                im0, _s = self.PSF_model.compute_PSF(coefs[i])
                images = [im0]
                imfoc, _s = self.PSF_model.compute_PSF(coefs[i] + coef_diversity)
                images.append(imfoc)
                dataset[i] = np.moveaxis(np.array(images), 0, -1)
            print("Updated")
            return dataset

        def create_calibration_model(self, name="CALIBR"):
            """
            Creates a CNN model for NCPA calibration
            :return:
            """
            input_shape = (pix, pix, 2,)  # Multiple Wavelength Channels
            model = Sequential()
            model.name = name
            model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(Conv2D(8, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(self.PSF_model.N_act))
            model.summary()

            self.calibration_model = model
            return

        def validation_loss(self, test_images, test_coefs):
            """
            Validation loss to see how the model performs
            We compare the norm of the residual coefficients to the initial norm
            :param test_images:
            :param test_coefs:
            :return:
            """
            guess_coef = self.calibration_model.predict(test_images)
            residual = test_coefs + guess_coef
            norm_coefs = norm(test_coefs, axis=1)
            norm_residual = norm(residual, axis=1)
            ratio = np.mean(norm_residual / norm_coefs) * 100
            return ratio

        def train_calibration_model(self, images_batches, coefs_batches, test_images, test_coefs,
                                    N_loops=10, epochs_loop=50, verbose=1, plot_val_loss=False,
                                    readout_noise=False, RMS_readout=[1./100],
                                    flat_field=False, flat_delta=[1./100, 2.5/100, 5./100]):
            """
            Train a CNN calibration model to estimate the aberrations from PSF images

            We use batches because the loss function accepts only a FIXED length for the datacubes
            In particular, of size N_train. If we want to train with many examples, this blows up the
            GPU memory, so we run N_batches of size N_train, N_loop times

            In other words, if N_loops = 5, N_batches = 10, N_train = 1000,
            we loop 5 times over a training set of 10,000 images, in 10 batches of 1,000.
            This is very helpful as it allow us to add NOISE EFFECTS multiple times (random instances)
            and make the calibration more robust to such effects

            NOISE_EFFECTS | We can add:
                - READOUT NOISE only
                - FLAT FIELD and READOUT NOISE
                - Nothing at all
            for each case of noise, we can cover a certain range, by randomly selecting from the list

            :param images_batches: list of the form [train_images1, train_images2, ...] ("clean")
            :param coefs_batches: list of the form [train_coefs1, train_coefs2, ...] ("clean")
            :param test_images: datacube of test PSF images ("clean")
            :param test_coefs: datacube of test coefficients
            :param N_loops: Number of loops to run over all batches of images
            :param epochs_loop: Number of epochs per loop to train the model
            :param verbose: Keras verbose option: 0 silent, 1 display stuff
            :param plot_val_loss: whether to plot the validation loss evolution at the end
            :param readout_noise: whether to add READOUT NOISE
            :param RMS_readout: how much READOUT NOISE to add, list of RMS [1/SNR1, 1/ SNR2]
            :param flat_field: whether to add FLAT FIELD errors
            :param flat_delta: how much FLAT FIELD to add, list of deltas [delta1, delta2]
            :return:
            """

            N_batches = len(images_batches)
            N_train = images_batches[0].shape[0]
            loss, val_loss = [], []
            print("\nTraining the Calibration Model")
            for i_times in range(N_loops):                  # Loop over all batches N_loops times
                for k_batch in range(N_batches):            # Loop over each training batch
                    print("\nIteration %d || Batch #%d (%d samples)" % (i_times + 1, k_batch + 1, N_train))

                    clean_images = images_batches[k_batch]
                    clean_coefs = coefs_batches[k_batch]

                    # Noise effects
                    if flat_field is True and readout_noise is True:
                        print("--0--")
                        i_flat = np.random.choice(range(len(flat_delta)))  # Randomly select the Flat Field
                        clean_images = self.noise_effects.add_flat_field(clean_images, flat_delta=flat_delta[i_flat])
                        noisy_test_images = self.noise_effects.add_flat_field(test_images, flat_delta=flat_delta[i_flat])

                        i_noise = np.random.choice(range(len(RMS_readout)))         # Randomly select the RMS Readout
                        clean_images = self.noise_effects.add_readout_noise(clean_images, RMS_READ=RMS_readout[i_noise])
                        noisy_test_images = self.noise_effects.add_readout_noise(noisy_test_images, RMS_READ=RMS_readout[i_noise])

                    if flat_field is False and readout_noise is True:
                        print("--1--")
                        i_noise = np.random.choice(range(len(RMS_readout)))         # Randomly select the RMS Readout
                        clean_images = self.noise_effects.add_readout_noise(clean_images, RMS_READ=RMS_readout[i_noise])
                        noisy_test_images = self.noise_effects.add_readout_noise(test_images, RMS_READ=RMS_readout[i_noise])

                    if flat_field is False and readout_noise is False:
                        print("--2--")
                        noisy_test_images = test_images

                    # Allergen Notice: At this point clean_images may contain noise
                    train_history = self.calibration_model.fit(x=clean_images, y=clean_coefs,
                                                               epochs=epochs_loop, batch_size=N_train,
                                                               shuffle=False, verbose=verbose)
                    loss.extend(train_history.history['loss'])
                    val_loss.append(self.validation_loss(noisy_test_images, test_coefs))

            if plot_val_loss:
                plt.figure()
                plt.plot(val_loss)
                plt.xlabel('Epoch')
                plt.ylabel('Validation Loss')

            return loss, val_loss

        def calculate_RMS(self, coef_before, coef_after):
            """
            Using the PSF model, calculate the RMS wavefront BEFORE and AFTER corrections
            :param coef_before:
            :param coef_after:
            :return:
            """

            N_samples = coef_before.shape[0]
            print("\nCalculating RMS before / after for %d samples" % N_samples)
            RMS0, RMS = [], []
            for k in range(N_samples):
                wavef_before = WAVE * 1e3 * np.dot(self.PSF_model.RBF_flat, coef_before[k])
                wavef_after = WAVE * 1e3 * np.dot(self.PSF_model.RBF_flat, coef_after[k])
                RMS0.append(np.std(wavef_before))
                RMS.append(np.std(wavef_after))
            mu0, mu = np.mean(RMS0), np.mean(RMS)
            med0, med = np.median(RMS0), np.median(RMS)
            std0, std = np.std(RMS0), np.std(RMS)
            print("RMS Before: %.1f +- %.1f nm (%.1f median)" % (mu0, std0, med0))
            print("RMS  After: %.1f +- %.1f nm (%.1f median)" % (mu, std, med))
            return RMS0, RMS

        def calibrate_iterations(self, test_images, test_coefs, N_iter=3,
                                 readout_noise=False, RMS_readout=1./200,
                                 flat_field=False, flat_delta=5. / 100):
            """
            Run the calibration for several iterations
            :param test_images: datacube of test PSF images ("clean")
            :param test_coefs: datacube of test coefficients ("clean")
            :param N_iter: how many iterations to run
            :param readout_noise: whether to add READOUT NOISE
            :param RMS_readout: how much READOUT NOISE to add, list of RMS [1/SNR1, 1/ SNR2]
            :param flat_field: whether to add FLAT FIELD errors
            :param flat_delta: how much FLAT FIELD to add, list of deltas [delta1, delta2]
            :return:
            """

            if flat_field is True and readout_noise is True:
                flat_test_images = self.noise_effects.add_flat_field(test_images, flat_delta=flat_delta)
                images_before = self.noise_effects.add_readout_noise(flat_test_images, RMS_READ=RMS_readout)

            if flat_field is False and readout_noise is True:
                images_before = self.noise_effects.add_readout_noise(test_images, RMS_READ=RMS_readout)

            if flat_field is False and readout_noise is False:
                images_before = test_images
            else:
                raise ValueError

            coefs_before = test_coefs
            RMS_evolution = []
            for k in range(N_iter):
                print("\nNCPA Calibration | Iteration %d/%d" % (k + 1, N_iter))
                predicted_coefs = self.calibration_model.predict(images_before)
                coefs_after = coefs_before + predicted_coefs  # Remember we predict the Corrections!
                rms_before, rms_after = self.calculate_RMS(coefs_before, coefs_after)
                rms_pair = [rms_before, rms_after]
                RMS_evolution.append(rms_pair)

                # Update the PSF and coefs
                images_before = cal.update_PSF(coefs_after)
                coefs_before = coefs_after

                if flat_field is True and readout_noise is True:
                    images_before = self.noise_effects.add_flat_field(images_before, flat_delta=flat_delta)
                    images_before = self.noise_effects.add_readout_noise(images_before, RMS_READ=RMS_readout)

                if flat_field is False and readout_noise is True:
                    images_before = self.noise_effects.add_readout_noise(images_before, RMS_READ=RMS_readout)

                if flat_field is False and readout_noise is False:
                    pass

                else:
                    raise ValueError

            return RMS_evolution

        def plot_RMS_evolution(self, RMS_evolution):
            """
            Plot the evolution of RMS wavefront with calibration iterations
            :param RMS_evolution: list of pairs of RMS [(BEFORE, AFTER)_0, ..., (BEFORE, AFTER)_N]
            """

            N_pairs = len(RMS_evolution)  # Pairs of [Before, After]
            blues = cm.Blues(np.linspace(0.5, 1.0, N_pairs))

            plt.figure()
            for i, rms_pair in enumerate(RMS_evolution):
                before, after = rms_pair[0], rms_pair[1]
                print(len(before), len(after))
                label = r"%d | $\mu$=%.1f $\pm$ %.1f nm" % (i + 1, np.mean(after), np.std(after))
                plt.scatter(before, after, s=4, color=blues[i], label=label)

            plt.xlim([0, 300])
            plt.ylim([0, 200])
            plt.grid(True)
            plt.legend(title='Iteration', loc=2)
            plt.xlabel(r'RMS wavefront BEFORE [nm]')
            plt.ylabel(r'RMS wavefront AFTER [nm]')


    # ================================================================================================================ #
    #
    # ================================================================================================================ #

    """ Create the Calibration Object and generate the CLEAN datasets """

    cal = Calibration(PSF_model=PSF)
    coef_strength = 1.5 / (2 * np.pi)
    N_batches = 20
    N_train, N_test = 500, 1000
    train_batches, coef_batches, test_images, test_coefs, perfect_PSFs = cal.generate_datasets(N_batches, N_train,
                                                                                               N_test, coef_strength)
    # _perfect_PSFs = cal.noise_effects.add_readout_noise(perfect_PSFs[:,:,:, np.newaxis], RMS_READ=1./200)

    """ Create the Calibration Model """
    cal.create_calibration_model()

    # Some bits for the Loss function definition
    phase_model_matrix = cal.PSF_model.RBF_mat.copy().T
    # Roll it to match the Theano convention for dot product that TF uses
    phase_model_matrix = np.rollaxis(phase_model_matrix, 1, 0)  # Model Matrix to compute the Phase with Actuators
    pup = cal.PSF_model.pupil_mask.copy()  # Pupil Mask
    peak = cal.PSF_model.PEAK.copy()  # Peak to normalize the FFT calculations

    # Transform them to TensorFlow
    pupil_tf = tf.constant(pup, dtype=tf.float32)
    model_matrix_tf = tf.constant(phase_model_matrix, dtype=tf.float32)
    # coef_t = tf.constant(train_coefs[:1500], dtype=tf.float32)
    perfect_tf = tf.constant(perfect_PSFs, dtype=tf.float32)

    def loss(y_true, y_pred):
        phase = 2*np.pi * K.dot(y_true + y_pred, model_matrix_tf)
        cos_x, sin_x = pupil_tf * K.cos(phase), pupil_tf * K.sin(phase)
        complex_phase = tf.complex(cos_x, sin_x)
        image = (K.abs(tf.fft2d(complex_phase))) ** 2 / peak

        res = K.mean(K.sum((image - perfect_tf) ** 2))
        # res = K.mean(K.sum((image - perfect_psf)**2)) / 500- K.mean(K.max(image, axis=(1, 2)))

        return res

    # Compile the calibration model
    cal.calibration_model.compile(optimizer='adam', loss=loss)

    """ Train the model """

    RMS_READ = 1. / 500

    loss, validation = cal.train_calibration_model(train_batches, coef_batches, test_images, test_coefs,
                                                   N_loops=25, epochs_loop=10, verbose=1, plot_val_loss=True,
                                                   readout_noise=True, RMS_readout=[RMS_READ],
                                                   flat_field=False, flat_delta=[0.0])

    RMS_evolution = cal.calibrate_iterations(test_images, test_coefs, N_iter=4,
                                             readout_noise=True, RMS_readout=RMS_READ)
    cal.plot_RMS_evolution(RMS_evolution)
    plt.show()


    ###
    SNR = [50, 100, 200, 300, 400, 500, 750, 1000]
    mus_ = [79.4, 54.0, 41.5, 34.1, 31.5, 27.1, 25.7, 24.1]
    stds_ = [17.1, 11.4, 8.5, 7.7, 6.8, 5.8, 6.4, 5.8]

    # plt.scatter(SNR, mus_)

    plt.figure()

    # plt.plot(pc_err, mean_err, color='black')
    plt.errorbar(SNR, mus_, yerr=stds_, fmt='o')
    plt.xlabel(r'SNR')
    plt.ylabel(r'RMS wavefront after calibration [nm]')
    plt.grid(True)
    plt.xlim([0, 1000])
    plt.ylim([0, 100])
    plt.show()











