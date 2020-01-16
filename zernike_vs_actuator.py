"""
Date: 20th November 2019

Comparison of Basis Function for Wavefront Definition in the context of ML calibration
________________________________________________________________________________________

Is there a basis that works better for calibration?
Is Zernike polynomials better-suited than Actuator Commands models?




"""

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import zern_core as zern
import time

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from numpy.linalg import norm as norm

# PARAMETERS
Z = 0.75                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 30                    # Pixels to crop the PSF
N_PIX = 1024                 # Pixels for the Fourier arrays

ELT_DIAM = 39
MILIARCSECS_IN_A_RAD = 206265000

def spaxel_scale(scale=4, wave=1.0):
    """
    Compute the aperture radius necessary to have a
    certain SPAXEL SCALE [in mas] at a certain WAVELENGTH [in microns]
    :param scale:
    :return:
    """

    scale_rad = scale / MILIARCSECS_IN_A_RAD
    rho = scale_rad * ELT_DIAM / (wave * 1e-6)
    print(rho)

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

WAVE = 1.5
SPAXEL_MAS = 4
RHO_APER = rho_spaxel_scale(SPAXEL_MAS, WAVE)
RHO_OBSC = 0.30 * RHO_APER             # Central obscuration


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

def draw_actuator_commands(commands, centers):
    """
    Plot of each actuator commands
    :param commands:
    :param centers:
    :return:
    """
    cent, delta = centers
    x = np.linspace(-1, 1, 2*N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    image = np.zeros((2*N_PIX, 2*N_PIX))
    for i, (xc, yc) in enumerate(cent):
        act_mask = (xx - xc)**2 + (yy - yc)**2 <= (delta/3)**2
        image += commands[i] * act_mask

    return image


# ==================================================================================================================== #
#                                          Point Spread Function
# ==================================================================================================================== #

class PointSpreadFunctionFast(object):
    """
    Computes the PSF image and Strehl ratio for a given set of coefficients
    """
    minPix, maxPix = (N_PIX + 1 - pix) // 2, (N_PIX + 1 + pix) // 2

    def __init__(self, matrices):
        """
        Matrices is a list that contains the Model Matrix (Zernike or Actuator),
        the Pupil Mask (used for FFT calculations) and the Model Matrix flattened
        :param matrices:
        """

        self.N_act = matrices[0].shape[-1]
        self.RBF_mat = matrices[0].copy()
        self.pupil_mask = matrices[1].copy()
        self.RBF_flat = matrices[2].copy()
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
        phase_flat = np.dot(self.RBF_flat, coef)
        phase_datacube = invert_mask(phase_flat, self.pupil_mask)
        pupil_function = self.pupil_mask * np.exp(1j * phase_datacube)

        # print("\nSize of Complex Pupil Function array: %.3f Gbytes" %(pupil_function.nbytes / 1e9))

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
        # print("\nSize of Image : %.3f Gbytes" % (image.nbytes / 1e9))
        return image, strehl

class Zernike_fit(object):

    """
    Uses Least Squares to fit Wavefronts back and forth between two basis
    the Zernike polynomials and an Actuator Commands model

    """

    def __init__(self, PSF_zernike, PSF_actuator):
        self.PSF_zernike = PSF_zernike
        self.PSF_actuator = PSF_actuator

        # Get the Model matrices
        self.H_zernike = self.PSF_zernike.RBF_mat.copy()
        self.H_zernike_flat = self.PSF_zernike.RBF_flat.copy()
        self.H_actuator = self.PSF_actuator.RBF_mat.copy()
        self.H_actuator_flat = self.PSF_actuator.RBF_flat.copy()

        self.pupil_mask = self.PSF_zernike.pupil_mask.copy()

    def plot_example(self, zern_coef, actu_coef, ground_truth='zernike', k=0, cmap='bwr'):

        if ground_truth == "zernike":
            true_phase = np.dot(self.H_zernike, zern_coef.T)[:, :, k]
            fit_phase = np.dot(self.H_actuator, actu_coef)[:, :, k]
            names = ['Zernike', 'Actuator']
            print(0)

        elif ground_truth == "actuator":
            true_phase = np.dot(self.H_actuator, actu_coef.T)[:, :, k]
            fit_phase = np.dot(self.H_zernike, zern_coef)[:, :, k]
            names = ['Actuator', 'Zernike']
            print(1)

        residual = true_phase - fit_phase
        rms0 = np.std(true_phase[self.pupil_mask])
        rms = np.std(residual[self.pupil_mask])

        mins = min(true_phase.min(), fit_phase.min())
        maxs = max(true_phase.max(), fit_phase.max())
        m = min(mins, -maxs)
        # mapp = 'bwr'

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = plt.subplot(1, 3, 1)
        img1 = ax1.imshow(true_phase, cmap=cmap, extent=[-1, 1, -1, 1])
        ax1.set_title('%s Wavefront [$\sigma=%.3f \lambda$]' % (names[0], rms0))
        img1.set_clim(m, -m)
        ax1.set_xlim([-RHO_APER, RHO_APER])
        ax1.set_ylim([-RHO_APER, RHO_APER])
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 3, 2)
        img2 = ax2.imshow(fit_phase, cmap=cmap, extent=[-1, 1, -1, 1])
        ax2.set_title('%s Fit Wavefront' % names[1])
        img2.set_clim(m, -m)
        ax2.set_xlim([-RHO_APER, RHO_APER])
        ax2.set_ylim([-RHO_APER, RHO_APER])
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

        ax3 = plt.subplot(1, 3, 3)
        img3 = ax3.imshow(residual, cmap=cmap, extent=[-1, 1, -1, 1])
        ax3.set_title('Residual [$\sigma=%.3f \lambda$]' % rms)
        img3.set_clim(m, -m)
        ax3.set_xlim([-RHO_APER, RHO_APER])
        ax3.set_ylim([-RHO_APER, RHO_APER])
        plt.colorbar(img3, ax=ax3, orientation='horizontal')

    def fit_actuator_wave_to_zernikes(self, actu_coef, plot=False, cmap='bwr'):
        """
        Fit a Wavefront defined in terms of Actuator commands to
        Zernike polynomials

        :param actu_coef:
        :param plot: whether to plot an example to show the fitting error
        :return:
        """

        actu_wave = np.dot(self.H_actuator_flat, actu_coef.T)
        x_zern = self.least_squares(y_obs=actu_wave, H=self.H_zernike_flat)

        if plot:
            self.plot_example(x_zern, actu_coef, ground_truth="actuator", cmap=cmap)

        return x_zern

    def fit_zernike_wave_to_actuators(self, zern_coef, plot=False, cmap='bwr'):
        """
        Fit a Wavefront defined in terms of Zernike polynomials to the
        model of Actuator Commands

        :param zern_coef:
        :param plot: whether to plot an example to show the fitting error
        :return:
        """

        # Generate Zernike Wavefronts [N_pix, N_pix, N_samples]
        zern_wave = np.dot(self.H_zernike_flat, zern_coef.T)
        x_act = self.least_squares(y_obs=zern_wave, H=self.H_actuator_flat)

        if plot:
            self.plot_example(zern_coef, x_act, ground_truth="zernike", cmap=cmap)

        return x_act

    def least_squares(self, y_obs, H):
        """
        High level definition of the Least Squares fitting problem

        y_obs = H * x_fit + noise
        H.T * y_obs = (H.T * H) * x_fit
        with N = H.T * H
        x_fit = inv(N) * H.T * y_obs

        H is the model matrix that we use for the fit
        For instance, if the wavefront (y_obs) is defined in terms of Zernike polynomials
        H would be the Model Matrix for the Actuator Commands
        and x_act would be the best fit in terms of actuator commands that describe that wavefront
        :param y_obs:
        :param H:
        :return:
        """

        Ht = H.T
        Hty_obs = np.dot(Ht, y_obs)
        N = np.dot(Ht, H)
        invN = np.linalg.inv(N)
        x_fit = np.dot(invN, Hty_obs)

        return x_fit


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### ============================================================================================================ ###
    #      EXPERIMENT (A) - Comparison with PSF images based on ZERNIKE polynomials
    ### ============================================================================================================ ###

    """ (1) Define the ACTUATOR model for the WAVEFRONT """

    N_actuators = 25
    centers, MAX_FREQ = actuator_centres(N_actuators)
    N_act = len(centers[0])
    plot_actuators(centers)

    alpha_pc = 30       # Height [percent] at the neighbour actuator
    rbf_mat = actuator_matrix(centers, alpha_pc=alpha_pc)        # Actuator matrix
    pupil = rbf_mat[1]

    c_act = np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(rbf_mat[0], c_act)
    p0 = min(np.min(phase0), -np.max(phase0))

    plt.figure()
    plt.imshow(phase0, extent=(-1,1,-1,1), cmap='bwr')
    plt.colorbar()
    plt.clim(p0, -p0)
    for c in centers[0]:
        plt.scatter(c[0], c[1], color='black', s=4)
    plt.xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.title(r'%d Actuators | Wavefront [$\lambda$]' % N_act)
    plt.show()

    """ (2) Define the ZERNIKE model for the WAVEFRONT """

    N_zern = 50
    x = np.linspace(-1, 1, N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)

    rho, theta = rho[pupil], theta[pupil]
    zernike = zern.ZernikeNaive(mask=pupil)
    _phase = zernike(coef=np.zeros(N_zern), rho=rho / (1.15*RHO_APER), theta=theta, normalize_noll=False,
                     mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix[:, 3:]
    H_matrix = zern.invert_model_matrix(H_flat, pupil)
    N_zern = H_matrix.shape[-1]

    """ (3) Define the PSF models for each case """

    PSF_actuator = PointSpreadFunctionFast(rbf_mat)
    PSF_zernike = PointSpreadFunctionFast([H_matrix, pupil, H_flat])

    ### Show the LS fit error when going back and forth
    fit = Zernike_fit(PSF_zernike, PSF_actuator)
    rand_zern = np.random.uniform(low=-1, high=1, size=(1, N_zern))
    fit_actu = fit.fit_zernike_wave_to_actuators(rand_zern, plot=True, cmap='seismic').T
    rand_actu = 2*np.random.uniform(low=-1, high=1, size=(1, N_act))
    back_zern = fit.fit_actuator_wave_to_zernikes(rand_actu, plot=True, cmap='seismic')
    plt.show()

    """ (4) Generate the TRAINING sets """

    defocus_strength = 1.5
    def generate_training_set_zernike(PSF_zernike, PSF_actuator, N_train=1500, N_test=500):
        """
        Generate a dataset of PSF images using the Zernike model as wavefront definition
        The Wavefronts are then matched to an Actuator model via Least-Squares
        because we will train 2 indentical ML models:

        - ZERNIKE: trained to recognize the Zernike coefficients from the PSF images
        - ACTUATOR: trained to recognize the Actuator commands (the LS fit)
        :param PSF_zernike: PSF model based on Zernike polynomials
        :param PSF_actuator: PSF model based on Actuator commands
        :param N_train:
        :param N_test:
        :return:
        """

        start = time.time()
        N_samples = N_train + N_test
        N_zern = PSF_zernike.RBF_mat.shape[-1]
        dataset = np.empty((N_samples, pix, pix, 2))

        # Zernike Coefficient
        coef_zern = np.random.uniform(low=-Z, high=Z, size=(N_samples, N_zern))
        defocus = np.zeros(N_zern)
        defocus[1] = defocus_strength

        # Fit Zernikes to Actuator Commands to get the "ground-truth" for the ML model
        print("\nRunning LS fit")
        fit = Zernike_fit(PSF_zernike, PSF_actuator)
        coef_actu = fit.fit_zernike_wave_to_actuators(coef_zern).T
        print("LS fit done")

        for k in range(N_samples):
            if k % 100 == 0:
                print(k)
            dataset[k, :, :, 0], _s = PSF_zernike.compute_PSF(coef_zern[k])
            dataset[k, :, :, 1], _s0 = PSF_zernike.compute_PSF(coef_zern[k] + defocus)

        end = time.time()
        dt = end - start
        print("\n%d examples created in %.3f sec" % (N_samples, dt))
        print("\n%.3f sec/example" % (dt / N_samples))

        return dataset[:N_train], dataset[N_train:], coef_zern[:N_train], coef_zern[N_train:], \
               coef_actu[:N_train], coef_actu[N_train:]


    N_train, N_test = 15000, 500
    # N_train, N_test = 50, 10

    train_PSF, test_PSF, train_zern, \
    test_zern, train_actu, test_actu = generate_training_set_zernike(PSF_zernike,
                                                                     PSF_actuator,
                                                                     N_train, N_test)

    """ Check the Fitting Error """
    # Make sure that when you LS fit the Zernike wavefront to
    # the Actuator model, the residual error is sufficiently small
    # less thatn 20% RMS or less

    # as this residual error will be translated into performance error
    # after calibration. Ideally it should be smaller than the
    # Calibration error of the Zernike CNN Model
    zern_coef = np.concatenate([train_zern, test_zern], axis=0)
    actu_coef = np.concatenate([train_actu, test_actu], axis=0)

    rms0, rms_fit = [], []
    # for k in range(N_train + N_test):
    for k in range(500):
        if k % 100 == 0:
            print(k)
        _phas = np.dot(PSF_zernike.RBF_mat, zern_coef[k])
        rms0.append(np.std(_phas[pupil]))

        _fit = _phas - np.dot(PSF_actuator.RBF_mat, actu_coef[k])
        rms_fit.append(np.std(_fit[pupil]))

    XMAX = 1.25
    _x = np.linspace(0, XMAX, 100)
    _10p = 0.1 * _x
    _5p = 0.05 * _x
    _2p = 0.025 * _x
    plt.figure()
    plt.scatter(rms0, rms_fit, s=6)
    plt.plot(_x, _10p, label='10', color='black', linestyle='--')
    plt.plot(_x, _5p, label='5', color='red', linestyle='-.')
    plt.plot(_x, _2p, label='2.5', color='orange', linestyle=':')
    plt.xlabel(r'RMS wavefont BEFORE fit [$\lambda$]')
    plt.ylabel(r'RMS wavefont residual AFTER fit [$\lambda$]')
    plt.xlim([0, XMAX])
    plt.ylim([0, 0.075])
    plt.legend(title='Residual [percent]')
    plt.show()

    plt.figure()
    plt.hist(rms0, bins=20, histtype='step', label='Zernike Model', color='coral')
    # plt.axvline(np.median(RMS_zern_rel), linestyle='--', color='coral')
    plt.hist(rms_fit, bins=20, histtype='step', label='Actuator Model')
    # plt.axvline(np.median(RMS_actu_rel), linestyle='--')
    plt.xlim([0, 1.50])
    plt.xlabel(r'RMS wavefront [$\lambda$]')
    plt.legend()

    ### ============================================================================================================ ###
    #                                              MACHINE LEARNING bits
    ### ============================================================================================================ ###

    def create_model(name, N_output):
        N_channels = 2
        input_shape = (pix, pix, N_channels,)
        model = Sequential()
        model.name = name
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(N_output))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    ### ZERNIKE model
    zern_model = create_model("ZERNIKE", N_output=N_zern)

    train_history_zern = zern_model.fit(x=train_PSF, y=train_zern, validation_data=(test_PSF, test_zern),
                                   epochs=50, batch_size=32, shuffle=True, verbose=1)
    guess_zern = zern_model.predict(test_PSF)
    residual_zern = test_zern - guess_zern
    norm_zern = np.mean(norm(residual_zern, axis=-1)) / N_zern

    ### ACTUATOR model
    actu_model = create_model("ACTUATOR", N_output=N_act)

    train_history_actu = actu_model.fit(x=train_PSF, y=train_actu, validation_data=(test_PSF, test_actu),
                                        epochs=50, batch_size=32, shuffle=True, verbose=1)
    guess_actu = actu_model.predict(test_PSF)
    residual_actu = test_actu - guess_actu
    norm_actu = np.mean(norm(residual_actu, axis=-1)) / N_act

    print("\nWhich Model is better?")
    print("Average residual error: Mean(Norm(residual) / N_component)")
    print("Zernike: %.4f" %norm_zern)
    print("Actuators: %.4f" %norm_actu)

    ### ============================================================================================================ ###
    #                                            PERFOMANCE checks
    ### ============================================================================================================ ###

    # Distribution on the Command Errors
    # are some actuators more difficult to predict?
    mean_error = np.mean((residual_actu)**2, axis=0)
    mean_error_img = draw_actuator_commands(commands=mean_error, centers=centers)

    min_err = np.min(mean_error_img[np.nonzero(mean_error_img)])
    max_err = np.max(mean_error_img)

    plt.figure()
    plt.imshow(mean_error_img, cmap='Reds', extent=[-1, 1, -1, 1])
    plt.xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.colorbar()
    plt.title(r'Mean Squared Error | Actuator Residual [$\lambda^2$]')
    plt.show()




    def fix_axes(ax):
        ax.set_xlim([-RHO_APER, RHO_APER])
        ax.set_ylim([-RHO_APER, RHO_APER])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    extent = [-1, 1, -1, 1]

    def relative_performance(PSF_zernike, PSF_actuator, residual_zern, residual_actu):
        mapp = 'RdBu'
        RMS_zern, RMS_actu = [], []
        pupil = PSF_zernike.pupil_mask
        for k in range(N_test):

            err_zern = np.dot(PSF_zernike.RBF_mat, residual_zern[k])
            err_actu = np.dot(PSF_actuator.RBF_mat, residual_actu[k])

            rms_zern = np.std(err_zern[pupil])
            RMS_zern.append(rms_zern)
            rms_actu = np.std(err_actu[pupil])
            RMS_actu.append(rms_actu)

            if k < 10:
                m1 = min(np.min(err_zern), -np.max(err_zern))
                m2 = min(np.min(err_actu), -np.max(err_actu))
                m = min(m1, m2)

                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1 = plt.subplot(1, 2, 1)
                img1 = ax1.imshow(err_zern, cmap=mapp, extent=extent)
                ax1.set_title(r'Zernike Residual [$\sigma=%.3f \lambda$]' % rms_zern)
                fix_axes(ax1)
                img1.set_clim(m, -m)
                plt.colorbar(img1, ax=ax1, orientation='horizontal')

                ax2 = plt.subplot(1, 2, 2)
                img2 = ax2.imshow(err_actu, cmap=mapp, extent=extent)
                ax2.set_title('Actuator Residual [$\sigma=%.3f \lambda$]' % rms_actu)
                fix_axes(ax2)
                img2.set_clim(m, -m)
                plt.colorbar(img2, ax=ax2, orientation='horizontal')

        plt.figure()
        plt.hist(RMS_zern, bins=20, histtype='step', label='Zernike Model', color='coral')
        med_zern = np.median(RMS_zern)
        plt.axvline(med_zern, linestyle='--', color='coral', label=r'Median = %.3f $\lambda$' % med_zern)
        plt.hist(RMS_actu, bins=20, histtype='step', label='Actuator Model')
        med_actu = np.median(RMS_actu)
        plt.axvline(med_actu, linestyle='--', label=r'Median = %.3f $\lambda$' % med_actu)
        plt.xlim([0, 0.50])
        plt.xlabel(r'RMS wavefront [$\lambda$]')
        plt.legend()
        return

    ### Evaluate the RELATIVE performance (How good is each model in Machine Learning terms?)
    relative_performance(PSF_zernike, PSF_actuator, residual_zern, residual_actu)

    def absolute_performance(PSF_zernike, PSF_actuator, test_true, guess_zern, guess_actu, basis="zernike"):

        pupil = PSF_zernike.pupil_mask
        RMS0, RMS_zern, RMS_actu = [], [], []
        if basis == "zernike":
            PSF_basis = PSF_zernike
        elif basis == "actuator":
            PSF_basis = PSF_actuator
        else:
            Exception("basis should be either 'zernike' or 'actuator'")

        for k in range(N_test):

            true_wave = np.dot(PSF_basis.RBF_mat, test_true[k])
            rms0 = np.std(true_wave[pupil])
            RMS0.append(rms0)
            corr_zern = np.dot(PSF_zernike.RBF_mat, guess_zern[k])
            corr_actu = np.dot(PSF_actuator.RBF_mat, guess_actu[k])

            res_zern = true_wave - corr_zern
            rms_zern = np.std(res_zern[pupil])
            RMS_zern.append(rms_zern)
            res_actu = true_wave - corr_actu
            rms_actu = np.std(res_actu[pupil])
            RMS_actu.append(rms_actu)

            if k < 10:
                mapp = 'RdBu'
                m = min(np.min(true_wave), -np.max(true_wave))
                f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1 = plt.subplot(1, 3, 1)
                img1 = ax1.imshow(true_wave, cmap=mapp, extent=extent)
                fix_axes(ax1)
                ax1.set_title(r'True Wavefront ($\sigma=%.3f \lambda$)' %rms0)
                img1.set_clim(m, -m)
                plt.colorbar(img1, ax=ax1, orientation='horizontal')

                ax2 = plt.subplot(1, 3, 2)
                img2 = ax2.imshow(res_zern, cmap=mapp, extent=extent)
                fix_axes(ax2)
                ax2.set_title('Residual ML [Zernike Model] ($\sigma=%.3f \lambda$)' %rms_zern)
                img2.set_clim(m, -m)
                plt.colorbar(img2, ax=ax2, orientation='horizontal')

                ax3 = plt.subplot(1, 3, 3)
                img3 = ax3.imshow(res_actu, cmap=mapp, extent=extent)
                fix_axes(ax3)
                ax3.set_title('Residual ML [Actuator Model] ($\sigma=%.3f \lambda$)' %rms_actu)
                img3.set_clim(m, -m)
                plt.colorbar(img3, ax=ax3, orientation='horizontal')

        plt.figure()
        plt.hist(RMS0, bins=20, histtype='step', label='Initial')
        med_zern = np.median(RMS_zern)
        plt.axvline(med_zern, linestyle='--', color='red', label=r'Median = %.3f $\lambda$' % med_zern)
        plt.hist(RMS_zern, bins=20, histtype='step', color='red', label='Zernike Model')

        med_actu = np.median(RMS_actu)
        plt.axvline(med_actu, linestyle='--', color='green', label=r'Median = %.3f $\lambda$' % med_actu)
        plt.hist(RMS_actu, bins=20, histtype='step', color='green', label='Actuator Model')
        plt.legend()
        plt.xlabel(r'RMS wavefront [$\lambda$]')
        plt.show()
    plt.show()

    absolute_performance(PSF_zernike, PSF_actuator, test_zern, guess_zern, guess_actu, basis="zernike")



    alpha = 0.75
    plt.figure()

    plt.scatter(RMS0, RMS_zern, s=3, label='Zernike Model')
    plt.scatter(RMS0, RMS_actu, s=3, label='Actuator Model')
    plt.plot(_x, 0.2*_x, label='20 pc', color='black', linestyle='--', alpha=alpha)
    plt.plot(_x, 0.3*_x, label='30 pc', color='black', linestyle='-.', alpha=alpha)
    plt.plot(_x, 0.4*_x, label='40 pc', color='black', linestyle=':', alpha=alpha)
    plt.xlim([0.4, 1.2])
    plt.ylim([0, 0.5])
    plt.xlabel(r'RMS wavefront BEFORE [$\lambda$]')
    plt.ylabel(r'RMS wavefront AFTER [$\lambda$]')
    plt.legend()
    plt.show()

    ### ============================================================================================================ ###
    #                           Opposite Case - Initial Wavefronts are Actuator-based
    ### ============================================================================================================ ###


    # N_zern = 52
    # x = np.linspace(-1, 1, N_PIX, endpoint=True)
    # xx, yy = np.meshgrid(x, x)
    # rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
    # # pupil = (rho <= RHO_APER) & (rho > RHO_OBSC)
    #
    # rho, theta = rho[pupil], theta[pupil]
    # zernike = zern.ZernikeNaive(mask=pupil)
    # _phase = zernike(coef=np.zeros(N_zern), rho=rho / (1.15*RHO_APER), theta=theta, normalize_noll=False,
    #                  mode='Jacobi', print_option='Silent')
    # H_flat = zernike.model_matrix[:, 3:]
    # H_matrix = zern.invert_model_matrix(H_flat, pupil)
    # N_zern = H_matrix.shape[-1]
    #
    # """ (3) Define the PSF models for each case """
    #
    # PSF_zernike2 = PointSpreadFunctionFast([H_matrix, pupil, H_flat])
    #
    # fit = Zernike_fit(PSF_zernike2, PSF_actuator)
    # rand_actu = np.random.uniform(-1, 1, size=(100, N_act))
    # zern_fit = fit.fit_actuator_wave_to_zernikes(rand_actu, True).T
    #
    # defocus_strength = 1.5


    def generate_training_set_actuator(PSF_zernike, PSF_actuator, N_train=1500, N_test=500):

        start = time.time()
        N_samples = N_train + N_test
        N_act = PSF_actuator.RBF_mat.shape[-1]
        dataset = np.empty((N_samples, pix, pix, 2))

        # Actu Coefficient
        coef_actu = np.random.uniform(low=-1, high=1, size=(N_samples, N_act))
        defocus = np.random.uniform(low=-0.75, high=0.75, size=(N_act))
        # defocus[1] = defocus_strength

        # Fit Zernikes to Actuator Commands
        fit = Zernike_fit(PSF_zernike, PSF_actuator)
        coef_zern = fit.fit_actuator_wave_to_zernikes(coef_actu).T

        for k in range(N_samples):
            if k % 100 == 0:
                print(k)
            dataset[k,:,:,0], _s = PSF_actuator.compute_PSF(coef_actu[k])
            dataset[k,:,:,1], _s0 = PSF_actuator.compute_PSF(coef_actu[k] + defocus)

        end = time.time()
        dt = end - start
        print("\n%d examples created in %.3f sec" % (N_samples, dt))
        print("\n%.3f sec/example" % (dt / N_samples))

        return dataset[:N_train], dataset[N_train:], coef_zern[:N_train], coef_zern[N_train:], \
               coef_actu[:N_train], coef_actu[N_train:]


    N_train, N_test = 30000, 1000

    train_PSF2, test_PSF2, train_zern2, \
    test_zern2, train_actu2, test_actu2 = generate_training_set_actuator(PSF_zernike, PSF_actuator, N_train, N_test)

    ### ZERNIKE model
    zern_model2 = create_model("ZERNIKE", N_output=N_zern)

    train_history_zern2 = zern_model2.fit(x=train_PSF2, y=train_zern2, validation_data=(test_PSF2, test_zern2),
                                   epochs=50, batch_size=32, shuffle=True, verbose=1)
    guess_zern2 = zern_model2.predict(test_PSF2)
    residual_zern2 = test_zern2 - guess_zern2
    norm_zern2 = np.mean(norm(residual_zern2, axis=-1)) / N_zern

    ### ACTUATOR model
    actu_model2 = create_model("ACTUATOR", N_output=N_act)

    train_history_actu2 = actu_model2.fit(x=train_PSF2, y=train_actu2, validation_data=(test_PSF2, test_actu2),
                                        epochs=50, batch_size=32, shuffle=True, verbose=1)
    guess_actu2 = actu_model2.predict(test_PSF2)
    residual_actu2 = test_actu2 - guess_actu2
    norm_actu2 = np.mean(norm(residual_actu2, axis=-1)) / N_act

    print("\nWhich Model is better?")
    print("Average residual error: Mean(Norm(residual) / N_component)")
    print("Zernike: %.4f" %norm_zern2)
    print("Actuators: %.4f" %norm_actu2)

    relative_performance(PSF_zernike, PSF_actuator, residual_zern2, residual_actu2)
    absolute_performance(PSF_zernike, PSF_actuator, test_actu2, guess_zern2, guess_actu2, 'actuator')

    ### IMPACT OF THE NUMBER OF ZERNIKES

    defocus_strength = 1.5
    def training_zern(PSF_zernike, N_train=1500, N_test=500):

        start = time.time()

        N_samples = N_train + N_test
        N_zern = PSF_zernike.RBF_mat.shape[-1]
        dataset = np.empty((N_samples, pix, pix, 2))

        # Zernike Coefficient
        coef_zern = ZZ * np.random.uniform(low=-1, high=1, size=(N_samples, N_zern))
        defocus = np.zeros(N_zern)
        defocus[1] = defocus_strength

        for k in range(N_samples):
            if k % 100 == 0:
                print(k)
            dataset[k, :, :, 0], _s = PSF_zernike.compute_PSF(coef_zern[k])
            dataset[k, :, :, 1], _s0 = PSF_zernike.compute_PSF(coef_zern[k] + defocus)

        end = time.time()
        dt = end - start
        print("\n%d examples created in %.3f sec" % (N_samples, dt))
        print("\n%.3f sec/example" % (dt / N_samples))

        return dataset[:N_train], dataset[N_train:], coef_zern[:N_train], coef_zern[N_train:]


    Z = 2.0
    NN = []
    models = []
    tests_im, tests_co = [], []
    for N in [25, 50, 75, 100, 150]:
        N_zern = N
        x = np.linspace(-1, 1, N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)

        rho, theta = rho[pupil], theta[pupil]
        zernike = zern.ZernikeNaive(mask=pupil)
        _phase = zernike(coef=np.zeros(N_zern), rho=rho / (1.*RHO_APER), theta=theta, normalize_noll=False,
                         mode='Jacobi', print_option='Silent')
        H_flat = zernike.model_matrix[:, 3:]
        H_matrix = zern.invert_model_matrix(H_flat, pupil)
        N_zern = H_matrix.shape[-1]
        NN.append(N_zern)

        PSF_zernike = PointSpreadFunctionFast([H_matrix, pupil, H_flat])
        ZZ = Z / np.sqrt(np.sqrt(N_zern))

        rand_zern = ZZ * np.random.uniform(low=-1, high=1, size=(N_zern))
        im, s = PSF_zernike.compute_PSF(rand_zern)

        plt.figure()
        plt.imshow(im)
        plt.colorbar()
        plt.title(N_zern)

        ###
        N_train, N_test = 15000, 500

        train_PSF, test_PSF, train_zern, test_zern = training_zern(PSF_zernike, N_train, N_test)

        ### ZERNIKE model
        zern_model = create_model("ZERNIKE", N_output=N_zern)

        train_history_zern = zern_model.fit(x=train_PSF, y=train_zern, validation_data=(test_PSF, test_zern),
                                            epochs=50, batch_size=32, shuffle=True, verbose=1)
        guess_zern = zern_model.predict(test_PSF)
        residual_zern = test_zern - guess_zern
        norm_zern = np.mean(norm(residual_zern, axis=-1)) / N_zern

        models.append(zern_model)
        tests_im.append(test_PSF)
        tests_co.append(test_zern)

    """ Some Fourier Thoughts on why the Actuators are a better model than Zernike """

    H_act = PSF_actuator.RBF_mat
    actu_rand = 1.5*np.random.uniform(-1, 1, size=(N_act))
    phi0_actu = np.dot(H_act, actu_rand)

    for k in np.arange(25, 44):
        one_actu = np.zeros(N_act)
        eps = 1e-3
        one_actu[k] = eps
        phi_actu = phi0_actu + np.dot(H_act, one_actu)

        P_actu0 = pupil * np.exp(1j * phi0_actu)
        P_actu = pupil * np.exp(1j * phi_actu)

        Fp_actu0 = fftshift(fft2(P_actu0))
        Fp_actu = fftshift(fft2(P_actu))

        Im_actu0 = np.real(Fp_actu0 * np.conj(Fp_actu0))
        peak_actu = Im_actu0.max()
        Im_actu0 /= peak_actu
        Im_actu = np.real(Fp_actu * np.conj(Fp_actu))
        Im_actu /= peak_actu

        delta_actu = 100*(Im_actu - Im_actu0)
        dmin = min(np.min(delta_actu), -np.max(delta_actu))

        p = 75
        d_actu = delta_actu[N_PIX//2 - p:N_PIX//2 + p, N_PIX//2 - p:N_PIX//2 + p]
        plt.figure()
        # plt.imshow(np.log10(np.abs(d_actu)), cmap='RdBu')
        plt.imshow(d_actu, cmap='RdBu')
        plt.clim(dmin, -dmin)
        plt.colorbar()
        plt.title('Actuator %d' % k)
        # plt.show()

        H_zern = PSF_zernike.RBF_mat
        zern_rand = 1.5*np.random.uniform(-1, 1, size=(N_zern))
        # phi0_zern = np.dot(H_zern, zern_rand)
        phi0_zern = phi0_actu
        one_zern = np.zeros(N_zern)
        eps = 1e-3
        one_zern[k] = eps
        phi_zern = phi0_zern + np.dot(H_zern, one_zern)

        P_zern0 = pupil * np.exp(1j * phi0_zern)
        P_zern = pupil * np.exp(1j * phi_zern)

        FP_zern0 = fftshift(fft2(P_zern0))
        FP_zern = fftshift(fft2(P_zern))

        Im_zern0 = np.real(FP_zern0 * np.conj(FP_zern0))
        peak_zern = Im_zern0.max()
        Im_zern0 /= peak_zern
        Im_zern = np.real(FP_zern * np.conj(FP_zern))
        Im_zern /= peak_zern

        delta_zern = 100*(Im_zern - Im_zern0)
        dmin = min(np.min(delta_zern), -np.max(delta_zern))

        d_zern = delta_zern[N_PIX//2 - p:N_PIX//2 + p, N_PIX//2 - p:N_PIX//2 + p]
        plt.figure()
        plt.imshow(d_zern, cmap='RdBu')
        # plt.imshow(np.log10(np.abs(d_zern)), cmap='RdBu')
        plt.clim(dmin, -dmin)
        plt.colorbar()
        plt.title('Zernike %d' % k)
    plt.show()

    ####

    Fpi = fftshift(fft2(pupil, norm='ortho'))

    F_e = fftshift(fft2(pupil * np.exp(1j * phi0_actu), norm='ortho'))

    f_actu = H_act[:,:, 2]
    f_zern = H_zern[:,:, 1]

    F_actu = fftshift(fft2(f_actu, norm='ortho'))
    F_zern = fftshift(fft2(f_zern, norm='ortho'))

    F_fe_actu = fftshift(fft2(f_actu * np.exp(1j * phi0_actu), norm='ortho'))
    F_fe_zern = fftshift(fft2(f_zern * np.exp(1j * phi0_actu), norm='ortho'))

    plt.figure()
    plt.imshow(np.abs(np.imag(F_fe_actu)), cmap='Reds')
    plt.colorbar()

    plt.figure()
    plt.imshow(np.abs(np.imag(F_fe_zern)), cmap='Reds')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.abs(F_e), cmap='Reds')
    plt.colorbar()

    plt.figure()
    plt.imshow(np.abs((F_actu)), cmap='Reds')
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs((F_zern)), cmap='Reds')
    plt.colorbar()
    plt.show()











#####

    ##

    #

    #
    #


    # Relative to each model
    mapp = 'RdBu'
    RMS_zern_rel, RMS_actu_rel = [], []
    for k in range(N_test):

        err_zern = np.dot(PSF_zernike2.RBF_mat, residual_zern[k])
        err_actu = np.dot(PSF_actuator.RBF_mat, residual_actu[k])

        rms_zern = np.std(err_zern[pupil])
        RMS_zern_rel.append(rms_zern)
        rms_actu = np.std(err_actu[pupil])
        RMS_actu_rel.append(rms_actu)

        if k < 10:
            # mapp = 'bwr'
            m1 = min(np.min(err_zern), -np.max(err_zern))
            m2 = min(np.min(err_actu), -np.max(err_actu))
            m = min(m1, m2)

            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1 = plt.subplot(1, 2, 1)
            img1 = ax1.imshow(err_zern, cmap=mapp, extent=extent)
            ax1.set_title(r'Zernike Residual [$\sigma=%.3f \lambda$]' %rms_zern)
            fix_axes(ax1)
            img1.set_clim(m, -m)
            plt.colorbar(img1, ax=ax1, orientation='horizontal')

            ax2 = plt.subplot(1, 2, 2)
            img2 = ax2.imshow(err_actu, cmap=mapp, extent=extent)
            ax2.set_title('Actuator Residual [$\sigma=%.3f \lambda$]' %rms_actu)
            fix_axes(ax2)
            img2.set_clim(m, -m)
            plt.colorbar(img2, ax=ax2, orientation='horizontal')

    plt.figure()
    # plt.hist(RMS0, bins=20, histtype='step', label='Initial')
    plt.hist(RMS_zern_rel, bins=20, histtype='step', label='Zernike Model', color='coral')
    med_zern = np.median(RMS_zern_rel)
    plt.axvline(med_zern, linestyle='--', color='coral', label=r'Median = %.3f $\lambda$' % med_zern)
    plt.hist(RMS_actu_rel, bins=20, histtype='step', label='Actuator Model')
    med_actu = np.median(RMS_actu_rel)
    plt.axvline(med_actu, linestyle='--', label=r'Median = %.3f $\lambda$' % med_actu)
    plt.xlim([0, 0.50])
    plt.xlabel(r'RMS wavefront [$\lambda$]')
    plt.legend()

    # In absolute terms
    RMS0, RMS_zern, RMS_actu = [], [], []

    for k in range(N_test):

        true_wave = np.dot(PSF_actuator.RBF_mat, test_actu[k])
        rms0 = np.std(true_wave[pupil])
        RMS0.append(rms0)
        corr_zern = np.dot(PSF_zernike2.RBF_mat, guess_zern[k])
        corr_actu = np.dot(PSF_actuator.RBF_mat, guess_actu[k])

        res_zern = true_wave - corr_zern
        rms_zern = np.std(res_zern[pupil])
        RMS_zern.append(rms_zern)
        res_actu = true_wave - corr_actu
        rms_actu = np.std(res_actu[pupil])
        RMS_actu.append(rms_actu)

        impr_zern = 1 - rms_zern / rms0
        impr_actu = 1 - rms_actu / rms0

        if k < 10:
            mapp = 'RdBu'
            m = min(np.min(true_wave), -np.max(true_wave))
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1 = plt.subplot(1, 3, 1)
            img1 = ax1.imshow(true_wave, cmap=mapp, extent=extent)
            fix_axes(ax1)
            ax1.set_title(r'True Wavefront ($\sigma=%.3f \lambda$)' %rms0)
            img1.set_clim(m, -m)
            plt.colorbar(img1, ax=ax1, orientation='horizontal')

            ax2 = plt.subplot(1, 3, 2)
            img2 = ax2.imshow(res_zern, cmap=mapp, extent=extent)
            fix_axes(ax2)
            ax2.set_title('Residual ML [Zernike Model] ($\sigma=%.3f \lambda$)' %rms_zern)
            img2.set_clim(m, -m)
            plt.colorbar(img2, ax=ax2, orientation='horizontal')

            ax3 = plt.subplot(1, 3, 3)
            img3 = ax3.imshow(res_actu, cmap=mapp, extent=extent)
            fix_axes(ax3)
            ax3.set_title('Residual ML [Actuator Model] ($\sigma=%.3f \lambda$)' %rms_actu)
            # m = min(np.min(residual), -np.max(residual))
            img3.set_clim(m, -m)
            plt.colorbar(img3, ax=ax3, orientation='horizontal')
    plt.show()

    plt.figure()
    plt.hist(RMS0, bins=20, histtype='step', label='Initial')
    med_zern = np.median(RMS_zern)
    plt.axvline(med_zern, linestyle='--', color='red', label=r'Median = %.3f $\lambda$' % med_zern)
    plt.hist(RMS_zern, bins=20, histtype='step', color='red', label='Zernike Model')

    med_actu = np.median(RMS_actu)
    plt.axvline(med_actu, linestyle='--', color='green', label=r'Median = %.3f $\lambda$' % med_actu)
    plt.hist(RMS_actu, bins=20, histtype='step', color='green',label='Actuator Model')
    plt.legend()
    plt.xlabel(r'RMS wavefront [$\lambda$]')
    plt.show()







