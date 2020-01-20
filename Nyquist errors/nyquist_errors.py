"""
What is the effect of errors in the Nyquist-Shannon sampling criterion

What happens to the performance when you show the model
PSF images that have a slightly different spaxel scale??

"""

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from numpy.linalg import norm as norm


# PARAMETERS
Z = 0.75                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 30                    # Pixels to crop the PSF
N_PIX = 512                 # Pixels for the Fourier arrays

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
#                                          Multiwave Equivalent
# ==================================================================================================================== #

# We use this to model sampling errors.
WAVE_MIN = WAVE
WAVE_MAX = WAVE
N_WAVES = 2

def actuator_centres_multiwave(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                               N_waves=N_WAVES, wave0=WAVE_MIN, waveN=WAVE_MAX, radial=True):
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

    waves = np.linspace(wave0, waveN, N_waves, endpoint=True)
    waves_ratio = waves / WAVE
    centres = []
    for wave in waves_ratio:
        x0 = np.linspace(-1./wave, 1./wave, N_actuators, endpoint=True)
        delta = x0[1] - x0[0]
        xx, yy = np.meshgrid(x0, x0)
        x_f = xx.flatten()
        y_f = yy.flatten()

        act = []    # List of actuator centres (Xc, Yc)
        for x_c, y_c in zip(x_f, y_f):
            r = np.sqrt(x_c ** 2 + y_c ** 2)
            if r < (rho_aper / wave - delta / 2) and r > (rho_obsc / wave + delta / 2):   # Leave some margin close to the boundary
                act.append([x_c, y_c])

        if radial:  # Add actuators at the boundaries, keeping a constant angular distance
            for r in [rho_aper / wave, rho_obsc / wave]:
                N_radial = int(np.floor(2*np.pi*r/delta))
                d_theta = 2*np.pi / N_radial
                theta = np.linspace(0, 2*np.pi - d_theta, N_radial)
                # Super important to do 2Pi - d_theta to avoid placing 2 actuators in the same spot... Degeneracy
                for t in theta:
                    act.append([r*np.cos(t), r*np.sin(t)])

        centres.append([act, delta])
    return centres


def rbf_matrix_multiwave(centres, alpha_pc=1, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                         N_waves=N_WAVES, wave0=WAVE_MIN, waveN=WAVE_MAX):

    waves = np.linspace(wave0, waveN, N_waves, endpoint=True)
    waves_ratio = waves / WAVE
    alpha = 1 / np.sqrt(np.log(100 / alpha_pc))
    # alpha = 1.

    matrices = [ ]
    for i, wave in enumerate(waves_ratio):

        cent, delta = centres[i]
        N_act = len(cent)
        matrix = np.empty((N_PIX, N_PIX, N_act))
        x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x0, x0)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        pupil = (rho <= rho_aper/wave) & (rho >= rho_obsc/wave)

        for k in range(N_act):
            xc, yc = cent[k][0], cent[k][1]
            r2 = (xx - xc) ** 2 + (yy - yc) ** 2
            matrix[:, :, k] = pupil * np.exp(-r2 / (alpha * delta) ** 2)

        mat_flat = matrix[pupil]
        matrices.append([matrix, pupil, mat_flat])

    return matrices


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
        pupil_function = self.pupil_mask * np.exp(1j * 2*np.pi * phase_datacube)

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

def generate_training_set(PSF_model, N_train=1500, N_test=500, foc=1.0, Z=1.0):

    N_act = PSF_model.N_act
    N_samples = N_train + N_test
    coef = Z * np.random.uniform(low=-1, high=1, size=(N_samples, N_act))

    # defocus = np.random.uniform(low=-1.25, high=1.25, size=N_act)
    # np.save('defocus', defocus)
    defocus = foc * np.load('defocus.npy')
    dataset = np.empty((N_samples, pix, pix, 2))

    # foc_phase = np.dot(PSF_model.RBF_mat[0], defocus)
    # std_foc = np.std(foc_phase[PSF_model.pupil_masks[0]])
    # cfoc = max(-np.min(foc_phase), np.max(foc_phase))
    #
    # plt.figure()
    # plt.imshow(foc_phase, cmap='bwr')
    # plt.clim(-cfoc, cfoc)
    # plt.title(r'RMS %.3f $\lambda$' % std_foc)
    # plt.colorbar()

    for i in range(N_samples):

        # Nominal image
        im0, _s = PSF_model.compute_PSF(coef[i])
        dataset[i, :, :, 0] = im0

        # Defocused image
        im_foc, _s = PSF_model.compute_PSF(coef[i] + defocus)
        dataset[i, :, :, 1] = im_foc

        if i % 50 == 0:
            print(i)

    return dataset[:N_train], dataset[N_train:], coef[:N_train], coef[N_train:]


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    ### ============================================================================================================ ###
    #                            Train the Calibration Model with 4 mas PSF images
    ### ============================================================================================================ ###

    """ (1) Define the ACTUATOR model for the WAVEFRONT """

    # WAVE: the nominal 4 mas spaxel scale
    SPAXEL_MAS = 4
    RHO_APER = rho_spaxel_scale(SPAXEL_MAS, WAVE)
    RHO_OBSC = 0.30 * RHO_APER  # Central obscuration

    # Find the Wavelength at which you have 4 mas, if you have 4 * (1 + eps) at 1.5
    SPAX_ERR = 0.075  # Percentage of error [10%]
    WAVE_BAD = WAVE * (1 + SPAX_ERR)
    # RHO_BAD = rho_spaxel_scale(SPAXEL_MAS, WAVE_BAD)
    check_spaxel_scale(RHO_APER, WAVE_BAD)

    WAVE_MIN = WAVE
    WAVE_MAX = WAVE_BAD

    centers_multiwave = actuator_centres_multiwave(N_actuators=20, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                               N_waves=N_WAVES, wave0=WAVE, waveN=WAVE_BAD, radial=True)
    N_act = len(centers_multiwave[0][0])

    waves = np.linspace(WAVE_MIN, WAVE_MAX, N_WAVES, endpoint=True)
    waves_ratio = waves / WAVE

    # Plot the actuators for each Wavelength
    for i, wave_r in enumerate(waves_ratio):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        circ1 = Circle((0, 0), RHO_APER/wave_r, linestyle='--', fill=None)
        circ2 = Circle((0, 0), RHO_OBSC/wave_r, linestyle='--', fill=None)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        for c in centers_multiwave[i][0]:
            ax.scatter(c[0], c[1], color='red', s=10)
            ax.scatter(c[0], c[1], color='black', s=10)
        ax.set_aspect('equal')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # plt.title('%d actuators' %N_act)
        plt.title('%.3f Wave' %wave_r)
    plt.show()

    alpha_pc = 30
    rbf_matrices = rbf_matrix_multiwave(centers_multiwave, alpha_pc=alpha_pc, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                         N_waves=N_WAVES, wave0=WAVE_MIN, waveN=WAVE_MAX)

    PSF_nom = PointSpreadFunctionFast(rbf_matrices[0])

    ### ============================================================================================================ ###
    #                            Machine Learning bits
    ### ============================================================================================================ ###


    def create_model(name):
        """
        Creates a CNN model for NCPA calibration
        :return:
        """
        input_shape = (pix, pix, 2,)        # Multiple Wavelength Channels
        model = Sequential()
        model.name = name
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(N_act))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    coef_strength = 1.5 / (2*np.pi)
    foc = 1 / (2*np.pi)
    N_train, N_test = 10000, 1000
    # N_train, N_test = 50, 50
    training_PSF, test_PSF, training_coef, test_coef = generate_training_set(PSF_nom, N_train, N_test, foc=foc, Z=coef_strength)

    np.save('training_PSF', training_PSF)
    np.save('test_PSF', test_PSF)
    np.save('training_coef', training_coef)
    np.save('test_coef', test_coef)

    training_PSF = np.load('training_PSF.npy')
    test_PSF = np.load('test_PSF.npy')
    training_coef = np.load('training_coef.npy')
    test_coef = np.load('test_coef.npy')

    ### CNN Calibration Model
    calibration_model = create_model("CALIBRATION")
    # Train the model
    train_history = calibration_model.fit(x=training_PSF, y=training_coef, validation_data=(test_PSF, test_coef),
                                          epochs=20, batch_size=32, shuffle=True, verbose=1)
    guess_coef = calibration_model.predict(test_PSF)
    residual_coef = test_coef - guess_coef

    """ Generate a set of PSF images with the wrong sampling """

    # Use the fake extra wavelength for which the sampling is off

    PSF_error = PointSpreadFunctionFast(rbf_matrices[1])
    test_PSF_error, _PSF, test_coef_error, _coef = generate_training_set(PSF_error, N_train=1000, N_test=1, foc=foc,
                                                                             Z=coef_strength)

    guess_coef_error = calibration_model.predict(test_PSF_error)
    residual_coef_error = test_coef_error - guess_coef_error

    print("\nPerformance Comparison")
    print("Norm Test Coef: %.4f" % (norm(test_coef)))
    print("Nominal Residual: %.4f" % (norm(residual_coef)))
    print("Nyquist Error Residual: %.4f" % (norm(residual_coef_error)))


    def RMS_wavefront(PSF_nominal, PSF_error, test_coef, guess_coef, test_coef_error, guess_coef_error):
        N_nom = test_coef.shape[0]
        N_err = test_coef_error.shape[0]

        RMS0_nom, RMS0_err = [], []
        RMS_nom, RMS_err = [], []

        for k in range(N_nom):

            wavefront0 = WAVE * 1e3 * np.dot(PSF_nominal.RBF_flat, test_coef[k])
            RMS0_nom.append(np.std(wavefront0))
            prediction = WAVE * 1e3 * np.dot(PSF_nominal.RBF_flat, guess_coef[k])
            error = wavefront0 - prediction
            RMS_nom.append(np.std(error))

        for k in range(N_err):
            wavefront0 = WAVE * 1e3 * np.dot(PSF_error.RBF_flat, test_coef_error[k])
            RMS0_err.append(np.std(wavefront0))
            prediction = WAVE * 1e3 * np.dot(PSF_error.RBF_flat, guess_coef_error[k])
            error = wavefront0 - prediction
            RMS_err.append(np.std(error))

        plt.figure()
        plt.scatter(RMS0_nom, RMS_nom, s=5, label='Nominal')
        plt.scatter(RMS0_err, RMS_err, s=5, label='Error')
        plt.legend(title='Sampling')

        return

    # RMS_wavefront(PSF_nom, PSF_error, test_coef, guess_coef, test_coef_error, guess_coef_error)

    ### ============================================================================================================ ###
    #                            Repeat the calibration one more iteration
    ### ============================================================================================================ ###


    def update_PSF(PSF_model, coef):

        N_samples = coef.shape[0]
        defocus = foc * np.load('defocus.npy')
        dataset = np.empty((N_samples, pix, pix, 2))

        for i in range(N_samples):

            # Nominal image
            im0, _s = PSF_model.compute_PSF(coef[i])
            dataset[i, :, :, 0] = im0

            # Defocused image
            im_foc, _s = PSF_model.compute_PSF(coef[i] + defocus)
            dataset[i, :, :, 1] = im_foc

            if i % 50 == 0:
                print(i)

        return dataset


    test_PSF2 = update_PSF(PSF_nom, residual_coef)
    test_PSF_error2 = update_PSF(PSF_error, residual_coef_error)


    guess_coef2 = calibration_model.predict(test_PSF2)
    residual_coef2 = residual_coef - guess_coef2

    guess_coef_error2 = calibration_model.predict(test_PSF_error2)
    residual_coef_error2 = residual_coef_error - guess_coef_error2

    """ Iteration 3 """
    test_PSF3 = update_PSF(PSF_nom, residual_coef2)
    test_PSF_error3 = update_PSF(PSF_error, residual_coef_error2)


    guess_coef3 = calibration_model.predict(test_PSF3)
    residual_coef3 = residual_coef2 - guess_coef3

    guess_coef_error3 = calibration_model.predict(test_PSF_error3)
    residual_coef_error3 = residual_coef_error2 - guess_coef_error3

    from itertools import tee

    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    COEF = [test_coef, residual_coef, residual_coef2, residual_coef3]
    COEF_ERR = [test_coef_error, residual_coef_error, residual_coef_error2, residual_coef_error3]

    import matplotlib.cm as cm
    blues = cm.Blues(np.linspace(0.5, 1.0, len(COEF) - 1))
    reds = cm.Reds(np.linspace(0.5, 1.0, len(COEF_ERR) - 1))

    plt.figure()
    for _coef, _model, _color in zip([COEF, COEF_ERR], [PSF_nom, PSF_error], [blues, reds]):

        print("Model: ")
        for i, (before, after) in enumerate(pairwise(_coef)):
            N_samples = before.shape[0]
            RMS_before, RMS_after = [], []

            for k in range(N_samples):
                wavefront0 = WAVE * 1e3 * np.dot(_model.RBF_flat, before[k])
                RMS_before.append(np.std(wavefront0))
                wavefront = WAVE * 1e3 * np.dot(_model.RBF_flat, after[k])
                RMS_after.append(np.std(wavefront))

            plt.scatter(RMS_before, RMS_after, s=4, color=_color[i])


    plt.xlim([0, 300])
    plt.ylim([0, 200])
    plt.grid(True)
    plt.xlabel(r'RMS wavefront BEFORE [nm]')
    plt.ylabel(r'RMS wavefront AFTER [nm]')
    # plt.show()

    ## Comparison Nominal VS Error, after 3 iterations.
    RMS_nom, RMS_err = [], []
    for k in range(N_test):
        wavefront_nom = WAVE * 1e3 * np.dot(PSF_nom.RBF_flat, residual_coef3[k])
        RMS_nom.append(np.std(wavefront_nom))
        wavefront_ee = WAVE * 1e3 * np.dot(PSF_error.RBF_flat, residual_coef_error3[k])
        RMS_err.append(np.std(wavefront_ee))

    plt.figure()
    plt.hist(RMS_nom, bins=20, histtype='step')
    plt.hist(RMS_err, bins=20, histtype='step')
    # plt.show()

    print(np.mean(RMS_nom), np.std(RMS_nom))
    print(np.mean(RMS_err), np.std(RMS_err))


    #### Error as a function of % SPAXEL ERROR
    pc_err = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]   # Watch Out: + means coarser
    mean_nom = [44.54]
    std_nom = [3.81]
    mean_err = [153.25, 108.59, 66.93, 46.54, 44.54, 55.62, 79.32, 132.47, 198.8]
    std_err = [16.26, 11.87, 6.44, 4.32, 3.81, 4.39, 5.80, 11.81, 21.24]

    plt.figure()

    plt.plot(pc_err, mean_err, color='black')
    plt.errorbar(pc_err, mean_err, yerr=std_err, fmt='o')
    plt.xlabel(r'Error in sampling [per cent]')
    plt.ylabel(r'Mean RMS wavefront after calibration [nm]')
    plt.grid(True)
    # plt.ylim([0, 250])
    plt.show()


    """ How do the Actuator Command errors look like for each case? """
    # Is there a "typically" residual?

    def draw_actuator_commands(commands, centers):
        """
        Plot of each actuator commands
        :param commands:
        :param centers:
        :return:
        """
        cent, delta = centers
        x = np.linspace(-1, 1, 2 * N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        image = np.zeros((2 * N_PIX, 2 * N_PIX))
        for i, (xc, yc) in enumerate(cent):
            act_mask = (xx - xc) ** 2 + (yy - yc) ** 2 <= (delta / 3) ** 2
            image += commands[i] * act_mask

        return image

    mean_command = WAVE * 1e3 * np.mean((residual_coef), axis=0)
    commands_nom = draw_actuator_commands(mean_command, centers=centers_multiwave[0])

    mean_command_err = WAVE * 1e3 * np.mean(np.abs(guess_coef_error), axis=0)
    commands_nom_err = draw_actuator_commands(mean_command_err, centers=centers_multiwave[0])

    cmin = np.min(commands_nom[np.nonzero(commands_nom)])
    cmax = np.max(commands_nom)

    plt.figure()
    plt.imshow(commands_nom, cmap='Reds')
    plt.clim(cmin, cmax)
    plt.colorbar()

    cmin = np.min(commands_nom_err[np.nonzero(commands_nom_err)])
    cmax = np.max(commands_nom_err)

    plt.figure()
    plt.imshow(commands_nom_err, cmap='Reds')
    plt.colorbar()
    plt.clim(cmin, cmax)
    plt.show()

    mean_wavefront = np.dot(PSF_error.RBF_mat, mean_command_err)
    plt.imshow(mean_wavefront, cmap='seismic')
    p0, s0 = PSF_error.compute_PSF(0*mean_command_err)
    p, s = PSF_error.compute_PSF(5*mean_command_err / (1e3))


    plt.figure()
    for i in range(N_act):
        _x = N_test * [i]

        plt.scatter(_x, residual_coef_error3[:, i], s=3, color='red')
        plt.scatter(_x, residual_coef3[:, i], s=3, color='black')
    plt.show()

    wavefronts = []
    for k in range(50):
        wavefront = WAVE * 1e3 * np.dot(PSF_error.RBF_mat, residual_coef3[k])
        wavefronts.append(wavefront)
        if k < 10:
            plt.figure()
            plt.imshow(wavefront)
            plt.colorbar()
    wavefronts = np.array(wavefronts)
    wavefronts = np.mean(wavefronts, axis=0)


    plt.figure()
    plt.imshow(training_PSF[0,:,:,0])
    plt.colorbar()

    cmap = 'hot'
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow((training_PSF[0,:,:,0]), cmap=cmap)
    ax1.set_title(r'Nominal Sampling')
    # img1.set_clim(0, 1)
    plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow((test_PSF_error[0,:,:,0]), cmap=cmap)
    ax2.set_title(r'Error Sampling')
    # img2.set_clim(0, 1)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')

    plt.show()

    # Histograms of norms
    norm_test_coef = norm(test_coef, axis=1)
    norm_test_coef_error = norm(test_coef_error, axis=1)
    norm_residual_coef = norm(residual_coef, axis=1)
    norm_residual_coef_error = norm(residual_coef_error, axis=1)

    plt.figure()
    plt.scatter(norm_test_coef, norm_residual_coef, s=4, label='Nominal')
    plt.scatter(norm_test_coef_error, norm_residual_coef_error, s=4, label='Error')
    plt.title('Sampling')
    plt.show()









