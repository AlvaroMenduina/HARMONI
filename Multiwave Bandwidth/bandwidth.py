"""
So far we have worked only in monochromatic regime
Or Multiwavelength with various channels

But how does a finite bandwidth affect the calibration?
What if we want to add several channels together to improve SNR?

The bandwidth smears out the PSF, probably that will affect the predictions
"""

import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from numpy.linalg import norm as norm

# PARAMETERS
Z = 1.75                    # Strength of the aberrations
pix = 50                    # Pixels to crop the PSF
N_PIX = 128

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

BANDWIDTH = 50 * 1e-3      # [nm] -> [microns]
WAVE_MIN = WAVE - BANDWIDTH / 2.0
WAVE_MAX = WAVE + BANDWIDTH / 2.0
N_WAVES = 5


" Actuators "

def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
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


def rbf_matrix(centres, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
               N_waves=N_WAVES, wave0=WAVE_MIN, waveN=WAVE_MAX):

    waves = np.linspace(wave0, waveN, N_waves, endpoint=True)
    waves_ratio = waves / WAVE
    alpha = 1/np.sqrt(np.log(100/30))
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


class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    N_pix = N_PIX             # Number of pixels for the FFT computations
    minPix, maxPix = (N_pix + 1 - pix) // 2, (N_pix + 1 + pix) // 2

    def __init__(self, RBF_matrices, N_waves=N_WAVES, wave0=WAVE_MIN, waveN=WAVE_MAX):

        self.N_act = RBF_matrices[0][0].shape[-1]
        self.RBF_mat = [r[0] for r in RBF_matrices]
        self.pupil_masks = [r[1] for r in RBF_matrices]
        self.RBF_flat = [r[2] for r in RBF_matrices]
        self.waves_ratio = np.linspace(1., waveN / wave0, N_waves, endpoint=True)

        self.PEAKS = self.peak_PSF(N_waves)

    def peak_PSF(self, N_waves):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """
        PEAKS = []
        for k in range(N_waves):
            im, strehl = self.compute_PSF(np.zeros(self.N_act), wave_idx=k)
            PEAKS.append(strehl)
        return PEAKS


    def compute_PSF(self, coef, wave_idx=0, crop=True):
        """
        Compute the PSF and the Strehl ratio
        """

        phase = np.dot(self.RBF_mat[wave_idx], coef/self.waves_ratio[wave_idx])
        # plt.figure()
        # plt.imshow(phase)
        # plt.colorbar()

        pupil_function = self.pupil_masks[wave_idx] * np.exp(1j * 2 * np.pi * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2

        try:
            image /= self.PEAKS[wave_idx]

        except AttributeError:
            # If self.PEAK is not defined, self.compute_PSF will compute the peak
            pass

        strehl = np.max(image)

        if crop:
            image = image[self.minPix:self.maxPix, self.minPix:self.maxPix]
        else:
            pass
        return image, strehl

    def broadband_PSF(self, coef, crop=True):
        """
        Compute the broadband PSF by combining all wavelength channels
        :param coef:
        :param crop:
        :return:
        """
        N_BANDS = self.waves_ratio.shape[0]
        broad = np.zeros((N_BANDS, pix, pix))
        for wave_idx in range(N_BANDS):
            _im, _s = self.compute_PSF(coef, wave_idx=wave_idx, crop=crop)
            broad[wave_idx] = _im
        return np.mean(broad, axis=0)

    def plot_PSF(self, coef, wave_idx, cmap='hot'):
        """
        Plot an image of the PSF
        """
        PSF, strehl = self.compute_PSF(coef, wave_idx)

        plt.figure()
        plt.imshow(PSF, cmap=cmap)
        plt.title('Strehl: %.3f' % strehl)
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)


def generate_training_set(PSF_model, N_train=1500, N_test=500, foc=1.0, Z=1.0):

    N_act = PSF_model.N_act
    N_samples = N_train + N_test
    coef = Z * np.random.uniform(low=-1, high=1, size=(N_samples, N_act))

    # defocus = np.random.uniform(low=-1.25, high=1.25, size=N_act)
    # np.save('defocus', defocus)
    defocus = foc * np.load('defocus.npy')
    dataset = np.empty((N_samples, pix, pix, 2*N_WAVES))

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
        for wave_idx in range(N_WAVES):
            # Nominal image
            im0, _s = PSF_model.compute_PSF(coef[i], wave_idx=wave_idx)
            dataset[i, :, :, wave_idx] = im0

            # Defocused image
            im_foc, _s = PSF_model.compute_PSF(coef[i] + defocus, wave_idx=wave_idx)
            dataset[i, :, :, wave_idx + N_WAVES] = im_foc

        if i % 50 == 0:
            print(i)

    return dataset[:N_train], dataset[N_train:], coef[:N_train], coef[N_train:]

if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    """ Generate the Actuator Model """
    # The pupil is rescaled at each wavelength to account for PSF growth

    N_actuators = 20
    centers = actuator_centres(N_actuators, radial=True)
    N_act = len(centers[0][0])

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
        for c in centers[i][0]:
            ax.scatter(c[0], c[1], color='red', s=10)
            ax.scatter(c[0], c[1], color='black', s=10)
        ax.set_aspect('equal')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # plt.title('%d actuators' %N_act)
        plt.title('%.3f Wave' %wave_r)
    plt.show()

    # Define the actuator model matrix
    rbf_matrices = rbf_matrix(centers)

    PSF = PointSpreadFunction(rbf_matrices)

    c_act = 1./(2*np.pi) * np.random.uniform(-1, 1, size=N_act)

    # for idx in range(N_WAVES):
    #     PSF.plot_PSF(c_act, wave_idx=idx)
    # plt.show()

    phase0 = np.dot(rbf_matrices[0][0], c_act)
    p0 = min(np.min(phase0), -np.max(phase0))

    plt.figure()
    plt.imshow(phase0, extent=(-1, 1, -1, 1), cmap='bwr')
    plt.colorbar()
    plt.clim(p0, -p0)
    for c in centers[0][0]:
        plt.scatter(c[0], c[1], color='black', s=4)
    plt.xlim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.ylim([-1.1*RHO_APER, 1.1*RHO_APER])
    plt.title(r'%d Actuators | Wavefront [$\lambda$]' % N_act)
    plt.show()

    """ Monochromatic to Broadband comparison """
    # Compare the PSF at a single wavelength
    # and the broadband from averaging across channels

    psf_onewave, _s = PSF.compute_PSF(c_act, wave_idx=0)
    psf_broad = PSF.broadband_PSF(c_act)
    psf_diff = psf_onewave - psf_broad

    cmin = min(np.min(psf_diff), -np.max(psf_diff))

    cmap = 'hot'
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = plt.subplot(1, 3, 1)
    img1 = ax1.imshow(psf_onewave, cmap=cmap)
    ax1.set_title(r'Monochromatic PSF [$\lambda=%.1f$ $\mu m$]' % WAVE)
    img1.set_clim(0, 1)
    plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 3, 2)
    img2 = ax2.imshow(psf_broad, cmap=cmap)
    ax2.set_title(r'Broadband PSF [Bandwidth $%.1f$ nm]' % (1e3*BANDWIDTH))
    img2.set_clim(0, 1)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')

    ax3 = plt.subplot(1, 3, 3)
    img3 = ax3.imshow(psf_onewave - psf_broad, cmap='seismic')
    ax3.set_title(r'Difference')
    img3.set_clim(cmin, -cmin)
    plt.colorbar(img3, ax=ax3, orientation='horizontal')
    plt.show()

    ### ============================================================================================================ ###
    #                                   MACHINE LEARNING
    ### ============================================================================================================ ###

    def create_model_multiwave(waves, name):
        """
        Creates a CNN model for NCPA calibration
        :param waves: Number of wavelengths in the training set (to adjust the number of channels)
        :return:
        """
        input_shape = (pix, pix, 2 * waves,)        # Multiple Wavelength Channels
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

    def create_model_monochrom(name):
        """
        Creates a Monochromatic calibration model
        :param name:
        :return:
        """
        input_shape = (pix, pix, 2,)            # 2 Channels [nominal, defocus]
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

    coef_strength = 1.25 / (2*np.pi)
    foc = 1 / (2*np.pi)
    N_train, N_test = 10000, 1000
    training_PSF, test_PSF, training_coef, test_coef = generate_training_set(PSF, N_train, N_test, foc=foc, Z=coef_strength)



    np.save('training_PSF', training_PSF)
    np.save('test_PSF', test_PSF)
    np.save('training_coef', training_coef)
    np.save('test_coef', test_coef)

    # Select the first Wavelength Channel
    def select_monochromatic(PSF_datacube, wave_idx=0):
        """
        Extracts a single Wavelength channel out of the PSF datacube
        :param PSF_datacube: [N_PSF, pix, pix, 2*N_WAVES]
        :param wave_idx: which wavelength in [range(N_WAVES) to extract
        It returns a datacube with Nominal and Defocus channels
        """

        N_PSF = PSF_datacube.shape[0]
        mono_PSF = np.empty((N_PSF, pix, pix, 2))
        mono_PSF[:, :, :, 0] = PSF_datacube[:, :, :, wave_idx]
        mono_PSF[:, :, :, 1] = PSF_datacube[:, :, :, wave_idx + N_WAVES]

        return mono_PSF

    train_mono_PSF = select_monochromatic(training_PSF)
    test_mono_PSF = select_monochromatic(test_PSF)

    ### CNN Calibration Model
    calibration_model = create_model_monochrom("MONOCHROM")
    # Train the model
    train_history = calibration_model.fit(x=train_mono_PSF, y=training_coef, validation_data=(test_mono_PSF, test_coef),
                                          epochs=10, batch_size=32, shuffle=True, verbose=1)
    guess_coef_mono = calibration_model.predict(test_mono_PSF)
    residual_coef_mono = test_coef - guess_coef_mono
    norm_resi_mono = np.mean(norm(residual_coef_mono, axis=-1)) / N_act
    # print("Monochromatic Performance: ")
    # print("Norm Test Coef: %.4f" % (norm(test_coef)))
    # print("Norm Residual: %.4f" % (norm(residual_coef_mono)))

    # See how the performance changes when doing the Broadband PSF
    def average_broadband(PSF_datacube):
        """
        Average through the datacube to get a broadband PSF
        :param PSF_array:
        :return:
        """
        N_PSF = PSF_datacube.shape[0]
        broad_PSF = np.empty((N_PSF, pix, pix, 2))
        nom_PSF = PSF_datacube[:, :, :, :N_WAVES]
        foc_PSF = PSF_datacube[:, :, :, N_WAVES:]
        broad_PSF[:, :, :, 0] = np.mean(nom_PSF, axis=-1)
        broad_PSF[:, :, :, 1] = np.mean(foc_PSF, axis=-1)
        return broad_PSF


    def compare_mono_to_broad(PSF_mono, PSF_broad, i_PSF=0, nominal=True):

        if nominal:
            mono = PSF_mono[i_PSF, :, :, 0]
            broad = PSF_broad[i_PSF, :, :, 0]

        else:
            mono = PSF_mono[i_PSF, :, :, 1]
            broad = PSF_broad[i_PSF, :, :, 1]

        psf_diff = mono - broad

        cmin = min(np.min(psf_diff), -np.max(psf_diff))

        cmap = 'hot'
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = plt.subplot(1, 3, 1)
        img1 = ax1.imshow(mono, cmap=cmap)
        ax1.set_title(r'Monochromatic PSF')
        # img1.set_clim(0, 1)
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 3, 2)
        img2 = ax2.imshow(broad, cmap=cmap)
        ax2.set_title(r'Broadband PSF' )
        # img2.set_clim(0, 1)
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

        ax3 = plt.subplot(1, 3, 3)
        img3 = ax3.imshow(psf_diff, cmap='seismic')
        ax3.set_title(r'Difference')
        img3.set_clim(cmin, -cmin)
        plt.colorbar(img3, ax=ax3, orientation='horizontal')

        return

    test_broad_PSF = average_broadband(test_PSF)

    for i in range(5):
        compare_mono_to_broad(test_mono_PSF, test_broad_PSF, i_PSF=i)
    plt.show()


    guess_coef_broad = calibration_model.predict(test_broad_PSF)
    residual_coef_broad = test_coef - guess_coef_broad
    norm_resi_broad = np.mean(norm(residual_coef_broad, axis=-1)) / N_act
    print("Monochromatic Performance: ")
    print("Norm Test Coef: %.4f" % (norm(test_coef)))
    print("Norm Residual: %.4f" % (norm(residual_coef_mono)))
    print("Broadband Performance: ")
    print("Norm Residual: %.4f" % (norm(residual_coef_broad)))



    def strehl_ratios(PSF_model, test_PSF, test_coef, guess_coef_mono, guess_coef_broad):
        """
        Calculate the Strehl ratio before and after calibration
        """

        # Compute the Strehl ratio for the Test PSF at the nominal WAVE

        N_test = test_PSF.shape[0]
        strehl_before = np.max(test_PSF[:, :, :, N_WAVES//2], axis=(1, 2))

        residual_mono = test_coef - guess_coef_mono
        residual_broad = test_coef - guess_coef_broad

        strehl_mono, strehl_broad = [], []
        for k in range(N_test):
            if k % 25 == 0:
                print(k)
            _im, s_mono = PSF_model.compute_PSF(residual_mono[k], wave_idx=N_WAVES//2)
            strehl_mono.append(s_mono)
            im, s_broad = PSF_model.compute_PSF(residual_broad[k], wave_idx=N_WAVES//2)
            strehl_broad.append(s_broad)
        # return strehl_mono, strehl_broad

        plt.figure()
        plt.hist(strehl_before, bins=20, histtype='step', label='Initial')

        med_mono = np.median(strehl_mono)         # [nm]
        sigma_mono = np.std(strehl_mono)
        plt.axvline(med_mono, linestyle='--', color='red', label=r'Median = %.2f,  $\sigma$ = %.2f' % (med_mono, sigma_mono))
        plt.hist(strehl_mono, bins=20, histtype='step', color='red', label='Monochromatic')

        med_broad = np.median(strehl_broad)
        sigma_broad = np.std(strehl_broad)
        plt.axvline(med_broad, linestyle='--', color='green', label=r'Median = %.2f,  $\sigma$ = %.2f' % (med_broad, sigma_broad))
        plt.hist(strehl_broad, bins=20, histtype='step', color='green', label='Broadband')
        plt.legend()
        plt.xlabel(r'Strehl ratio [ ]')

        plt.xlim([0.25, 1.0])
        plt.ylim([0, 250])

    strehl_ratios(PSF, test_PSF, test_coef, guess_coef_mono, guess_coef_broad)
    plt.show()

    def RMS_wavefront(PSF_model, test_coef, guess_coef_mono, guess_coef_broad):

        RBF_mat = PSF_model.RBF_flat[N_WAVES//2]
        RMS0, RMS_mono, RMS_broad = [], [], []

        for k in range(N_test):
            if k % 25 == 0:
                print(k)
            wavefront0 = WAVE * 1e3 * np.dot(RBF_mat, test_coef[k])
            RMS0.append(np.std(wavefront0))
            correction_mono = WAVE * 1e3 * np.dot(RBF_mat, guess_coef_mono[k])
            residual_mono = wavefront0 - correction_mono
            RMS_mono.append(np.std(residual_mono))
            correction_broad = WAVE * 1e3 * np.dot(RBF_mat, guess_coef_broad[k])
            residual_broad = wavefront0 - correction_broad
            RMS_broad.append(np.std(residual_broad))

        plt.figure()
        plt.hist(RMS0, bins=20, histtype='step', label='Initial')

        med_mono = np.median(RMS_mono)         # [nm]
        sigma_mono = np.std(RMS_mono)
        plt.axvline(med_mono, linestyle='--', color='red', label=r'Median = %.2f,  $\sigma$ = %.2f' % (med_mono, sigma_mono))
        plt.hist(RMS_mono, bins=20, histtype='step', color='red', label='Monochromatic')

        med_broad = np.median(RMS_broad)
        sigma_broad = np.std(RMS_broad)
        plt.axvline(med_broad, linestyle='--', color='green', label=r'Median = %.2f,  $\sigma$ = %.2f' % (med_broad, sigma_broad))
        plt.hist(RMS_broad, bins=20, histtype='step', color='green', label='Broadband')
        plt.legend()
        plt.xlim([0, 250])
        plt.xlabel(r'RMS wavefront [nm]')


    RMS_wavefront(PSF, test_coef, guess_coef_mono, guess_coef_broad)
    plt.show()








