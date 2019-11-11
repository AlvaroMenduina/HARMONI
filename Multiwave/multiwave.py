


import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from numpy.linalg import norm as norm

# PARAMETERS
Z = 1.5                    # Strength of the aberrations
pix = 30                    # Pixels to crop the PSF
N_PIX = 256
RHO_APER = 0.5
RHO_OBSC = 0.15
N_WAVES = 15
WAVE_N = 2.0

" Actuators "

def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                     N_waves=N_WAVES, wave0=1.0, waveN=WAVE_N, radial=True):
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

    waves_ratio = np.linspace(1., waveN / wave0, N_waves, endpoint=True)
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
            if r < (rho_aper / wave - delta/2) and r > (rho_obsc / wave + delta/2):   # Leave some margin close to the boundary
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
               N_waves=N_WAVES, wave0=1.0, waveN=WAVE_N):

    waves_ratio = np.linspace(1., waveN / wave0, N_waves, endpoint=True)

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
            matrix[:, :, k] = pupil * np.exp(-r2 / (1. * delta) ** 2)

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

    def __init__(self, RBF_matrices, N_waves=N_WAVES, wave0=1.0, waveN=WAVE_N):

        self.N_act = RBF_matrices[0][0].shape[-1]
        self.RBF_mat = [r[0] for r in RBF_matrices]
        self.pupil_masks = [r[1] for r in RBF_matrices]
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

        pupil_function = self.pupil_masks[wave_idx] * np.exp(1j * phase)
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

    def plot_PSF(self, coef, wave_idx):
        """
        Plot an image of the PSF
        """
        PSF, strehl = self.compute_PSF(coef, wave_idx)

        plt.figure()
        plt.imshow(PSF)
        plt.title('Strehl: %.3f' %strehl)
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

def generate_training_set(PSF_model, N_train=1500, N_test=500):

    N_act = PSF_model.N_act
    N_samples = N_train + N_test
    coef = np.random.uniform(low=-Z, high=Z, size=(N_samples, N_act))

    # defocus = np.zeros(N_zern)
    # defocus[1] = focus
    # FIXME! Watch out when using the ACTUATOR MODE
    defocus = np.random.uniform(low=-1, high=1, size=N_act)
    dataset = np.empty((N_samples, pix, pix, 2*N_WAVES))

    for i in range(N_samples):
        for wave_idx in range(N_WAVES):
            # Nominal image
            im0, _s = PSF_model.compute_PSF(coef[i], wave_idx=wave_idx)
            dataset[i, :, :, 2*wave_idx] = im0

            # Defocused image
            im_foc, _s = PSF_model.compute_PSF(coef[i] + defocus, wave_idx=wave_idx)
            dataset[i, :, :, 2*wave_idx+1] = im_foc

        if i%100 == 0:
            print(i)

    return dataset[:N_train], dataset[N_train:], coef[:N_train], coef[N_train:]

# def flat_field_training(N_flats=5, sigma=0.15):

if __name__ == "__main__":


    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    wave0 = 1.0
    waves_ratio = np.linspace(1., WAVE_N / wave0, N_WAVES, endpoint=True)

    N_actuators = 20
    centers = actuator_centres(N_actuators, radial=False)
    N_act = len(centers[0][0])

    for i, wave_r in enumerate(waves_ratio):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        circ1 = Circle((0,0), 0.5/wave_r, linestyle='--', fill=None)
        circ2 = Circle((0,0), 0.15/wave_r, linestyle='--', fill=None)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        for c in centers[i][0]:
            ax.scatter(c[0], c[1], color='red', s=10)
            ax.scatter(c[0], c[1], color='black', s=10)
        ax.set_aspect('equal')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # plt.title('%d actuators' %N_act)
        plt.title('%.2f Wave' %wave_r)
    plt.show()
    #
    rbf_matrices = rbf_matrix(centers)

    plt.figure()
    plt.imshow(rbf_matrices[0][0][:,:,0])

    plt.figure()
    plt.imshow(rbf_matrices[1][0][:,:,0])
    # plt.show()

    c_act = Z*np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(rbf_matrices[0][0], c_act)
    phaseN = np.dot(rbf_matrices[1][0], c_act)

    plt.figure()
    plt.imshow(phase0)
    plt.colorbar()

    plt.figure()
    plt.imshow(phaseN)
    plt.colorbar()
    # plt.show()

    PSF = PointSpreadFunction(rbf_matrices)

    for idx in range(N_WAVES):
        PSF.plot_PSF(c_act, wave_idx=idx)

    plt.show()

    # ================================================================================================================ #
    """ Could the Multiwavelength information be useful for calibration? """
    ### The Answer is OF COURSE NOT
    coeff_degen = np.zeros(N_act)
    k_act = 25
    coeff_degen[k_act] = 15
    coeff_degen[-k_act] = 15
    phase_degen = np.dot(rbf_matrices[0][0], coeff_degen)

    plt.figure()
    plt.imshow(phase_degen)
    plt.colorbar()
    plt.show()
    for wave_idx in range(N_WAVES):
        p_plus, _s = PSF.compute_PSF(coeff_degen, wave_idx)
        coeff_degen[-1] *= -1
        p_minus, _s = PSF.compute_PSF(coeff_degen, wave_idx)

        mapp = 'viridis'

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = plt.subplot(1, 3, 1)
        img1 = ax1.imshow(p_plus, cmap=mapp)
        ax1.set_title('Plus')
        # img1.set_clim(m, -m)
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 3, 2)
        img2 = ax2.imshow(p_minus, cmap=mapp)
        ax2.set_title('Minus')
        # img2.set_clim(m, -m)
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

        ax3 = plt.subplot(1, 3, 3)
        img3 = ax3.imshow(p_plus - p_minus, cmap=mapp)
        ax3.set_title('Diff')
        # img3.set_clim(m, -m)
        plt.colorbar(img3, ax=ax3, orientation='horizontal')
    plt.show()

    # ------------------------------------------------------------------------------- #

    N_train, N_test = 5000, 500
    training_PSF, test_PSF, training_coef, test_coef = generate_training_set(PSF, N_train, N_test)

    ### Does it saturate because of the defocus range?
    # We put a defocus in [nm] that affects the PSF at each wavelength different
    # For very long wavelengths, the defocus makes almost no difference in intensity
    # Could it be that at such point we do not gain from "diversity" but from error statistics?
    # more samples of the readout noise...

    k_train = 0
    plot_waves = N_WAVES
    f, axes = plt.subplots(plot_waves, 2)
    for i in range(plot_waves):
        ax = plt.subplot(plot_waves, 2, 2*i + 1)
        PSF_nom = training_PSF[k_train, :,:,2*i]
        img = ax.imshow(PSF_nom)
        ax.set_title('Nominal PSF [Wave %.2f]' %waves_ratio[i])

        ax = plt.subplot(plot_waves, 2, 2*i + 2)
        PSF_nom = training_PSF[k_train, :,:,2*i+1]
        img = ax.imshow(PSF_nom)
        ax.set_title('Defocus PSF [Wave %.2f]' %waves_ratio[i])
    plt.show()

    waves_considered=1
    N_channels = 2*waves_considered
    input_shape = (pix, pix, N_channels,)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(N_act))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    train_history = model.fit(x=training_PSF[:,:,:,:2*waves_considered], y=training_coef,
                              validation_data=(test_PSF[:,:,:,:2*waves_considered], test_coef),
                              epochs=250, batch_size=32, shuffle=True, verbose=1)
    loss_hist = train_history.history['loss']
    val_hist = train_history.history['val_loss']

    guess = model.predict(test_PSF[:,:,:,:2*waves_considered])
    residual = test_coef - guess

    def test_models(PSF_model, training_PSF, test_PSF, training_coef, test_coef):

        """
        Check the influence of the number of Wavelength channels
        on the performance of the algorithm

        :param training_PSF:
        :param test_PSF:
        :param training_coef:
        :param test_coef:
        :return:
        """
        N_test = test_PSF.shape[0]
        N_waves = training_PSF.shape[-1] // 2
        list_waves = list(np.arange(1, N_waves + 1))

        # Evaluate the Initial RMS and Wavefronts
        rbf_mat = PSF_model.RBF_mat[0]
        pupil_mask = PSF_model.pupil_masks[0]
        rms0, rms = [], []
        wavefr0, wavefronts = [], []
        for k in range(N_test):
            phase0 = np.dot(rbf_mat, test_coef[k])
            wavefr0.append(phase0)
            phase0_f = phase0[pupil_mask]
            rms0.append(np.std(phase0_f))

        validation_losses = []
        guessed_coef = []

        for waves in list_waves:

            print("\nConsidering %d Wavelengths" %waves)

            N_channels = 2 * waves
            input_shape = (pix, pix, N_channels,)
            print(input_shape)
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(N_act))
            model.summary()
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Slice Channels
            train_wave = training_PSF[:, :, :, :N_channels]
            test_wave = test_PSF[:, :, :, :N_channels]

            train_history = model.fit(x=train_wave, y=training_coef,
                                      validation_data=(test_wave, test_coef),
                                      epochs=20, batch_size=32, shuffle=True, verbose=1)
            loss_hist = train_history.history['loss']
            val_hist = train_history.history['val_loss']
            validation_losses.append(val_hist)

            guess = model.predict(test_wave)
            guessed_coef.append(guess)
            residual = test_coef - guess


            # Check the RMS wavefront error for each case
            wavefr = []
            _rms = []
            for k in range(N_test):
                phase = np.dot(rbf_mat, residual[k])
                # if k < 50:
                #     wavefr.append(phase)
                phase_f = phase[pupil_mask]
                _rms.append(np.std(phase_f))
            wavefronts.append(wavefr)
            rms.append(_rms)

        plt.figure()
        for i in range(N_waves):
            plt.plot(validation_losses[i], label=list_waves[i])
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(title='Channels')

        return validation_losses, guessed_coef, rms0, rms, wavefr0, wavefronts

    validation_losses, guessed_coef, rms0, rms, wavefr0, wavefronts = test_models(PSF, training_PSF, test_PSF,
                                                                             training_coef, test_coef)

    list_waves = list(np.arange(1, N_WAVES + 1))

    plot_waves = N_WAVES
    # n_rows = plot_waves // 2
    n_columns = 4
    n_rows = 4
    f, axes = plt.subplots(n_rows, n_columns)
    mus = []
    rmses = []
    for i in range(n_rows):
        for j in range(n_columns):
            k = n_columns * i + j
            if k >= plot_waves:
                break
            rms_wave = rms[k]
            avg_rms = np.mean(rms_wave)
            med_rms = np.median(rms_wave)
            std_rms = np.std(rms_wave)
            mus.append(med_rms)
            rmses.append(std_rms)
            print(k)
            ax = plt.subplot(n_rows, n_columns, k + 1)
            ax.hist(rms0, histtype='step')
            ax.hist(rms_wave, histtype='step')
            ax.axvline(med_rms, linestyle='--', color='black')
            ax.set_ylim([0, 1250])
            ax.set_title(r'Channels %d | WFE %.3f ($\sigma$=%.3f)' % (list_waves[k], avg_rms, std_rms))
            ax.set_xlim([0, 1.25])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_rows - 1:
                ax.get_xaxis().set_visible(True)
                ax.set_xlabel(r'RMS Wavefront [$\lambda$]')
            if j == 0:
                ax.get_yaxis().set_visible(True)

    ax = plt.subplot(n_rows, n_columns, n_rows*n_columns)
    # ax.scatter(list_waves, medians)
    ax.errorbar(list_waves, mus, yerr=rmses, fmt='o')
    ax.set_xlabel('Waves considered')
    ax.set_ylabel(r'RMS Wavefront [$\lambda$]')
    ax.set_xticks(list_waves)
    plt.show()

    k_phase = 5
    plot_waves = N_WAVES
    mapp = 'bwr'
    f, axes = plt.subplots(1, plot_waves + 1)
    # Initial Wavefront
    phase0 = wavefr0[k_phase]
    pmin = min(np.min(phase0), -np.max(phase0))
    ax = plt.subplot(1, plot_waves + 1, 1)
    img = ax.imshow(phase0, cmap=mapp)
    img.set_clim(pmin, -pmin)
    ax.set_title(r'Initial Wavefront %.4f' % rms0[k_phase])
    plt.colorbar(img, ax=ax, orientation='horizontal')

    for k in range(plot_waves):     # Loop over the
        phase = wavefronts[k][k_phase]
        ax = plt.subplot(1, plot_waves + 1, k+2)
        img = ax.imshow(phase, cmap=mapp, label='A')
        img.set_clim(pmin, -pmin)
        ax.set_title(r'Residual %.4f [%d Waves]' % (rms[k][k_phase], list_waves[k]))
        plt.colorbar(img, ax=ax, orientation='horizontal')

    # plt.legend()
    plt.show()


    """ What if we include noise in the training? """

    def noise_images(dataset, coef, RMS_READ, N_copies=3):
        N_PSF, pix, _pix, N_chan = dataset.shape
        N_act = coef.shape[-1]
        new_data = np.empty((N_copies * N_PSF, pix, pix, N_chan))
        new_coef = np.empty((N_copies * N_PSF, N_act))

        for k in range(N_PSF):
            if k %100 == 0:
                print(k)
            PSF = dataset[k].copy()
            coef_copy = coef[k].copy()
            for i in range(N_copies):
                read_out = np.random.normal(loc=0, scale=RMS_READ, size=(pix, pix, N_chan))
                new_data[N_copies * k + i] = PSF + read_out
                new_coef[N_copies * k + i] = coef_copy

        return new_data, new_coef

    RMS_READ = 1./100
    N_copies = 6
    read_train_PSF, read_train_coef = noise_images(training_PSF, training_coef, RMS_READ, N_copies)
    read_test_PSF, read_test_coef = noise_images(test_PSF, test_coef, RMS_READ, N_copies)

    ### Check the performance on noisy data
    val_loss, guessed_coef, rms0, rms, wavefr0, wavefronts = test_models(PSF, read_train_PSF, read_test_PSF,
                                                                         read_train_coef, read_test_coef)

    """ Good Question """
    ### Is the improvement because of HAVING EXTRA IMAGES
    # or is it because we have one image at a higher sampling??

    # To test that, train only on the longest wavelength

    N_channels = 2
    input_shape = (pix, pix, N_channels,)
    print(input_shape)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(N_act))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Keep only the last Channels for the Longest Wavelength
    read_train_wave = read_train_PSF[:, :, :, -N_channels:]
    read_test_wave = read_test_PSF[:, :, :, -N_channels:]

    train_history = model.fit(x=read_train_wave, y=read_train_coef,
                              validation_data=(read_test_wave, read_test_coef),
                              epochs=20, batch_size=32, shuffle=True, verbose=1)

    guess_longwave = model.predict(read_test_wave)
    residual_longwave = read_test_coef - guess_longwave

    rms_longwave = []
    for k in range(residual_longwave.shape[0]):
        phase = np.dot(PSF.RBF_mat[0], residual_longwave[k])
        phase_f = phase[PSF.pupil_masks[0]]
        rms_longwave.append(np.std(phase_f))

    plt.figure()
    plt.hist(rms0, histtype='step', label='Initial')
    plt.hist(rms[-1], histtype='step', label='%d waves considered' % N_WAVES)
    plt.axvline(np.median(rms[-1]), color='orange', linestyle='--')
    std_5 = np.std(rms[-1])
    plt.hist(rms_longwave, histtype='step', label='1 wave [longest]')
    plt.axvline(np.median(rms_longwave), color='green', linestyle='--')
    std_longwave = np.std(rms_longwave)
    plt.legend()
    plt.xlim([0, 1.25])
    plt.xlabel(r'RMS wavefront [$\lambda$]')
    plt.show()





    def compute_RMS(test_images, true_coef, mask=PSF.pupil_masks[0]):

        rbf_mat = PSF.RBF_mat[0]
        M_test = test_images.shape[0]
        guess = model.predict(test_images)
        residual = true_coef - guess

        # Initial wavefront
        rms0, rms = [], []
        wavefr0, wavefr = [], []
        for k in range(M_test):
            phase0 = np.dot(rbf_mat, true_coef[k])
            wavefr0.append(phase0)
            phase0_f = phase0[mask]
            rms0.append(np.std(phase0))

            # Guessed wavefront
            guess0 = np.dot(rbf_mat, guess[k])
            wavefr.append(guess0)

            # Final wavefront
            phase = np.dot(rbf_mat, residual[k])
            phase_f = phase[mask]
            rms.append(np.std(phase))

        plt.figure()
        plt.scatter(np.arange(M_test), rms0, label='Before', color='red')
        plt.scatter(np.arange(M_test), rms, label='After', color='blue')
        plt.legend()
        plt.xlabel('Test PSF')
        plt.ylabel(r'RMS [$\lambda$]')
        plt.ylim([0, 1.1 * np.max(rms0)])


        for k in range(10):
            phase0, guess0 = wavefr0[k], wavefr[k]
            mins = min(phase0.min(), guess0.min())
            maxs = max(phase0.max(), guess0.max())

            m = min(mins, -maxs)
            mapp = 'bwr'
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1 = plt.subplot(1, 3, 1)
            img1 = ax1.imshow(phase0, cmap=mapp)
            ax1.set_title('True Wavefront ($\sigma=%.2f \lambda$)' %rms0[k])
            img1.set_clim(m, -m)
            plt.colorbar(img1, ax=ax1, orientation='horizontal')

            ax2 = plt.subplot(1, 3, 2)
            img2 = ax2.imshow(guess0, cmap=mapp)
            ax2.set_title('Guessed Wavefront')
            img2.set_clim(m, -m)
            plt.colorbar(img2, ax=ax2, orientation='horizontal')

            ax3 = plt.subplot(1, 3, 3)
            img3 = ax3.imshow(phase0 - guess0, cmap=mapp)
            ax3.set_title('Residual ($\sigma=%.2f \lambda$)' %rms[k])
            img3.set_clim(m, -m)
            plt.colorbar(img3, ax=ax3, orientation='horizontal')
            # plt.show()

    def performance(test_images, true_coef):

        " Evaluate the performance "
        guess = model.predict(test_images)
        residual = true_coef - guess

        norm0 = norm(true_coef, axis=1)
        mean_norm0 = np.mean(norm0)

        norm_res = norm(residual, axis=1)
        mean_norm_res = np.mean(norm_res)

        improv = (norm0 - norm_res) / norm0
        mean_improv = np.mean(improv)

        print('Average Improvement: %.1f [per cent]' % (100 * mean_improv))

        s0 = np.max(test_images[:,:,:,0], axis=(1,2))

        # Corrected PSF images
        sf = []
        for res_coef in residual:
            im, _sf = PSF.compute_PSF(res_coef)
            sf.append(_sf)

        plt.figure()
        plt.hist(s0, histtype='step', label='Before')
        plt.hist(sf, histtype='step', label='After')
        plt.xlabel('Strehl ratio')
        plt.legend()
        plt.show()

    performance(test_PSF, test_coef)
    compute_RMS(test_PSF, test_coef)
    plt.show()
