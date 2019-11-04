


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
Z = 1.25                    # Strength of the aberrations
pix = 25                    # Pixels to crop the PSF
N_PIX = 256
RHO_APER = 0.5
RHO_OBSC = 0.15
N_WAVES = 5
WAVE_N = 1.6

" Actuators "

def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC,
                     N_waves=N_WAVES, wave0=1.0, waveN=WAVE_N):
    """
    Computes the (Xc, Yc) coordinates of actuator centres
    inside a circle of rho_aper, assuming there are N_actuators
    along the [-1, 1] line

    :param N_actuators:
    :param rho_aper:
    :return:
    """
    waves_ratio = np.linspace(1., waveN/wave0, N_waves, endpoint=True)

    centres = []
    for wave in waves_ratio:
        x0 = np.linspace(-1./wave, 1./wave, N_actuators, endpoint=True)
        delta = x0[1] - x0[0]
        xx, yy = np.meshgrid(x0, x0)
        x_f = xx.flatten()
        y_f = yy.flatten()

        act = []
        for x_c, y_c in zip(x_f, y_f):
            r = np.sqrt(x_c ** 2 + y_c ** 2)
            if r < 0.95 * rho_aper / wave and r > 1.1 * rho_obsc /wave:
                act.append([x_c, y_c])

        total_act = len(act)
        print('Total Actuators: ', total_act)
        centres.append([act, delta])
    return centres

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

    wave0 = 1.0
    waves_ratio = np.linspace(1., WAVE_N / wave0, N_WAVES, endpoint=True)

    N_actuators = 25
    centers = actuator_centres(N_actuators)
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
    plt.show()

    c_act = np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(rbf_matrices[0][0], c_act)
    phaseN = np.dot(rbf_matrices[1][0], c_act)

    plt.figure()
    plt.imshow(phase0)
    plt.colorbar()

    plt.figure()
    plt.imshow(phaseN)
    plt.colorbar()
    plt.show()

    PSF = PointSpreadFunction(rbf_matrices)

    PSF.plot_PSF(c_act, wave_idx=0)
    PSF.plot_PSF(c_act, wave_idx=1)
    PSF.plot_PSF(c_act, wave_idx=2)

    plt.show()

    # ------------------------------------------------------------------------------- #

    N_train, N_test = 15000, 100
    training_PSF, test_PSF, training_coef, test_coef = generate_training_set(PSF, N_train, N_test)

    # plt.figure()
    # plt.imshow(training_PSF[0,:,:,4], cmap='hot')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(training_PSF[0,:,:,5], cmap='hot')
    # plt.colorbar()
    #
    # plt.show()

    from keras.regularizers import l2

    N_channels = 2*N_WAVES
    input_shape = (pix, pix, N_channels,)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(N_act))
    # model.add(Dense(N_act, kernel_regularizer=l2(0.01)))
    model.summary()


    model.compile(optimizer='adam', loss='mean_squared_error')
    train_history = model.fit(x=training_PSF, y=training_coef,
                              validation_data=(test_PSF, test_coef),
                              epochs=250, batch_size=32, shuffle=True, verbose=1)
    loss_hist = train_history.history['loss']
    val_hist = train_history.history['val_loss']

    guess = model.predict(test_PSF)
    residual = test_coef - guess


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
