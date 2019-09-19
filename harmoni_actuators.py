"""
==========================================================
                      Multiwavelength
==========================================================

Experiments: see if adding Multiwavelength information improves

(1) The resilience against FLAT FIELDING errors

(2) The overall quality of the PREDICTIONS

** So far we will not use Zemax, to make everything quicker

"""

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import zern_core as zern
from scipy.optimize import least_squares

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
        matrix[:, :, k] = pupil * np.exp(-r2 / (1. * delta) ** 2)

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

class LS_fit(object):

    def __init__(self, mat_high, mat_low):
        self.high, self.low = mat_high, mat_low

    def residuals(self, x, truth):
        guess = np.dot(self.low, x)
        return truth - guess

    def __call__(self, coef_high, **kwargs):
        true_phase = np.dot(self.high, coef_high)
        x0 = np.zeros(self.low.shape[-1])
        result = least_squares(self.residuals, x0, args=(true_phase,))
        return result.x

class Zernike_fit(object):

    def __init__(self, PSF_mat, N_zern=100):

        self.PSF_mat = PSF_mat

        x = np.linspace(-1, 1, N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        self.pupil = (rho <= RHO_APER) & (rho > RHO_OBSC)

        rho, theta = rho[self.pupil], theta[self.pupil]
        zernike = zern.ZernikeNaive(mask=self.pupil)
        _phase = zernike(coef=np.zeros(N_zern), rho=rho / RHO_APER, theta=theta, normalize_noll=False,
                         mode='Jacobi', print_option='Silent')
        H_flat = zernike.model_matrix
        self.H_matrix = zern.invert_model_matrix(H_flat, self.pupil)
        self.N_zern = self.H_matrix.shape[-1]

    def residuals(self, x, wavefront):
        zernike_phase = np.dot(self.H_matrix, x)
        return (wavefront - zernike_phase)[self.pupil]

    def __call__(self, PSF_coef, **kwargs):
        wavefront = np.dot(self.PSF_mat, PSF_coef)
        x0 = np.zeros(self.N_zern)
        result = least_squares(self.residuals, x0, args=(wavefront,))

        x = result.x
        zernike_phase = np.dot(self.H_matrix, x)

        wf = wavefront[self.pupil]
        zf = zernike_phase[self.pupil]
        residual = wf - zf
        rms0 = np.std(wf)
        rmsz = np.std(zf)
        rmsr = np.std(residual)

        print("\nFit Wavefront to Zernike Polynomials: ")
        print("Initial RMS: %.3f" %rms0)
        print("Residual RMS after correction: %.3f" %rmsr)

        # mins = min(wavefront.min(), zernike_phase.min())
        # maxs = max(wavefront.max(), zernike_phase.max())
        #
        # m = min(mins, -maxs)
        # mapp = 'bwr'
        # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1 = plt.subplot(1, 3, 1)
        # img1 = ax1.imshow(wavefront, cmap=mapp)
        # ax1.set_title('True Wavefront')
        # img1.set_clim(m, -m)
        # plt.colorbar(img1, ax=ax1, orientation='horizontal')
        #
        # ax2 = plt.subplot(1, 3, 2)
        # img2 = ax2.imshow(zernike_phase, cmap=mapp)
        # ax2.set_title('Zernike Fit Wavefront')
        # img2.set_clim(m, -m)
        # plt.colorbar(img2, ax=ax2, orientation='horizontal')
        #
        # ax3 = plt.subplot(1, 3, 3)
        # img3 = ax3.imshow(wavefront - zernike_phase, cmap=mapp)
        # ax3.set_title('Residual')
        # img3.set_clim(m, -m)
        # plt.colorbar(img3, ax=ax3, orientation='horizontal')
        # # plt.show()

        return result.x


def generate_training_set(PSF_model_high, PSF_model_low, N_train=1500, N_test=500):

    N_samples = N_train + N_test
    N_act_high, N_act_low = PSF_model_high.N_act, PSF_model_low.N_act
    coef_high = np.random.uniform(low=-Z, high=Z, size=(N_samples, N_act_high))
    coef_low = np.zeros((N_samples, N_act_low))

    # FIXME! Watch out when using the ACTUATOR MODE
    defocus_coef = np.random.uniform(low=-1, high=1, size=N_act_low)
    defocus = np.dot(PSF_model_low.RBF_mat, defocus_coef)
    dataset = np.empty((N_samples, pix, pix, 2))

    ls_phase_fit = LS_fit(PSF_model_high.RBF_flat, PSF_model_low.RBF_flat)

    for i in range(N_samples):
        c_high = coef_high[i]
        # Nominal image
        im0, _s = PSF_model_high.compute_PSF(c_high)
        dataset[i, :, :, 0] = im0

        # Defocused image
        PSF_model_high.defocus = defocus.copy()
        im_foc, _s = PSF_model_high.compute_PSF(c_high)
        dataset[i, :, :, 1] = im_foc
        PSF_model_high.defocus = np.zeros_like(defocus)

        # Fit the Phase to lower order
        coef_low[i] = ls_phase_fit(c_high)

        if i%100 == 0:
            print(i)

    return dataset[:N_train], dataset[N_train:], coef_high[:N_train], coef_high[N_train:], coef_low[:N_train], coef_low[N_train:]

def adapt_training_set_coef(PSF_model_high, PSF_model_low, train_coef, test_coef):

    # To recycle the high frequency PSF images for another LOW model, we just adapt the LS fit coefs
    N_train, N_test = train_coef.shape[0], test_coef.shape[0]
    N_samples = N_train + N_test
    ls_phase_fit = LS_fit(PSF_model_high.RBF_flat, PSF_model_low.RBF_flat)
    N_act_low = PSF_model_low.N_act
    coef_high = np.concatenate([train_coef, test_coef], axis=0)
    coef_low = np.zeros((N_samples, N_act_low))

    for i in range(N_samples):
        c_high = coef_high[i]
        coef_low[i] = ls_phase_fit(c_high)

        if i%100 == 0:
            print(i)

    return coef_low[:N_train], coef_low[N_train:]


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    """ (1) Define the ACTUATORS """

    N_actuators = 25
    centers = actuator_centres(N_actuators)
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

    """ (2) Define a lower order actuator model """

    centers_low = actuator_centres(N_actuators=17)
    N_act_low = len(centers_low[0])
    plot_actuators(centers_low)
    rbf_mat_low = rbf_matrix(centers_low)

    # Fit the high-frequency map to the LOW actuator model
    ls_phase_fit = LS_fit(rbf_mat[-1], rbf_mat_low[-1])
    c_act_low = ls_phase_fit(c_act)

    guess0 = np.dot(rbf_mat_low[0], c_act_low)

    mins = min(phase0.min(), guess0.min())
    maxs = max(phase0.max(), guess0.max())

    m = min(mins, -maxs)
    mapp = 'bwr'
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = plt.subplot(1, 3, 1)
    img1 = ax1.imshow(phase0, cmap=mapp)
    ax1.set_title('True Wavefront')
    img1.set_clim(m, -m)
    plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 3, 2)
    img2 = ax2.imshow(guess0, cmap=mapp)
    ax2.set_title('Guessed Wavefront')
    img2.set_clim(m, -m)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')

    ax3 = plt.subplot(1, 3, 3)
    img3 = ax3.imshow(phase0 - guess0, cmap=mapp)
    ax3.set_title('Residual')
    img3.set_clim(m, -m)
    plt.colorbar(img3, ax=ax3, orientation='horizontal')
    plt.show()

    """ (3) Define the PSF model """

    PSF = PointSpreadFunction(rbf_mat)
    PSF_low = PointSpreadFunction(rbf_mat_low)

    N_train, N_test = 15000, 300
    train_PSF, test_PSF, train_coef, test_coef, train_low, test_low = generate_training_set(PSF, PSF_low, N_train, N_test)
    # train_PSF2, test_PSF2, train_coef2, test_coef2, train_low2, test_low2 = generate_training_set(PSF, PSF_low, N_train, N_test)
    #
    # train_PSF = np.concatenate([train_PSF, train_PSF2], axis=0)
    # test_PSF = np.concatenate([test_PSF, test_PSF2], axis=0)
    # train_coef = np.concatenate([train_coef, train_coef2], axis=0)
    # test_coef = np.concatenate([test_coef, test_coef2], axis=0)
    # train_low = np.concatenate([train_low, train_low2], axis=0)
    # test_low = np.concatenate([test_low, test_low2], axis=0)

    """ Crop to 20-pixels """
    # def crop_datacubes(datacube, crop_pix=25):
    #     N_PSF, pix = datacube.shape[0], datacube.shape[1]
    #     minPix, maxPix = (pix + 1 - crop_pix) // 2, (pix + 1 + crop_pix) // 2
    #     new_data = np.empty((N_PSF, crop_pix, crop_pix, 2))
    #     for k in range(N_PSF):
    #         new_data[k, :, :, 0] =  datacube[k,:,:,0][minPix:maxPix, minPix:maxPix]
    #         new_data[k, :, :, 1] =  datacube[k,:,:,1][minPix:maxPix, minPix:maxPix]
    #     return new_data
    #
    #
    # train_PSF_crop = crop_datacubes(train_PSF)
    # test_PSF_crop = crop_datacubes(test_PSF)


    N_channels = 2
    input_shape = (pix, pix, N_channels,)
    # input_shape = (25, 25, N_channels,)

    from keras.regularizers import l2

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(N_act_low))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    train_history = model.fit(x=train_PSF, y=train_low,
                              validation_data=(test_PSF, test_low),
                              epochs=250, batch_size=32, shuffle=True, verbose=1)

    guess_low = model.predict(test_PSF)
    residual_low = test_low - guess_low

    def draw_actuator_commands(commands, centers=centers_low):
        cent, delta = centers
        x = np.linspace(-1, 1, N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        image = np.zeros((N_PIX, N_PIX))
        for i, (xc, yc) in enumerate(cent):
            act_mask = (xx - xc)**2 + (yy - yc)**2 <= (delta/2)**2
            image += commands[i] * act_mask

        return image

    def command_check(model, test_PSF, test_low):
        N_test = test_low.shape[0]
        guess_low = model.predict(test_PSF)
        residual_low = test_low - guess_low
        error = []
        for k in range(N_test):
            c = residual_low[k]
            im = np.abs(draw_actuator_commands(c))
            error.append(im)
            # m_act = min(im.min(), -im.max())
            # plt.figure()
            # plt.imshow(im, cmap='bwr')
            # plt.clim(m_act, -m_act)
            # plt.colorbar()
        err = np.array(error)
        mean_err = np.mean(err, axis=0)
        plt.figure()
        plt.imshow(mean_err, cmap='Reds')
        plt.colorbar()
        plt.clim(np.min(mean_err[np.nonzero(mean_err)]), mean_err.max())
        plt.title(r'Average Actuator Error (Absolute Value)')
        plt.show()

    def psd_check(PSF_low, model, test_PSF, test_low):

        guess_low = model.predict(test_PSF)

        p0, pr = [], []
        for k in range(10):
            phase0 = np.dot(PSF_low.RBF_mat, test_low[k])
            residual = np.dot(PSF_low.RBF_mat, test_low[k] - guess_low[k])

            four = fftshift(fft2(phase0, norm='ortho'))
            psd = (np.abs(four)) ** 2

            psd_2dx = psd[N_PIX // 2, N_PIX // 2:]
            psd_2dy = psd[N_PIX // 2:, N_PIX // 2]
            freq = fftshift(np.fft.fftfreq(N_PIX, d=2 / N_PIX))
            p0.append((psd_2dy + psd_2dx)/2)

            f_res = fftshift(fft2(residual, norm='ortho'))
            psd_res = (np.abs(f_res)) ** 2
            psd_res_2dx = psd_res[N_PIX // 2, N_PIX // 2:]
            psd_res_2dy = psd_res[N_PIX // 2:, N_PIX // 2]
            pr.append((psd_res_2dx + psd_res_2dy)/2)

        p0 = np.array(p0)
        p0 = np.mean(p0, axis=0)

        pr = np.array(pr)
        pr = np.mean(pr, axis=0)

        plt.figure()
        plt.plot(freq[N_PIX // 2:], p0)
        plt.scatter(freq[N_PIX // 2:], p0)
        plt.plot(freq[N_PIX // 2:], pr)
        plt.scatter(freq[N_PIX // 2:], pr)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\lambda / D$')
        plt.show()






    z_fit = Zernike_fit(PSF_low.RBF_mat, N_zern=150)
    def fit_zernikes(z_fit, PSF_low, model, test_PSF, test_low, k=0):
        guess_low = model.predict(test_PSF)
        residual_low = test_low - guess_low

        # phase_fit_low = np.dot(PSF_low.RBF_mat, test_low[k])
        # phase_guess_low = np.dot(PSF_low.RBF_mat, guess_low[k])
        # residual = phase_fit_low - phase_guess_low
        # m = min(phase_fit_low.min(), -phase_fit_low.max())
        #
        # # plt.figure()
        # # plt.imshow(phase_fit_low, cmap='bwr')
        # # plt.clim(m, -m)
        # # plt.colorbar()
        # #
        # # plt.figure()
        # # plt.imshow(residual, cmap='bwr')
        # # plt.clim(m, -m)
        # # plt.colorbar()

        z_true = z_fit(test_low[k])
        z_resi = z_fit(residual_low[k])
        N_zern = z_fit.N_zern

        return z_true, z_resi

    def extract_radial_orders(zern_coef):
        N = zern_coef.shape[0]
        cop = zern_coef.copy()
        radial = []
        for k in np.arange(1, N):
            if len(cop)==0:
                break
            extract = cop[:k]
            cop = cop[k:]
            print(len(cop))
            radial.extend([np.mean(extract)])
        return radial

    Z_t, Z_r = [], []
    for k in range(100):
        z1, z2 = fit_zernikes(z_fit, PSF_low, model, test_PSF, test_low, k)
        Z_t.append(np.abs(z1))
        Z_r.append(np.abs(z2))

    Z_t = np.array(Z_t)
    Z_t = np.mean(Z_t, axis=0)

    Z_r = np.array(Z_r)
    Z_r = np.mean(Z_r, axis=0)

    N_zern = z_fit.N_zern
    # nn = np.arange(N_zern)
    nn = np.arange(len(extract_radial_orders(Z_t)))
    plt.figure()
    plt.scatter(nn, extract_radial_orders(Z_t), label='Before calibration', s=15)
    plt.scatter(nn, extract_radial_orders(Z_r), label='Residual after', s=15)
    # plt.yscale('log')
    # plt.ylim([-5, 0])
    plt.xlabel(r'Zernike radial order')
    plt.ylabel(r'Zernike coefficient')
    plt.legend()
    plt.show()






    def performance_check(PSF_high, PSF_low, model, test_PSF, test_coef, test_low):

        pupil_mask = PSF_high.pupil_mask
        s0 = np.max(test_PSF[:,:,:,0], axis=(1,2))      # Initial Strehl ratios

        guess_low = model.predict(test_PSF)
        residual_low = test_low - guess_low
        n_test = norm(test_low, axis=1)
        n_residual = norm(residual_low, axis=1)
        improve = 100 * np.mean((n_test - n_residual) / n_test)
        print('Average improvement [Norm LOW coefficients] : %.2f per cent' %improve)

        RMS0, RMS_ideal, RMS_true = [], [], []

        for k in range(300):

            phase_high = np.dot(PSF_high.RBF_mat, test_coef[k])
            phase_fit_low = np.dot(PSF_low.RBF_mat, test_low[k])
            ideal_residual = phase_high - phase_fit_low
            phase_guess_low = np.dot(PSF_low.RBF_mat, guess_low[k])
            true_residual = phase_high - phase_guess_low

            rms = []
            for p in [phase_high, phase_fit_low, ideal_residual, phase_guess_low, true_residual]:
                _flat = p[pupil_mask]
                rms.append(np.std(_flat))

            RMS0.append(rms[0])
            RMS_ideal.append(rms[2])
            RMS_true.append((rms[-1]))

        plt.figure()
        plt.hist(RMS0, histtype='step', label='Initial Wavefront')
        plt.hist(RMS_ideal, histtype='step', label='Ideal Residual')
        plt.hist(RMS_true, histtype='step', label='Machine Learning Residual')
        plt.xlabel(r'RMS wavefront $\lambda$')
        plt.xlim([0, 1.25])
        plt.legend()

        # return RMS_ideal, RMS_true

            mins = min(phase_high.min(), phase_fit_low.min())
            maxs = max(phase_high.max(), phase_fit_low.max())

            m = min(mins, -maxs)
            mapp = 'bwr'
            f, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
            ax1 = plt.subplot(2, 3, 1)
            img1 = ax1.imshow(phase_high, cmap=mapp)
            ax1.set_title('True Wavefront [High Freq.]  $\sigma=%.2f \lambda$' %rms[0])
            img1.set_clim(m, -m)
            # plt.colorbar(img1, ax=ax1, orientation='horizontal')

            ax2 = plt.subplot(2, 3, 2)
            img2 = ax2.imshow(phase_fit_low, cmap=mapp)
            ax2.set_title('Low Order Actuator Fit')
            img2.set_clim(m, -m)
            # plt.colorbar(img2, ax=ax2, orientation='horizontal')

            ax3 = plt.subplot(2, 3, 3)
            img3 = ax3.imshow(ideal_residual, cmap=mapp)
            ax3.set_title(r'Ideal Residual [True - Fit]  $\sigma=%.2f \lambda$' %rms[2])
            img3.set_clim(m, -m)
            # plt.colorbar(img3, ax=ax3, orientation='horizontal')
            # plt.show()

            ax4 = plt.subplot(2, 3, 4)
            img4 = ax4.imshow(draw_actuator_commands(guess_low[k], centers_low2), cmap=mapp)
            ax4.set_title('Actuators')
            m_act = min(guess_low[k].min(), -guess_low[k].max())
            img4.set_clim(m_act, -m_act)
            # plt.colorbar(img4, ax=ax4, orientation='horizontal')

            ax5 = plt.subplot(2, 3, 5)
            img5 = ax5.imshow(phase_guess_low, cmap=mapp)
            ax5.set_title('Machine Learning Guess')
            img5.set_clim(m, -m)
            # plt.colorbar(img5, ax=ax5, orientation='horizontal')

            ax6 = plt.subplot(2, 3, 6)
            img6 = ax6.imshow(true_residual, cmap=mapp)
            ax6.set_title('True Residual [True - Guess]  $\sigma=%.2f \lambda$' %rms[-1])
            img6.set_clim(m, -m)
            # plt.colorbar(img6, ax=ax6, orientation='horizontal')

    r_i, r_t = performance_check(PSF, PSF_low, model, test_PSF_crop, test_coef, test_low)
    performance_check(PSF, PSF_low2, model2, test_PSF, test_coef, test_low2)
    plt.show()




    """ Repeat the training for a new actuator model """

    centers_low2 = actuator_centres(N_actuators=14)
    plot_actuators(centers_low2)
    N_act_low2 = len(centers_low2[0])
    rbf_mat_low2 = rbf_matrix(centers_low2)
    PSF_low2 = PointSpreadFunction(rbf_mat_low2)
    train_low2, test_low2 = adapt_training_set_coef(PSF, PSF_low2, train_coef, test_coef)
    model2 = Sequential()
    model2.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(Conv2D(128, (3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dense(N_act_low2))
    model2.summary()

    model2.compile(optimizer='adam', loss='mean_squared_error')
    train_history2 = model2.fit(x=train_PSF, y=train_low2,
                              validation_data=(test_PSF, test_low2),
                              epochs=250, batch_size=32, shuffle=True, verbose=1)

    performance_check(PSF, PSF_low2, model2, test_PSF, test_coef, test_low2)
    plt.show()

    centers_low3 = actuator_centres(N_actuators=11)
    plot_actuators(centers_low3)
    N_act_low3 = len(centers_low3[0])
    rbf_mat_low3 = rbf_matrix(centers_low3)
    PSF_low3 = PointSpreadFunction(rbf_mat_low3)
    train_low3, test_low3 = adapt_training_set_coef(PSF, PSF_low3, train_coef, test_coef)
    model3 = Sequential()
    model3.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model3.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model3.add(Conv2D(128, (3, 3), activation='relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Flatten())
    model3.add(Dense(N_act_low3))
    model3.summary()

    model3.compile(optimizer='adam', loss='mean_squared_error')
    train_history3 = model3.fit(x=train_PSF, y=train_low3,
                              validation_data=(test_PSF, test_low3),
                              epochs=250, batch_size=32, shuffle=True, verbose=1)

    performance_check(PSF, PSF_low3, model3, test_PSF, test_coef, test_low3)




    # Sanity check: plot a PSF
    coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
    print(coef)

    PSF = PointSpreadFunction(N_zern)
    PSF.plot_PSF(np.zeros_like(coef), wave=1.0)
    plt.show()

    N_train, N_test = 5000, 100
    # N_train, N_test = 50000, 500
    # training_PSF, test_PSF, training_coef, test_coef = generate_training_set(PSF, focus, N_train, N_test)

    training_PSF, test_PSF, training_coef, test_coef = train_cosine(PSF, N_train, N_test)

    test_sine, _t, sine_coef, _c = train_cosine(PSF, 500, 1, cosine=False)

    plt.figure()
    plt.imshow(training_PSF[0,:,:,0], cmap='hot')
    plt.colorbar()

    plt.figure()
    plt.imshow(training_PSF[0,:,:,1], cmap='hot')
    plt.colorbar()

    plt.show()

    N_channels = 1
    input_shape = (pix, pix, N_channels,)

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
    # model.add(Dense(2*N_zern, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(3))
    # model.add(Activation('tanh', axis=-1))
    model.summary()


    R_m =  tf.constant(rbf_flat, dtype=tf.float32)
    def loss(y_true, y_pred):

        # true_phase = K.dot(R_m, tf.transpose(y_true))
        # phase = K.dot(R_m, tf.transpose(y_pred))
        # residual = true_phase - phase
        residual = K.dot(R_m, tf.transpose(y_true - y_pred))
        print(residual.shape)
        res = K.mean(K.mean((residual) ** 2))
        return res

    model.compile(optimizer='adam', loss='mean_squared_error')
    train_history = model.fit(x=training_PSF, y=training_coef,
                              validation_data=(test_PSF, test_coef),
                              epochs=25, batch_size=32, shuffle=True, verbose=1)
    loss_hist = train_history.history['loss']
    val_hist = train_history.history['val_loss']

    guess = model.predict(test_PSF)
    residual = test_coef - guess
    plt.figure()
    plt.scatter(test_coef[:,2], guess[:,2], s=10)
    plt.xlabel(r'True Coeff')
    plt.ylabel(r'Predicted Coeff')
    plt.axhline(0, linestyle='--')
    plt.axhline(1, linestyle='--', color='black')
    plt.axhline(-1, linestyle='--', color='black')
    plt.ylim([-2, 2])
    plt.show()

    # Test sine
    guess_sine = model.predict(test_sine)



    plt.figure()
    plt.semilogy(loss_hist, label='Training')
    plt.semilogy(val_hist, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Loss')
    plt.show()

    def compute_RMS(test_images, true_coef, mask=PSF.pupil):

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

            four = fftshift(fft2(phase0, norm='ortho'))
            psd = (np.abs(four)) ** 2
            psd_2dx = psd[N_PIX // 2, N_PIX // 2:]
            psd_2dy = psd[N_PIX // 2:, N_PIX // 2]
            freq = fftshift(np.fft.fftfreq(N_PIX, d=2 / N_PIX))

            f_res = fftshift(fft2(phase0 - guess0, norm='ortho'))
            psd_res = (np.abs(f_res)) ** 2
            psd_res_2dx = psd_res[N_PIX // 2, N_PIX // 2:]

            plt.figure()
            plt.plot(freq[N_PIX // 2:], (psd_2dy + psd_2dx)/2)
            plt.scatter(freq[N_PIX // 2:], (psd_2dy + psd_2dx)/2)
            plt.plot(freq[N_PIX // 2:], psd_res_2dx)
            plt.scatter(freq[N_PIX // 2:], psd_res_2dx)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\lambda / D$')
            plt.show()

    def performance(test_images, test_noisy, true_coef):

        " Evaluate the performance "
        guess = model.predict(test_images)
        residual = true_coef - guess

        guess_noisy = model.predict(test_noisy)
        residual_noisy = true_coef - guess_noisy

        norm0 = norm(true_coef, axis=1)
        mean_norm0 = np.mean(norm0)

        norm_res = norm(residual, axis=1)
        mean_norm_res = np.mean(norm_res)

        improv = (norm0 - norm_res) / norm0
        mean_improv = np.mean(improv)

        print('Average Improvement: %.1f [per cent]' % (100 * mean_improv))

        s0 = np.max(test_images[:,:,:,0], axis=(1,2))

        # Corrected PSF images
        sf, sf_noisy = [], []
        for res_coef, res_coef_noisy in zip(residual, residual_noisy):
            im, _sf = PSF.compute_PSF(res_coef)
            im, _sf_noisy = PSF.compute_PSF(res_coef_noisy)
            sf.append(_sf)
            sf_noisy.append(_sf_noisy)

        plt.figure()
        plt.hist(s0, histtype='step', label='Before')
        plt.hist(sf, histtype='step', label='After')
        plt.hist(sf_noisy, histtype='step', label='Noisy')
        plt.xlabel('Strehl ratio')
        plt.legend()
        plt.show()

    performance(test_PSF, test_PSF, test_coef)

    """ Sensitivity to Flat Field errors """



    # Clean Version
    s0 = np.max(test_PSF[:, :, :, 0], axis=(1, 2))
    guess = model.predict(test_PSF)
    residual = test_coef - guess

    sf = []
    for res_coef in residual:
        im, _sf = PSF.compute_PSF(res_coef)
        sf.append(_sf)

    # Dithered
    c1 = np.roll(test_PSF, shift=1, axis=2)
    c2 = np.roll(test_PSF, shift=2, axis=2)

    # sigmas = np.array([0.01, 0.05, 0.10, 0.15, 0.20])
    sigmas = np.array([0.20])
    mean_s = np.zeros_like(sigmas)
    mean_s_drizz = np.zeros_like(sigmas)
    for i in range(sigmas.shape[0]):

        plt.figure()
        plt.hist(s0, histtype='step', label='Before')
        plt.hist(sf, histtype='step', label='After (Clean)')

        avg, avg_drizz = [], []
        for k in range(1):
            flat = np.random.normal(loc=1, scale=sigmas[i], size=(pix, pix))
            flat = flat[np.newaxis, :, :, np.newaxis]
            test_PSF_flat = test_PSF * flat
            guess_noisy = model.predict(test_PSF_flat)
            residual_noisy = test_coef - guess_noisy

            # No Drizzle
            sf_noisy = []
            for res_coef in residual_noisy:
                im, _sf = PSF.compute_PSF(res_coef)
                sf_noisy.append(_sf)
            avg.append(sf_noisy)

            # Drizzle
            a, b, c, d, e = test_PSF * flat, c1 * flat, c2 * flat, c1.T * flat, c2.T * flat
            test_drizzled = np.zeros_like(test_PSF)
            for j in range(100):
                list_images_nom = [a[j,:,:,0], b[j,:,:,0], c[j,:,:,0]]
                list_images_foc = [a[j,:,:,1], b[j,:,:,1], c[j,:,:,1]]
                drizz_nom = drizzle(list_images_nom, shifts=[[0,0], [0, -1], [0, -2]])
                drizz_foc = drizzle(list_images_foc, shifts=[[0,0], [0, -1], [0, -2]])
                # print(drizz.shape)
                test_drizzled[j, :, :, 0] = drizz_nom
                test_drizzled[j, :, :, 1] = drizz_foc
            guess_drizz = model.predict(test_drizzled)
            residual_drizz = test_coef - guess_drizz
            sf_drizz = []
            for res_coef in residual_drizz:
                im, _sf = PSF.compute_PSF(res_coef)
                sf_drizz.append(_sf)
            avg_drizz.append(sf_drizz)


        avg = np.array(avg)
        print(avg.shape)
        sff = np.mean(avg, axis=0)
        mean_s[i] = np.mean(sff)

        avg_drizz = np.array(avg_drizz)
        sff_drizz = np.mean(avg_drizz, axis=0)
        mean_s_drizz[i] = np.mean(sff_drizz)

        plt.hist(sff, histtype='step', label='After (Noisy)')
        plt.hist(sff_drizz, histtype='step', label='After (Drizzled)')
        plt.legend()
        plt.title(r'Flat Field $\sigma=$%.2f' %sigmas[i])
        plt.xlabel('Strehl ratios')

    plt.figure()
    plt.scatter(sigmas, mean_s)
    plt.plot(sigmas, mean_s)
    plt.axhline(np.mean(sf), color='black', linestyle='--')
    plt.xlabel('Flat field $\sigma$')
    plt.ylabel('Average Strehl')
    plt.ylim([0.5, 0.9])
    plt.xlim([0.0, 0.25])
    plt.show()


    ## ==========================================================================

    def compute_input_centers(pix, size, shifts=(0, 0)):
        """
        Compute the centers (x_i, y_i) coordinates for each pixel
        :param pix: number of pixels
        :param size: physical size of the array
        :param shifts: fraction of pixel shift
        :return:
        """

        pix_scale = size / pix  # physical size of input pixels

        centers = np.empty((pix, pix, 2))

        offset_x = -size / 2 + pix_scale / 2 + shifts[0] * pix_scale
        offset_y = -size / 2 + pix_scale / 2 + shifts[1] * pix_scale
        for i in range(pix):
            for j in range(pix):
                x_c = i * pix_scale + offset_x
                y_c = j * pix_scale + offset_y
                centers[i, j, :] = [x_c, y_c]

        return centers


    def compute_output_centers(pix, size, p=2):
        """
        Compute the centers for the OUTPUT array, making use of the
        :param pix:
        :param size:
        :param p: oversampling factor of the OUTPUT array.
        p = 2 means it duplicates the number of pixels
        :return:
        """

        PIX = p * pix
        centers = compute_input_centers(pix=PIX, size=size)
        return centers


    def compute_overlap(centers, drop_size, CENTERS, PIX_SIZE):

        pix = centers.shape[0]
        PIX = CENTERS.shape[0]

        d, D = drop_size / 2, PIX_SIZE / 2
        overlaps = np.zeros((pix, pix, PIX, PIX))
        PIX_area = PIX_SIZE ** 2

        for i in range(pix):
            for j in range(pix):
                xi, yi = centers[i, j]

                # For each input pixel
                for k in range(PIX):
                    for l in range(PIX):
                        x0, y0 = CENTERS[k, l]
                        dx, dy = np.abs(x0 - xi), np.abs(y0 - yi)
                        if (d + D) > dy and (d + D) > dx:
                            hy = (d + D) - dy
                            hx = (d + D) - dx
                            overlap = hy * hx

                            overlaps[i, j, k, l] = overlap / PIX_area
        return overlaps


    def drizzle(input_images, shifts):

        pix = input_images[0].shape[0]
        size = 2
        p = 1
        pix_scale = size / pix
        drop_size = pix_scale

        N_images = len(input_images)
        output_CENTERS = compute_output_centers(pix, size, p)

        PIX_SIZE = pix_scale / p

        drizzled = []
        for i in range(N_images):
            shift = shifts[i]
            input_centers = compute_input_centers(pix, size, shift)
            overlaps = compute_overlap(input_centers, drop_size, output_CENTERS, PIX_SIZE)

            drizzled.append(np.tensordot(input_images[i], overlaps))

        drizzled = np.mean(np.array(drizzled), axis=0)
        return drizzled


    test_PSF

    drizz = drizzle([noise * clean, noise * c1, noise * c2, noise * c1.T,  noise * c2.T],
                    shifts=[[0,0], [0, -1], [0, -2], [-1, 0], [-2, 0]])


    ## ==========================================================================


    guess = model.predict(test_PSF)
    residual = test_coef - guess

    plt.figure()
    plt.axhline(y=Z, color='red', linestyle='--')
    plt.axhline(y=-Z, color='red', linestyle='--')
    for i in range(N_zern):
        plt.scatter(i*np.ones(N_test) + 1, residual[:,i], s=2, color='black')
    plt.ylim([-2, 2])
    plt.xlabel('Zernike Aberration')
    plt.ylabel('Residuals')
    plt.show()



    " Apply some Flat Field error "
    sigma = 0.2
    # flat = np.random.uniform(low=1-sigma, high=1+sigma, size=(pix, pix))
    flat = np.random.normal(loc=1, scale=sigma, size=(pix, pix))
    plt.figure()
    plt.imshow(flat, cmap='bwr')
    plt.colorbar()
    plt.clim(flat.min(), flat.max())
    plt.show()

    plt.hist(flat)
    flat = flat[np.newaxis, :, :, np.newaxis]
    # flat = flat[:, :, :, np.newaxis]
    test_PSF_flat = test_PSF * flat

    performance(test_PSF, test_PSF_flat, test_coef)


    " Actuators "

    def actuator_centres(N_actuators, rho_aper=0.5, rho_obsc=0.3/2):
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
            r = np.sqrt(x_c**2 + y_c**2)
            if r < 0.95*rho_aper and r > 1.1 * rho_obsc:
                act.append([x_c, y_c])

        total_act = len(act)
        print('Total Actuators: ', total_act)
        return act, delta

    def rbf_matrix(cent, delta, rho_aper=0.5, rho_obsc=0.3/2):
        N_act = len(cent)
        matrix = np.empty((N_PIX, N_PIX, N_act))
        x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x0, x0)
        rho = np.sqrt(xx**2 + yy**2)
        pupil = (rho <= rho_aper) & (rho >= rho_obsc)

        for k in range(N_act):
            xc, yc = cent[k][0], cent[k][1]
            r2 = (xx - xc)**2 + (yy - yc)**2
            matrix[:, :, k] = pupil * np.exp(-r2 / (1.*delta)**2)

        mat_flat = matrix[pupil]

        return matrix, mat_flat


    N_actuators = 33
    cent, delta = actuator_centres(N_actuators)
    N_act = len(cent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    circ1 = Circle((0,0), 0.5, linestyle='--', fill=None)
    circ2 = Circle((0,0), 0.15, linestyle='--', fill=None)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    for c in cent:
        ax.scatter(c[0], c[1], color='black')
    ax.set_aspect('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('%d actuators' %N_act)
    plt.show()

    rbf_mat, rbf_flat = rbf_matrix(cent, delta)
    one_act = np.zeros(N_act)
    k = 5
    one_act[k] = 1
    one_act[-(k+1)] = 1

    # 1 Actuator
    plt.figure()
    plt.imshow(np.dot(rbf_mat, one_act), cmap='coolwarm')
    plt.colorbar()
    plt.show()


    c_act = np.random.uniform(-1, 1, size=N_act)
    phase_act = np.dot(rbf_mat, c_act)

    plt.figure()
    plt.imshow(phase_act, cmap='coolwarm')
    plt.colorbar()
    plt.show()

    P = []
    for i in range(50):
        phase_act = np.dot(rbf_mat, np.random.uniform(-1, 1, size=N_act))
        p = (np.abs(fftshift(fft2(phase_act, norm='ortho'))))**2
        P.append(p)
    P = np.array(P)
    P = np.mean(P, axis=0)

    pp = [P]

    plt.figure()
    plt.semilogy(pp[0][N_PIX//2, N_PIX//2:])
    plt.semilogy(pp[1][N_PIX//2, N_PIX//2:])
    plt.semilogy(pp[2][N_PIX//2, N_PIX//2:])
    plt.show()



    guess = model.predict(test_PSF)
    residual = test_coef - guess

    PSD_t, PSD_g = [], []
    for j in range(5):
        true_act = np.dot(rbf_mat, test_coef[j])
        guess_act = np.dot(rbf_mat, guess[j])
        res = np.dot(rbf_mat, test_coef[j]-guess[j])
        # res = true_act - guess_act

        PSD_true = (np.abs(fftshift(fft2(true_act, norm='ortho'))))**2
        PSD_t.append(PSD_true)

        PSD_guess = (np.abs(fftshift(fft2(guess_act, norm='ortho'))))**2
        PSD_g.append(PSD_guess)

    PSD_t = np.array(PSD_t)
    pt = np.mean(PSD_t, axis=0)

    PSD_g = np.array(PSD_g)
    pg = np.mean(PSD_g, axis=0)

    plt.figure()
    plt.semilogy(pt[N_PIX//2, N_PIX//2:])
    plt.semilogy(pg[N_PIX//2, N_PIX//2:])
    plt.show()

        # mins = min(true_act.min(), guess_act.min())
        # maxs = max(true_act.max(), guess_act.max())
        #
        # m = min(mins, -maxs)
        # mapp = 'bwr'
        # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1 = plt.subplot(1, 3, 1)
        # img1 = ax1.imshow(true_act, cmap=mapp)
        # ax1.set_title('True Wavefront')
        # img1.set_clim(m, -m)
        # plt.colorbar(img1, ax=ax1, orientation='horizontal')
        #
        # ax2 = plt.subplot(1, 3, 2)
        # img2 = ax2.imshow(guess_act, cmap=mapp)
        # ax2.set_title('Guessed Wavefront')
        # img2.set_clim(m, -m)
        # plt.colorbar(img2, ax=ax2, orientation='horizontal')
        #
        # ax3 = plt.subplot(1, 3, 3)
        # img3 = ax3.imshow(res, cmap=mapp)
        # ax3.set_title('Residual')
        # img3.set_clim(m, -m)
        # plt.colorbar(img3, ax=ax3, orientation='horizontal')
        # plt.show()

    ### OLD STUFF - COSINE ANALYSIS
    #
    # class PointSpreadFunction(object):
    #     """
    #     PointSpreadFunction is in charge of computing the PSF
    #     for a given set of Zernike coefficients
    #     """
    #
    #     ### Parameters
    #     rho_aper = 0.5  # Size of the aperture relative to 1.0
    #     rho_obsc = 0.3 / 2  # Size of the central obscuration
    #     N_pix = N_PIX  # Number of pixels for the FFT computations
    #     minPix, maxPix = (N_pix + 1 - pix) // 2, (N_pix + 1 + pix) // 2
    #
    #     def __init__(self, N_zern):
    #         ### Zernike Wavefront
    #         x = np.linspace(-1, 1, self.N_pix, endpoint=True)
    #         xx, yy = np.meshgrid(x, x)
    #         rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
    #         self.pupil = (rho <= self.rho_aper) & (rho > self.rho_obsc)
    #         plt.figure()
    #         plt.imshow(self.pupil)
    #
    #         rho, theta = rho[self.pupil], theta[self.pupil]
    #         zernike = zern.ZernikeNaive(mask=self.pupil)
    #         _phase = zernike(coef=np.zeros(N_zern + 3), rho=rho / self.rho_aper, theta=theta, normalize_noll=False,
    #                          mode='Jacobi', print_option='Silent')
    #         H_flat = zernike.model_matrix[:, 3:]  # remove the piston and tilts
    #         self.H_matrix = zern.invert_model_matrix(H_flat, self.pupil)
    #
    #         # Save only the part of the model matrix that we need
    #         self.H_matrix = self.H_matrix[:, :, :N_zern]
    #         # FIXME: Watch out! The way we form H with the Zernike pyramid means that we can end up using the aberrations we
    #         # don't want. FIX this is in the future
    #         self.N_zern = self.H_matrix.shape[-1]
    #
    #         self.PEAK = self.peak_PSF()


    #
    # def train_cosine(PSF_model, N_train=1500, N_test=500, cosine=True):
    #     minPix, maxPix = (N_PIX + 1 - pix) // 2, (N_PIX + 1 + pix) // 2
    #
    #     mask = PSF_model.pupil
    #     N_samples = N_train + N_test
    #     freqx = np.random.uniform(low=0, high=5, size=(N_samples, 1))
    #     freqy = np.random.uniform(low=-5, high=5, size=(N_samples, 1))
    #     cos_sine = np.random.choice([-1, 1], size=(N_samples, 1))
    #     freq = np.concatenate([freqx, freqy, cos_sine], axis=1)
    #
    #     dataset = np.empty((N_samples, pix, pix, 1))
    #     x = np.linspace(-1, 1, N_PIX, endpoint=True)
    #     xx, yy = np.meshgrid(x, x)
    #     for i in range(N_samples):
    #         fx, fy, s = freqx[i], freqy[i], cos_sine[i][0]
    #         # fx, fy, s = 1, 2, cos_sine[i][0]
    #         # print(s)
    #         # if cosine==True:
    #         if s == -1:
    #             wavef = mask * np.cos(2 * np.pi * (fx * xx + fy * yy))
    #             print('C')
    #         # elif cosine==False:
    #         elif s == 1:
    #             print('S')
    #             wavef = mask * np.sin(2 * np.pi * (fx * xx + fy * yy))
    #
    #         pupil_function = mask * np.exp(1j * wavef)
    #         image = (np.abs(fftshift(fft2(pupil_function)))) ** 2
    #         image /= PSF_model.PEAK
    #
    #         dataset[i, :, :, 0] = image[minPix:maxPix, minPix:maxPix]
    #         if i % 100 == 0:
    #             print(i)
    #             # plt.figure()
    #             # plt.imshow(wavef)
    #             # plt.colorbar()
    #
    #     return dataset[:N_train], dataset[N_train:], freq[:N_train], freq[N_train:]
    #
    #
    # def training_cosine(PSF_model, N_act, N_train=1500, N_test=500):
    #
    #     N_samples = N_train + N_test
    #     coef = np.random.uniform(low=0, high=Z, size=(N_samples, 1))
    #     idx = np.random.randint(0, N_act // 2, size=N_samples)
    #
    #     true_coef = np.zeros((N_samples, 2))
    #
    #     dataset = np.empty((N_samples, pix, pix, 1))
    #     for i in range(N_samples):
    #         # Nominal image
    #         c = coef[i][0]
    #         j = idx[i]
    #         one_act = np.zeros(N_act)
    #         one_act[j] = c
    #         one_act[-(j + 1)] = c
    #         im0, _s = PSF_model.compute_PSF(one_act)
    #         dataset[i, :, :, 0] = im0
    #         true_coef[i, 0] = c
    #         true_coef[i, -1] = j
    #         # true_coef[i] = one_act
    #
    #         if i % 100 == 0:
    #             print(i)
    #
    #     return dataset[:N_train], dataset[N_train:], true_coef[:N_train], true_coef[N_train:]









