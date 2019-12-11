from __future__ import print_function  # for Python2

import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns


from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from numpy.linalg import norm as norm

from itertools import tee

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# PARAMETERS
Z = 1.75                    # Strength of the aberrations
pix = 30                    # Pixels to crop the PSF
N_PIX = 256
RHO_APER = 0.5
RHO_OBSC = 0.15
N_WAVES = 5                 # Normally 10
WAVE_N = 2.0

# ### Super useful code to display variables and their sizes (helps you clear the RAM)
# import sys
#
# for var, obj in locals().items():
#     print(var, sys.getsizeof(obj))

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
    # alpha = 1/np.sqrt(np.log(100/30))
    alpha = 1.

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

    def __init__(self, RBF_matrices, N_waves=N_WAVES, wave0=1.0, waveN=WAVE_N):

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

    def plot_PSF(self, coef, wave_idx, cmap='hot'):
        """
        Plot an image of the PSF
        """
        PSF, strehl = self.compute_PSF(coef, wave_idx)

        plt.figure()
        plt.imshow(PSF, cmap=cmap)
        plt.title('Strehl: %.3f' %strehl)
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

def generate_training_set(PSF_model, N_train=1500, N_test=500, foc=1.0):

    N_act = PSF_model.N_act
    N_samples = N_train + N_test
    coef = np.random.uniform(low=-Z, high=Z, size=(N_samples, N_act))
    # training_coef, test_coef
    # coef = np.concatenate([np.load('training_coef.npy'),
    #                        np.load('test_coef.npy')], axis=0)
    print(coef.shape)

    # defocus = np.zeros(N_zern)
    # defocus[1] = focus
    # FIXME! Watch out when using the ACTUATOR MODE
    # defocus = np.random.uniform(low=-1.25, high=1.25, size=N_act)
    # np.save('defocus', defocus)
    defocus = foc * np.load('defocus.npy')
    dataset = np.empty((N_samples, pix, pix, 2*N_WAVES))

    foc_phase = np.dot(PSF_model.RBF_mat[0], defocus)
    std_foc = np.std(foc_phase[PSF_model.pupil_masks[0]])
    cfoc = max(-np.min(foc_phase), np.max(foc_phase))

    plt.figure()
    plt.imshow(foc_phase, cmap='bwr')
    plt.clim(-cfoc, cfoc)
    plt.title(r'RMS %.3f $\lambda$' % std_foc)
    plt.colorbar()

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

def readout_noise_images(dataset, coef, RMS_READ, N_copies=3):
    """
    Introduces Readout Noise in the datasets
    It creates N_copies of each PSF image with noise
    The noise level is given by the RMS_READ
    :param dataset: set of PSF images
    :param coef: set of coefficients associated to the PSF images
    :param RMS_READ: noise level
    :param N_copies: number of copies to make for each PSF
    :return:
    """
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
    del dataset ### Remove clean PSF to save memory
    # Offset the arrays by the RMS_READ to minimize the number of pixels with negative values
    new_data += 5 * RMS_READ
    return new_data, new_coef


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    wave0 = 1.0
    waves_ratio = np.linspace(1., WAVE_N / wave0, N_WAVES, endpoint=True)

    N_actuators = 20
    centers = actuator_centres(N_actuators, radial=False)
    N_act = len(centers[0][0])

    ### Compute Distance Matrix
    # Similar to a Covariance matrix on the neighbour actuator distance
    c0, delta0 = centers[0]
    dist_mat = np.empty((N_act, N_act))
    for i in range(N_act):
        xi, yi = c0[i]
        for j in range(N_act):
            xj, yj = c0[j]
            dist_mat[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)

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
    plt.imshow(phase0, cmap='RdBu')
    plt.colorbar()

    plt.figure()
    plt.imshow(phaseN, cmap='RdBu')
    plt.colorbar()
    plt.show()

    PSF = PointSpreadFunction(rbf_matrices)

    for idx in range(N_WAVES):
        PSF.plot_PSF(c_act, wave_idx=idx)

    plt.show()

    # ================================================================================================================ #

    """ Generate a training set of CLEAN PSF images """

    N_train, N_test = 5000, 500
    N_batches = 3
    for k in range(N_batches):
        training_PSF, test_PSF, training_coef, test_coef = generate_training_set(PSF, N_train, N_test)
        np.save('training_PSF%d' % k, training_PSF)
        np.save('test_PSF%d' % k, test_PSF)
        np.save('training_coef%d' % k, training_coef)
        np.save('test_coef%d' % k, test_coef)

    def load_dataset(N_batches, load_waves):
        training_PSF, test_PSF, training_coef, test_coef = [], [], [], []
        for k in range(N_batches):
            print(k)
            training_p = np.load('training_PSF%d.npy' % k)[:, :, :, :2*load_waves]
            test_p = np.load('test_PSF%d.npy' % k)[:, :, :, :2*load_waves]
            training_c = np.load('training_coef%d.npy' % k)
            test_c = np.load('test_coef%d.npy' % k)

            training_PSF.append(training_p)
            test_PSF.append(test_p)
            training_coef.append(training_c)
            test_coef.append(test_c)

        training_PSF = np.concatenate(training_PSF, axis=0)
        test_PSF = np.concatenate(test_PSF, axis=0)
        training_coef = np.concatenate(training_coef, axis=0)
        test_coef = np.concatenate(test_coef, axis=0)
        return training_PSF, test_PSF, training_coef, test_coef

    training_PSF, test_PSF, training_coef, test_coef = load_dataset(N_batches=5, load_waves=N_WAVES)

    H = PSF.RBF_mat[0]
    pup = PSF.pupil_masks[0]
    eps = 1e-3
    c = np.random.uniform(low=-eps, high=eps, size=N_act)
    w0 = np.dot(H, c)
    f_w0 = fftshift(fft2(w0, norm='ortho'))
    f_pi = fftshift(fft2(pup, norm='ortho'))
    delta = np.real(1j * (np.conj(f_pi) * f_w0 - f_pi * np.conj(f_w0)))

    B = np.linalg.tensorsolve(w0, delta)
    im_f_w0 = np.imag(f_w0)


    # ================================================================================================================ #

    # Mutual Information
    from scipy.special import gamma, psi
    from scipy import ndimage
    from scipy.linalg import det
    from numpy import pi

    from sklearn.neighbors import NearestNeighbors


    def nearest_distances(X, k=1):
        '''
        X = array(N,M)
        N = number of points
        M = number of dimensions

        returns the distance to the kth nearest neighbor for every point in X
        '''
        knn = NearestNeighbors(n_neighbors=k + 1)
        knn.fit(X)
        d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
        return d[:, -1]  # returns the distance to the kth nearest neighbor


    def entropy(X, k=1):
        ''' Returns the entropy of the X.

        Parameters
        ===========

        X : array-like, shape (n_samples, n_features)
            The data the entropy of which is computed

        k : int, optional
            number of nearest neighbors for density estimation

        Notes
        ======

        Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
        of a random vector. Probl. Inf. Transm. 23, 95-101.
        See also: Evans, D. 2008 A computationally efficient estimator for
        mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
        and:
        Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
        information. Phys Rev E 69(6 Pt 2):066138.
        '''

        # Distance to kth nearest neighbor
        r = nearest_distances(X, k)  # squared distances
        n, d = X.shape
        volume_unit_ball = (pi ** (.5 * d)) / gamma(.5 * d + 1)
        '''
        F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
        for Continuous Random Variables. Advances in Neural Information
        Processing Systems 21 (NIPS). Vancouver (Canada), December.

        return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
        '''
        return (d * np.mean(np.log(r + np.finfo(X.dtype).eps))
                + np.log(volume_unit_ball) + psi(n) - psi(k))

    def mutual_information(variables, k=1):
        '''
        Returns the mutual information between any number of variables.
        Each variable is a matrix X = array(n_samples, n_features)
        where
          n = number of samples
          dx,dy = number of dimensions

        Optionally, the following keyword argument can be specified:
          k = number of nearest neighbors for density estimation

        Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
        '''
        if len(variables) < 2:
            raise AttributeError(
                "Mutual information must involve at least 2 variables")
        all_vars = np.hstack(variables)
        return (sum([entropy(X, k=k) for X in variables])
                - entropy(all_vars, k=k))

    X = training_PSF[:,:,:,0]
    Y = training_PSF[:,:,:,1]
    X = X.reshape((25000, 30*30))
    Y = Y.reshape((25000, 30*30))
    variables = [X, Y]
    mi = mutual_information(variables)


    # ================================================================================================================ #



    # ================================================================================================================ #
    #                              UNCERTAINTY - Bayesian Networks approx with Dropout                                 #
    # ================================================================================================================ #

    def create_model(waves, name):
        """
        Creates a CNN model for NCPA calibration
        :param waves: Number of wavelengths in the training set (to adjust the number of channels)
        :return:
        """
        input_shape = (pix, pix, 2 * waves,)
        model = Sequential()
        model.name = name
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        # model.add(Conv2D(32, (3, 3), activation='relu'))
        # model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(N_act))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def create_model_dropout(waves, keep_rate):
        """
        Creates a CNN model for NCPA calibration with Dropout
        A keep_rate of 0.95 means 95% of the weights are not set to 0
        :param waves: Number of wavelengths in the training set (to adjust the number of channels)
        :param keep_rate: Probability of keeping a weight (opposite to dropout rate)
        :return:
        """
        drop_rate = 1 - keep_rate
        input_shape = (pix, pix, 2 * waves,)
        model = Sequential()
        name = "model_dropout%.2f" % keep_rate
        model.name = name
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(Dropout(rate=drop_rate))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(rate=drop_rate))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(rate=drop_rate))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Dropout(rate=drop_rate))
        model.add(Flatten())
        #
        model.add(Dense(N_act))
        model.add(Dropout(rate=drop_rate))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def predict_with_uncertainty(f, test_set, N_classes, N_samples=100):
        """
        Makes use of the fact that our model has Dropout to sample from
        the posterior of predictions to get an estimate of the uncertainty of the predictions
        :param f: a Keras function that forces the model to act on "training mode" because Keras
        freezes the Dropout during testing
        :param test_set: the test images to be used
        :param N_classes: number of classes that the model predicts
        :param N_samples: number of times to sample the posterior
        :return:
        """

        result = np.zeros((N_samples,) + (test_set.shape[0], N_classes))

        for i in range(N_samples):
            result[i, :, :] = f((test_set, 1))[0]

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)

        return result, prediction, uncertainty

    ### Loop over several values of Dropout to see how the BIAS and Epistemic Uncertainty changes
    for dropout in [0.90]:

        dropout_model = create_model_dropout(waves=N_WAVES, keep_rate=dropout)
        train_history = dropout_model.fit(x=training_PSF, y=training_coef,
                                  validation_data=(test_PSF, test_coef),
                                  epochs=30, batch_size=32, shuffle=True, verbose=1)

        guess = dropout_model.predict(test_PSF)
        residual = test_coef - guess
        print(norm(residual))

        print("\nModel with Dropout: %.2f" % dropout)
        print("After training:")
        print("Norm of residuals: %.2f" % norm(residual))

        f = K.function([dropout_model.layers[0].input, K.learning_phase()],
                       [dropout_model.layers[-1].output])

        result, avg_pred, unc = predict_with_uncertainty(f, test_PSF[:500], N_classes=N_act, N_samples=1000)
        bias = test_coef[:500] - avg_pred                       # [N_test, N_act]
        mean_bias_per_act = np.mean(bias, axis=0)               # [N_act]
        mean_abs_bias_per_act = np.mean(np.abs(bias), axis=0)
        uncertainty_per_act = np.mean(unc, axis=0)              # [N_act]

        n_act = 10
        print("\nFor the first %d Actuators" % n_act)
        print("Average Bias: ")
        print(mean_bias_per_act[:n_act])
        print("Average Abs(Bias): %.3f" %np.mean(mean_abs_bias_per_act))
        print(mean_abs_bias_per_act[:n_act])
        print("Uncertainty: %.3f" % np.mean(uncertainty_per_act))
        print(uncertainty_per_act[:n_act])

    # ================================================================================================================ #

    """ Analyze the errors in the predictions """

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
            act_mask = (xx - xc) ** 2 + (yy - yc) ** 2 <= (delta / 2) ** 2
            image += commands[i] * act_mask

        return image

    im_bias = draw_actuator_commands(mean_bias_per_act, centers[0])
    minn = min(np.min(im_bias), -np.max(im_bias))
    plt.figure()
    plt.imshow(im_bias, cmap='RdBu')
    plt.clim(vmin=minn, vmax=-minn)
    plt.colorbar()

    im_abs_bias = draw_actuator_commands(mean_abs_bias_per_act, centers[0])
    minn = np.min(im_abs_bias[np.nonzero(im_abs_bias)])
    plt.figure()
    plt.imshow(im_abs_bias, cmap='Reds')
    plt.clim(vmin=minn, vmax=np.max(im_abs_bias))
    plt.colorbar()
    plt.show()

    ### Try to fit a Gaussian to the Dropout predictions
    from scipy.stats import norm as gaussian
    from scipy.stats import normaltest
    from matplotlib import cm
    # colors = ['red', 'blue', 'black', 'green', 'pink']
    N_show = 5
    colors = cm.copper(np.linspace(0, 1, N_show))
    print("\nThe True Coefficient was within [x] SIGMA of the average prediction")
    k_act = 10
    for i_ex in range(N_show):

        predictions = result[:, i_ex, k_act]
        _mean, _std = avg_pred[i_ex, k_act], unc[i_ex, k_act]

        statistic, p_value = normaltest(predictions)        # Normal Test
        mean, std = gaussian.fit(predictions)               # Fit a Gaussian profile

        # True value
        true_value = test_coef[i_ex, k_act]
        sigmas = np.abs(true_value - _mean) / std

        print("%.2f" % sigmas)

        plt.figure()
        plt.imshow(test_PSF[i_ex, :, :, 0], cmap='hot')

        plt.figure()
        plt.hist(predictions, bins=20, histtype='step', color=colors[i_ex])

        plt.axvline(true_value, linestyle='--', color=colors[i_ex])
        plt.xlim([-2, 2])
        plt.xlabel("Actuator Command")
        plt.title('PSF #%d (True Coeff within %.2f $\sigma$)' % (i_ex + 1, sigmas))
    plt.show()

    N_samples = result.shape[0]
    N_examples = 150
    k_act = 1
    plt.figure()
    _x = np.linspace(-2, 2, 50)
    plt.plot(_x, _x, linestyle='--', color='black')
    plt.errorbar(x=test_coef[:N_examples, k_act], y=avg_pred[:N_examples, k_act],
                 yerr=2*unc[:N_examples, k_act], fmt='o', ms=2)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

    plt.figure()
    plt.plot(_x, 0 * _x, linestyle='--', color='black')
    plt.errorbar(x=test_coef[:N_examples, k_act], y=avg_pred[:N_examples, k_act] - test_coef[:N_examples, k_act],
                 yerr=2*unc[:N_examples, k_act], fmt='o', ms=2)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()

    # ================================================================================================================ #
    #                                                Ensemble Approach                                                 #
    # ================================================================================================================ #

    N_models = 20
    keep_rate = 0.90

    # Train a Single Model to
    one_model = create_model_dropout(waves=N_WAVES, keep_rate=keep_rate)
    train_history = one_model.fit(x=training_PSF, y=training_coef,
                                      validation_data=(test_PSF, test_coef),
                                      epochs=30, batch_size=32, shuffle=True, verbose=1)

    one_guess = one_model.predict(test_PSF)
    one_residual = test_coef - one_guess
    print(norm(one_residual))

    print("\nModel with Dropout: %.2f" % keep_rate)
    print("After training:")
    print("Norm of residuals: %.2f" % norm(one_residual))

    ### Train N_models
    list_models = []
    for k in range(N_models):
        _model = create_model_dropout(waves=N_WAVES, keep_rate=keep_rate)
        train_history = _model.fit(x=training_PSF, y=training_coef,
                                      validation_data=(test_PSF, test_coef),
                                      epochs=30, batch_size=32, shuffle=True, verbose=1)
        list_models.append(_model)

    # Combine the guesses
    list_results = []
    list_guesses, list_uncertain = [], []
    N_examples = 250
    for _model in list_models:
        print(_model.name)
        f = K.function([_model.layers[0].input, K.learning_phase()],
                       [_model.layers[-1].output])
        # Use the Uncertain predictions
        result, avg_pred, unc = predict_with_uncertainty(f, test_PSF[:N_examples], N_classes=N_act, N_samples=250)
        list_results.append(result)
        list_guesses.append(avg_pred)
        list_uncertain.append(unc)

    guesses = np.stack(list_guesses)
    many_guesses = np.mean(guesses, axis=0)
    many_residual = test_coef[:N_examples] - many_guesses

    uncertainties = np.stack(list_uncertain)
    variance = np.sqrt(np.sum(uncertainties ** 2, axis=0) / N_models ** 2)

    print("\nEnsemble Approach with %d Models" % N_models)
    print("\nNorm Residuals for 1 model")
    print(np.mean(norm(one_residual, axis=1)))
    print("\nNorm Residuals for %d models averaged" % N_models)
    print(np.mean(norm(many_residual, axis=1)))

    """ Show how the residual decreases with the Number of Models"""
    plt.figure()
    m0 = 0
    for i in np.arange(1, len(list_models) + 1):
        mean_guess = np.mean(guesses[:i], axis=0)
        resis = test_coef[:N_examples] - mean_guess
        mm = np.mean(norm(resis, axis=1))
        if i == 1:
            m0 = mm.copy()
        plt.scatter(i, mm/m0*100, color='blue')
    plt.xlabel(r'N models (Ensemble)')
    plt.ylabel(r'Percentage error relative to 1 Model')
    plt.ylim([80, 100])
    plt.show()


    k_act = 1
    truth = test_coef[:N_examples, k_act]
    pred = many_guesses[:, k_act]
    one_pred = one_guess[:N_examples, k_act]
    var = variance[:, k_act]
    plt.figure()
    _x = np.linspace(-2, 2, 50)
    plt.plot(_x, _x, linestyle='--', color='black')
    plt.errorbar(x=truth, y=pred, yerr=var, fmt='o', ms=2)
    plt.errorbar(x=truth, y=one_pred, yerr=var, fmt='o', ms=2)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()

    plt.figure()
    plt.scatter(truth, np.log10(np.abs(one_pred - truth)), s=6)
    plt.scatter(truth, np.log10(np.abs(pred - truth)), s=6)
    plt.show()

    plt.hist(np.abs(one_pred - truth))
    plt.hist(np.abs(pred - truth))

    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #

    # ================================================================================================================ #
    #                                                      NOISY                                              #
    # ================================================================================================================ #

    RMS_READ = 1./500
    N_copies = 5

    read_train_PSF, read_train_coef = readout_noise_images(training_PSF, training_coef, RMS_READ, N_copies)
    read_test_PSF, read_test_coef = readout_noise_images(test_PSF, test_coef, RMS_READ, N_copies)

    # Train a Single Model to
    keep_rate = 0.90

    for keep_rate in [0.95, 0.99]:
        noisy_model = create_model_dropout(waves=N_WAVES, keep_rate=keep_rate)
        train_history = noisy_model.fit(x=read_train_PSF, y=read_train_coef,
                                          validation_data=(read_test_PSF, read_test_coef),
                                          epochs=10, batch_size=32, shuffle=True, verbose=1)

        noisy_guess = noisy_model.predict(read_test_PSF)
        noisy_residual = read_test_coef - noisy_guess
        noisy_testnorm = np.mean(norm(read_test_coef, axis=1))
        noisy_errnorm = np.mean(norm(noisy_residual, axis=1))
        print("\nModel with Dropout: %.2f" % keep_rate)
        print("Before: %.3f" % noisy_testnorm)
        print("After: %.3f" % noisy_errnorm)


    ### Train N_models
    list_noisy_models = []
    keep_rate = 0.95
    N_models = 10
    for k in range(N_models):
        _model = create_model_dropout(waves=N_WAVES, keep_rate=keep_rate)
        train_history = _model.fit(x=read_train_PSF, y=read_train_coef,
                                          validation_data=(read_test_PSF, read_test_coef),
                                          epochs=10, batch_size=32, shuffle=True, verbose=1)
        list_noisy_models.append(_model)

    # Combine the guesses
    list_results = []
    list_guesses, list_uncertain = [], []
    N_examples = 150
    for _model in list_noisy_models:
        print(_model.name)
        f = K.function([_model.layers[0].input, K.learning_phase()],
                       [_model.layers[-1].output])
        # Use the Uncertain predictions
        result, avg_pred, unc = predict_with_uncertainty(f, read_test_PSF[:N_examples], N_classes=N_act, N_samples=250)
        # list_results.append(result)
        list_guesses.append(avg_pred)
        list_uncertain.append(unc)

    guesses = np.stack(list_guesses)
    many_guesses = np.mean(guesses, axis=0)
    many_residual = read_test_coef[:N_examples] - many_guesses
    one_residual = read_test_coef[:N_examples] - guesses[0]


    print("\nEnsemble Approach with %d Models" % N_models)
    print("\nNorm Residuals for 1 model")
    print(np.mean(norm(one_residual, axis=1)))
    print("\nNorm Residuals for %d models averaged" % N_models)
    print(np.mean(norm(many_residual, axis=1)))

    """ Show how the residual decreases with the Number of Models"""
    plt.figure()
    m0 = 0
    for i in np.arange(1, len(list_noisy_models) + 1):
        mean_guess = np.mean(guesses[:i], axis=0)
        resis = read_test_coef[:N_examples] - mean_guess
        mm = np.mean(norm(resis, axis=1))
        if i == 1:
            m0 = mm.copy()
        plt.scatter(i, mm/m0*100, color='blue')
    plt.xlabel(r'N models (Ensemble)')
    plt.ylabel(r'Percentage error relative to 1 Model')
    plt.ylim([80, 100])
    plt.show()



    ######
    #####


    class PSFGenerator(object):

        def __init__(self, PSF_model):
            self.PSF_model = PSF_model
            pass

        def update_psf(self, residual_coef, focus):
            """
            After getting some predictions, update the PSF images based on the residual aberrations
            :param residual_coef: residual coefficients after calibration
            :param focus: defocus strength used to generate the PSF images and train the model
            """
            N_samples = residual_coef.shape[0]
            defocus = focus * np.load('master_defocus.npy')
            dataset = np.empty((N_samples, pix, pix, 2 * N_WAVES))

            for i in range(N_samples):
                if i % 100 == 0:
                    print(i)
                for wave_idx in range(N_WAVES):
                    # Nominal image
                    im0, _s = self.PSF_model.compute_PSF(residual_coef[i], wave_idx=wave_idx)
                    dataset[i, :, :, 2 * wave_idx] = im0
                    # Defocused image
                    im_foc, _s = self.PSF_model.compute_PSF(residual_coef[i] + defocus, wave_idx=wave_idx)
                    dataset[i, :, :, 2 * wave_idx + 1] = im_foc
            return dataset

        def generate_dataset(self, N_train, N_test, min_RMS, max_RMS, focus_strength, focus_coef, show_fig=False):
            """
            Uses the PSF model to generate [N_train + N_test] PSF images
            We try to same the range of RMS wavefront error uniformly to allow iterative calibration
            i.e. we want the CNN model to look at PSF with all cases of RMS (low aberrations, high aberrations)
            To do that, we estimate the scaling of the aberration coefficients, so that the RMS is
            approximately within a given range [min_RMS, max_RMS]

            With regards to the defocus (or diversity) we can use whatever type we want
            by providing the location of a set of wavefront coefficients [focus_coef]
            The rescaling [focus_strength] allows us to investigate how the intensity of defocus
            for a given shape, affects the performance
            :param N_train:
            :param N_test:
            :param min_RMS:
            :param max_RMS:
            :param focus_strength:
            :param focus_coef: path to a set of model coefficients to be used as 'defocus' / 'diversity'
            :param show_fig: whether to show the estimates the code uses to fulfill the [min_RMS, max_RMS]
            :return:
            """

            #TODO: Investigate if the Defocus intensity matters
            #TODO: Investigate if the Random Map matters (Correlations between Actuator error and Random Map??)

            N_coef = self.PSF_model.N_act
            N_samples = N_train + N_test

            # defocus = focus * np.load('master_defocus.npy')
            defocus = focus_strength * np.load(focus_coef)
            dataset = np.empty((N_samples, pix, pix, 2 * N_WAVES))


            # We need to find a range of coefficient intensities that gives us a reasonable range
            # for the RMS wavefront
            c_guess = np.linspace(0.0, 2.5, 50)
            N_tries = 50
            mean_RMS = []
            # fig = plt.figure()
            for c in c_guess:
                # print(c)
                _coef = c * np.random.uniform(low=-1, high=1, size=(N_tries, N_coef))
                _rms = []
                for k in range(N_tries):
                    wavefront = np.dot(self.PSF_model.RBF_flat[0], _coef[k])
                    _rms.append(np.std(wavefront))
                # plt.scatter(c * np.ones(N_tries), _rms, s=3, color='blue')
                mean_RMS.append(np.mean(_rms))
            # plt.xlabel(r'Coefficient Scale')
            # plt.ylabel(r'RMS wavefront [$2\pi$ waves]')
            # plt.axhline(min_RMS, linestyle='--', color='black')
            # plt.axhline(max_RMS, linestyle='--', color='red')

            mean_RMS = np.array(mean_RMS)
            i_cmax = np.argwhere(mean_RMS < max_RMS)
            c_max = c_guess[i_cmax[-1, 0]]
            i_cmin = np.argwhere(mean_RMS > min_RMS)
            c_min = c_guess[i_cmin[0, 0]]
            # plt.axvline(c_min, linestyle='--', color='black')
            # plt.axvline(c_max, linestyle='--', color='red')
            print(c_min, c_max)

            if show_fig:
                plt.show()

            # Now that we know the range of scales that gives us our desired RMS range
            # we can uniformly sample it, to get a training set that covers both low and high
            # wavefront errors

            coef = np.random.uniform(low=-1, high=1, size=(N_samples, N_coef))
            print("Generating %d PSF images" % N_samples)
            RMS = []
            for i in range(N_samples):
                if i % 100 == 0:
                    print(i)

                scale = np.random.uniform(low=c_min, high=c_max, size=1)
                coef[i] *= scale        # rescale to sample the RMS range

                wavefront = np.dot(self.PSF_model.RBF_flat[0], coef[i])
                RMS.append(np.std(wavefront))
                for wave_idx in range(N_WAVES):
                    # Nominal image
                    im0, _s = self.PSF_model.compute_PSF(coef[i], wave_idx=wave_idx)
                    dataset[i, :, :, 2 * wave_idx] = im0

                    # Defocused image
                    im_foc, _s = self.PSF_model.compute_PSF(coef[i] + defocus, wave_idx=wave_idx)
                    # TODO: Add Defocus uncertainties in the future!
                    dataset[i, :, :, 2 * wave_idx + 1] = im_foc

            # Show histogram of RMS wavefronts
            if show_fig:
                plt.figure()
                plt.hist(RMS, bins=20, histtype='step')
                plt.xlabel(r'RMS wavefront')
                plt.axvline(min_RMS, linestyle='--', color='black')
                plt.axvline(max_RMS, linestyle='--', color='red')
                plt.show()

            return dataset[:N_train], dataset[N_train:], coef[:N_train], coef[N_train:]

        def save_dataset_batches(self, path_to_save, N_train, N_test, N_batches,
                                 min_RMS, max_RMS, focus_strength, focus_coef, k0=0):
            """
            Generate and save N_batches of [N_train + N_test] PSF images
            This code tries to generate PSF images that cover a wide range of
            RMS wavefront errors, to allow iterative NCPA calibration

            You can specify your desired range of RMS [min_RMS, max_RMS] in 2 pi waves at minimum wavelength
            :param path_to_save: name of the directory to save the files
            :param N_train: Number of PSF images for the Training set
            :param N_test: Number of PSF images for the Test set
            :param N_batches: Number of batches of [N_train + N_test] to generate
            :param min_RMS: minimum RMS wavefront error in your PSF images
            :param max_RMS: maximum RMS wavefront error in your PSF images
            :param focus_strength: intensity of the defocus term, it rescales the focus_coef
            :param focus_coef: path to a set of coefficients to be used to define the defocus wavefront / diversity
            :param k0: initial index for the filenames (to avoid overwriting old files)
            :return:
            """

            cur_dir = os.getcwd()
            path = os.path.join(cur_dir, path_to_save)
            # Watch out because it won't create a Folder / Subfolder simultaneously
            # if you do os.mkdir(path) and path is cwd\\ Folder \\ Subfolder
            split_paths = os.path.split(path)
            path0 = ''
            # We must loop over the Folder structure, creating one layer at a time
            for _path in split_paths:
                path0 = os.path.join(path0, _path)
                path = os.path.join(cur_dir, path0)
                try:
                    os.mkdir(path)
                    print("Directory ", path, " Created ")
                except FileExistsError:
                    print("Directory ", path, " already exists")

            for k in np.arange(k0, k0 + N_batches):
                training_PSF, test_PSF, \
                training_coef, test_coef = self.generate_dataset(N_train, N_test, min_RMS, max_RMS, focus_strength, focus_coef)
                np.save(os.path.join(path, 'training_PSF%d' % k), training_PSF)
                np.save(os.path.join(path, 'test_PSF%d' % k), test_PSF)
                np.save(os.path.join(path, 'training_coef%d' % k), training_coef)
                np.save(os.path.join(path, 'test_coef%d' % k), test_coef)

        def load_dataset_batches(self, path_to_load, N_batches, N_waves):
            """
            Load just a few batches, you can specify how many wavelength channels you want to keep

            Sometimes training with a large dataset with many batches and many wavelengths can be
            problematic, specially if you want to add Noise effects or other Data Augmetation techniques
            :param path_to_load: where the files are stored
            :param N_batches: how many batches to load
            :param N_waves: how many wavelength channels to keep
            :return:
            """

            cur_dir = os.getcwd()
            path = os.path.join(cur_dir, path_to_load)

            training_PSF, test_PSF, training_coef, test_coef = [], [], [], []
            print("Loading %d Batches of %d Wavelength channels")
            for k in range(N_batches):
                print(k)
                training_p = np.load(os.path.join(path, 'training_PSF%d.npy' % k))[:, :, :, :2 * N_waves]
                test_p = np.load(os.path.join(path, 'test_PSF%d.npy' % k))[:, :, :, :2 * N_waves]
                training_c = np.load(os.path.join(path, 'training_coef%d.npy' % k))
                test_c = np.load(os.path.join(path, 'test_coef%d.npy' % k))

                training_PSF.append(training_p)
                test_PSF.append(test_p)
                training_coef.append(training_c)
                test_coef.append(test_c)

            training_PSF = np.concatenate(training_PSF, axis=0)
            test_PSF = np.concatenate(test_PSF, axis=0)
            training_coef = np.concatenate(training_coef, axis=0)
            test_coef = np.concatenate(test_coef, axis=0)

            print("Size of the loaded dataset: ", training_PSF.shape)

            return training_PSF, training_coef, test_PSF, test_coef


    class BatchNoiseTraining(object):

        def __init__(self, PSF_model, dataset, model_options, batch_size=500):

            self.PSF_model = PSF_model
            _train_PSF = dataset[0]
            N_samples = _train_PSF.shape[0]
            self.batch_size = batch_size
            self.N_batches = N_samples // batch_size

            self.clean_dataset = dataset        # [train_PSF, train_coef, test_PSF, test_coef]
            print("\nInitializing BatchNoiseTraining:\n")
            self.model = create_model_dropout(waves=model_options["N_WAVES"],
                                              keep_rate=model_options["keep_rate"])

        def select_bad_pixels(self, N_channels, p_bad=0.10, max_bad_pixels=3):
            """
            Randomly select which pixels to asign as BAD PIXELS for a datacube with N_channels
            First, we randomly select WHICH CHANNELS will be affected by bad pixels
            This is done using a binomial distribution with success rate [p_bad]

            For the selected channels we randomly choose HOW MANY bad pixels to put sampling uniformly
            from [1, max_bad_pixels]

            Finally, we select WHERE in the image the bad pixels will be

            :param N_channels:
            :param p_bad:
            :param max_bad_pixels:
            :return: a list of channel indices, and a list of pixel indices for each channel
            """

            which_channels = np.random.binomial(n=1, p=p_bad, size=N_channels)
            channels = list(np.argwhere(which_channels == 1)[:, 0])
            if len(channels) == 0:
                channels = [np.random.choice(N_channels, size=1)]
            pixels = []
            for _chan in channels:
                _pix = []
                how_many = np.random.randint(low=1, high=max_bad_pixels)
                for i in range(how_many):
                    x_bad, y_bad = np.random.choice(pix, size=2)
                    _pix.append([x_bad, y_bad])
                pixels.append(_pix)

            # print(channels)
            # print(pixels)
            return channels, pixels

        def add_bad_pixels(self, PSF_images, coef, N_copies):
            """
            Data augmentation to include Bad Pixels in the images
            For each case within the dataset of PSF_images, we generate N_copies
            Each copy contains a random instance of Bad Pixel corruption

            The selection of Bad Pixels is done in self.select_bad_pixels
            :param PSF_images:
            :param coef:
            :param N_copies:
            :return:
            """

            BAD, HOT = 0, 99
            N_PSF, pix, _pix, N_chan = PSF_images.shape
            N_coef = coef.shape[-1]
            new_data = np.empty((N_copies * N_PSF, pix, pix, N_chan))
            new_coef = np.empty((N_copies * N_PSF, N_coef))

            print("\nAdding Bad Pixels to %d PSF images [%d copies]" % (N_PSF, N_copies))

            for k in range(N_PSF):
                # if k % 100 == 0:
                #     print("PSFs done: %d" % k)
                PSF = PSF_images[k].copy()
                coef_copy = coef[k].copy()
                for i_copy in range(N_copies):
                    new_data[N_copies * k + i_copy] = PSF
                    # Select the bad pixels
                    bad_channels, bad_pixels = self.select_bad_pixels(N_chan, p_bad=1./N_chan, max_bad_pixels=3)
                    for j_chan, _badpix in zip(bad_channels, bad_pixels):   # Loop over the Channels
                        for i_bad, j_bad in _badpix:                        # Loop over the pixel positions
                            # value = np.random.choice([BAD, HOT], size=1)    # Is it hot or is it cold?
                            value = BAD
                            new_data[N_copies * k + i_copy, i_bad, j_bad, j_chan] = value
                    new_coef[N_copies * k + i_copy] = coef_copy
            return new_data, new_coef

        def add_readout_noise(self, PSF_images, coef, SNR_min=100, SNR_max=500, N_copies=2, random=False):
            """
            Data augmentation to include Readout Noise in the images
            For each case within the dataset of PSF_images, we generate N_copies
            Each copy contains a random instance of Readout Noise

            The noise for the copies ranges (in terms of SNR) from SNR_min to SNR_max
            if random==False we uniformly sample the range [SNR_1, ..., SNR_Ncopies]
            if random==True for each copy we randomly select the SNR within the range [SNR_min, SNR_max]

            The noise map is generated using a Gaussian distribution with std = 1 / SNR
            and it's added to the images

            We finally offset the PSF_images with 5 * 1 / SNR_min to make sure most pixels have positive values

            :param PSF_images: [N_samples, pix, pix, N_wavechan] clean PSF images
            :param coef: aberration coefficients (we need to create multiple copies)
            :param SNR_min: minimum Signal-to-Noise ratio
            :param SNR_max: maximum Signal-to-Noise ratio
            :param N_copies: How many copies with noise to create for each PSF image
            :param random: whether to use fixed values of SNR of random values within the range [SNR_min, SNR_max]
            :return: [N_copies * N_samples, pix, pix, N_wavechan]
            """

            N_PSF, pix, _pix, N_chan = PSF_images.shape
            N_coef = coef.shape[-1]
            new_data = np.empty((N_copies * N_PSF, pix, pix, N_chan))
            new_coef = np.empty((N_copies * N_PSF, N_coef))

            ### SNR range
            if random == False:
                SNR_range = np.linspace(SNR_min, SNR_max, N_copies, endpoint=True)
                RMS_READ = 1. / SNR_range
            else:
                RMS_READ = [1/SNR_min]

            print("\nAdding Readout Noise to %d PSF images [%d copies]" % (N_PSF, N_copies))
            print("SNR range from %d to %d" % (SNR_min, SNR_max))
            for k in range(N_PSF):
                # if k % 100 == 0:
                #     print("PSFs done: %d" % k)
                PSF = PSF_images[k].copy()
                coef_copy = coef[k].copy()
                for i_copy in range(N_copies):
                    if random is False:     # Fixed values in the Range
                        read_out = np.random.normal(loc=0, scale=RMS_READ[i_copy], size=(pix, pix, N_chan))
                    else:                   # Select the SNR randomly within the Range
                        SNR = np.random.uniform(low=SNR_min, high=SNR_max, size=1)
                        read_out = np.random.normal(loc=0, scale=1./SNR, size=(pix, pix, N_chan))
                    new_data[N_copies * k + i_copy] = PSF + read_out
                    new_coef[N_copies * k + i_copy] = coef_copy
            # Offset the arrays by the RMS_READ to minimize the number of pixels with negative values
            new_data += 5 / SNR_min
            return new_data, new_coef

        def train_in_batches(self, loops, epochs_per_batch, N_noise_copies, verbose=1):
            """
            We want to train with as many examples of aberrations as possible, and with as many
            copies of noise effects as possible to increase robustness

            However, doing that with a single dataset can cause Memory issues. For that reason,
            we train in small batches

            We loop over the CLEAN dataset in batches of self.batch_size, possibily "loops" x times
            For each batch, we run data augmentation, creating N_noise_copies of each PSF example
            We ran the training for epochs_per_batch, and after that we select the next batch and start over

            This ensures that the model is exposed to many instances of the noise effects (Readout Noise, etc)
            because they are randomly generated multiple times, without blowing up the memory

            :param loops: Number of times to cycle over the complete CLEAN dataset
            :param epochs_per_batch: Number of epochs to run the training for a mini_batch of the CLEAN dataset
            :param N_noise_copies: Number of copies of Noise effects to generate for each batch
            :param verbose:
            :return:
            """

            loss, val_loss = [], []
            print("\nTraining Model with %d batches of %d PSF images" % (self.N_batches, self.batch_size))
            for l in range(loops):
                print("Loop #%d/%d" % (l + 1, loops))
                for k_iter in range(self.N_batches):
                    print("\nBatch #%d/%d" % (k_iter + 1, self.N_batches))
                    train_batch = [x[k_iter * self.batch_size : (k_iter + 1) * self.batch_size] for x in self.clean_dataset[:2]]
                    test_p, test_c = self.clean_dataset[2][:500], self.clean_dataset[3][:500]

                    # Introduce Noise in the samples
                    noisy_train_PSF, noisy_train_coef = self.add_readout_noise(train_batch[0], train_batch[1], N_copies=N_noise_copies)
                    noisy_test_PSF, noisy_test_coef = self.add_readout_noise(test_p, test_c, N_copies=2)

                    # Add Bad Pixels
                    # badpix_train_PSF, badpix_train_coef = self.add_bad_pixels(noisy_train_PSF, noisy_train_coef, N_copies=2)
                    # badpix_test_PSF, badpix_test_coef = self.add_bad_pixels(noisy_test_PSF, noisy_test_coef, N_copies=2)

                    train_history = self.model.fit(x=noisy_train_PSF, y=noisy_train_coef,
                                                   validation_data=(noisy_test_PSF, noisy_test_coef),
                                                   epochs=epochs_per_batch, batch_size=32,
                                                   shuffle=True, verbose=verbose)

                    loss.extend(train_history.history['loss'])
                    val_loss.extend(train_history.history['val_loss'])

        def test_one_iteration(self, SNR_min=100, SNR_max=500, N_noise=5, random_noise=False):
            """
            Test the performance of the trained model [1 single iteration]
            :param N_noise:
            :return:
            """

            test_PSF, test_coef = self.clean_dataset[-2], self.clean_dataset[-1]
            noisy_test_PSF, noisy_test_coef = self.add_readout_noise(test_PSF, test_coef,
                                                                     SNR_min=SNR_min, SNR_max=SNR_max,
                                                                     N_copies=N_noise, random=random_noise)

            guess = self.model.predict(noisy_test_PSF)
            residual = noisy_test_coef - guess

            print("_____________________________________________________________")
            print("\nTesting the Model on %d PSF images" % N_test)
            print("%d copies with Readout Noise" % N_noise)
            print("\nOverall Performance")
            mean_test = np.mean(norm(noisy_test_coef, axis=1))
            mean_res = np.mean(norm(residual, axis=1))
            print("Averaged morm of Test Coefficients: %.3f" % mean_test)
            print("Averaged norm of Residual Coefficients: %.3f" % mean_res)

            ### Aggregated results across all cases of SNR
            RMS0, RMS = [], []
            H_matrix = self.PSF_model.RBF_flat[0]
            N_PSF = noisy_test_PSF.shape[0]
            for k in range(N_PSF):
                true_coef = noisy_test_coef[k]
                true_wavefront = np.dot(H_matrix, true_coef)
                RMS0.append(np.std(true_wavefront))

                residual_wavefront = np.dot(H_matrix, residual[k])
                RMS.append(np.std(residual_wavefront))
            # plt.figure()
            # plt.hist(RMS0, bins=50, histtype='step', label='Initial')
            # plt.hist(RMS, bins=50, histtype='step', label='Final')
            # plt.xlabel(r'RMS wavefront error [waves]')
            # plt.legend()

            metric = [np.mean(RMS0), np.std(RMS0), np.mean(RMS), np.std(RMS)]

            # ### Slice by SNR to see the impact of Readout Noise
            # SNR = np.linspace(SNR_min, SNR_max, N_noise, endpoint=True)
            # mus, stds = [], []
            # for i, s in enumerate(SNR):
            #     print("\nSNR: ", s)
            #     _p = noisy_test_PSF[i::N_noise]      # Read Out only
            #     # _p = badpix_test_PSF[2 * i::2 * N_noise]  # Read Out + Bad Pixels
            #     _c = noisy_test_coef[i::N_noise]
            #     # _c = badpix_test_coef[2 * i::2 * N_noise]
            #
            #     guess = self.model.predict(_p)
            #     residual = _c - guess
            #     mean_test = np.mean(norm(_c, axis=1))
            #     mean_res = np.mean(norm(residual, axis=1))
            #     print("Averaged morm of Test Coefficients: %.3f" % mean_test)
            #     print("Averaged norm of Residual Coefficients: %.3f" % mean_res)
            #
            #     RMS0 = self.compute_RMS(H_matrix, _c)
            #     RMS = self.compute_RMS(H_matrix, residual)
            #
            #     mu_RMS = np.mean(RMS)
            #     mus.append(mu_RMS)
            #     std_RMS = np.std(RMS)
            #     stds.append(std_RMS)
            #     med_RMS = np.median(RMS)

                # plt.figure()
                # ax = sns.kdeplot(RMS0, RMS, cmap="Blues", shade=True, bw='scott', clip=[0, 1.0, 0, 0.4], shade_lowest=True)
                # ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='No Correction')
                # plt.xlabel(r'Initial RMS [waves]')
                # plt.ylabel(r'Final RMS [waves]')
                # plt.xlim([0, 1.0])
                # plt.ylim([0, 0.4])
                # plt.title(r'SNR %d : $\mu=%.3f$, $\tilde{x}=%.3f$, $\sigma=%.3f$ [$\lambda$]' % (int(s), mu_RMS, med_RMS, std_RMS))
                # plt.legend()

            # plt.figure()
            # plt.errorbar(SNR, mus, stds, fmt='o')
            # plt.plot(SNR, mus, color='black')
            # plt.xlabel(r'SNR')
            # plt.ylabel(r'RMS wavefront')
            # plt.show()

            # Return a Performance Metrics

            return metric

        def compute_RMS(self, H_matrix, coef):
            """
            Helper function to compute the RMS
            :param H_matrix: Model matrix [pupil] with the pupil mask applied to only count the nonzero points
            :param coef: [N_samples, N_coef] set of aberration coefficients
            :return: a list of RMS wavefront errors
            """
            rms = []
            for i in range(coef.shape[0]):
                wavefront = np.dot(H_matrix, coef[i])
                rms.append(np.std(wavefront))
            return rms

        def test_iteratively(self, N_iter, random_noise=False):
            """
            Test the model for several iterations
            At the beginning of each iteration, the PSF images are corrupted with Readout Noise
            The predictions of the model are applied as corrections, and using the residual coefficients
            the PSF images are updated for the next iteration
            :param N_iter:
            :return:
            """
            test_PSF, test_coef = self.clean_dataset[-2], self.clean_dataset[-1]
            H_matrix = self.PSF_model.RBF_flat[0]

            list_iter = list(range(N_iter + 1))
            RMS0 = self.compute_RMS(H_matrix, test_coef)    # Compute the initial RMS wavefront
            list_RMS = [RMS0]
            mus, stds, meds = [np.mean(RMS0)], [np.std(RMS0)], [np.median(RMS0)]  # Average RMS, std and median at each iter

            plt.figure()
            plt.hist(RMS0, bins=50, histtype='step', label='Initial')

            print("\nTesting the performance for %d iterations" % N_iter)
            print("Size of the test set: ", test_PSF.shape[0])
            print("Initial RMS: mean=%.3f, std=%.3f, median=%.3f" % (mus[0], stds[0], meds[0]))
            clean_PSF, clean_coef = test_PSF, test_coef

            for k in range(N_iter):
                print("\nAt iteration #%d/%d" % (k + 1, N_iter))

                noisy_test_PSF, noisy_test_coef = self.add_readout_noise(clean_PSF, clean_coef, N_copies=1, random=random_noise)
                guess = self.model.predict(noisy_test_PSF)
                residual = noisy_test_coef - guess
                mean_test = np.mean(norm(noisy_test_coef, axis=1))
                mean_res = np.mean(norm(residual, axis=1))
                print("Averaged morm of Test Coefficients: %.3f" % mean_test)
                print("Averaged norm of Residual Coefficients: %.3f" % mean_res)

                RMS = self.compute_RMS(H_matrix, residual)
                plt.hist(RMS, bins=50, histtype='step', label='Iteration %d' % (k + 1))
                list_RMS.append(RMS)
                mu, rms, med = np.mean(RMS), np.std(RMS), np.median(RMS)        # Save the statistics
                mus.append(mu)
                stds.append(rms)
                meds.append(med)
                print("After correction: mean=%.3f, std=%.3f, median=%.3f" % (mu, rms, med))

                updated_PSF = Generator.update_psf(residual, focus=focus)
                clean_PSF, clean_coef = updated_PSF, residual       # For the next iteration

            plt.xlabel(r'RMS wavefront error [waves]')
            plt.legend()

            # Plot the mean RMS and the std as a function of Iterations
            plt.figure()
            plt.errorbar(list_iter, mus, stds)
            plt.plot(list_iter, mus, color='black')
            plt.xlabel('Iteration')
            plt.ylabel(r'RMS wavefront error')
            plt.ylim([0, 1.0])

            # Use seaborn to plot the 2D density plots of RMS before and after each iteration
            for i_iter, (rms_initial, rms_final) in enumerate(pairwise(list_RMS)):
                plt.figure()
                ax = sns.kdeplot(rms_initial, rms_final, cmap="Blues", shade=True, bw='scott', clip=[0, 1.0, 0, 0.4], shade_lowest=True)
                ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='No Correction')
                plt.xlabel(r'Initial RMS [waves]')
                plt.ylabel(r'Final RMS [waves]')
                plt.xlim([0, 1.0])
                # plt.xlim([0, 1.1*np.max(rms_initial)])
                plt.ylim([0, 0.4])
                # plt.ylim([0, 1.1*np.max(rms_initial)])
                plt.legend()
                plt.title('Iteration #%d' % (i_iter + 1))

            # Plot the 2D density plot of RMS before and after the LAST ITERATION
            plt.figure()
            ax = sns.kdeplot(list_RMS[0], list_RMS[-1], cmap="Blues", shade=True, bw='scott', clip=[0, 1.0, 0, 0.4],
                             shade_lowest=True)
            plt.xlabel(r'Before Calibration RMS [waves]')
            plt.ylabel(r'After %d Iterations RMS [waves]' % (N_iter))
            plt.xlim([0, 1.0])
            plt.ylim([0, 0.15])

            return list_RMS

    master_defocus = np.random.uniform(low=-1, high=1, size=N_act)
    np.save('master_defocus', master_defocus)
    path_coef_focus = os.path.join(os.getcwd(), 'master_defocus.npy')

    focus = 1.5
    min_RMS, max_RMS = 0.10, 0.85
    N_train, N_test = 1500, 500
    Generator = PSFGenerator(PSF)

    # train_p, train_c, test_p, test_c = Generator.generate_dataset(N_train, N_test, min_RMS, max_RMS, focus)
    N_batches = 10
    path_to_save = "Datasets"
    Generator.save_dataset_batches(path_to_save, N_train, N_test, N_batches, min_RMS, max_RMS, focus, path_coef_focus)

    N_waves = 10
    dataset = Generator.load_dataset_batches(path_to_load=path_to_save, N_batches=N_batches, N_waves=10)
    model_options = {"N_WAVES": N_waves, "keep_rate": 0.95}

    batch_trainer = BatchNoiseTraining(PSF, dataset, model_options, batch_size=2500)
    batch_trainer.train_in_batches(loops=1, epochs_per_batch=5, N_noise_copies=5, verbose=1)
    # batch_trainer.test_one_iteration(SNR_min=100, SNR_max=500, N_noise=5)

    list_RMS = batch_trainer.test_iteratively(N_iter=3, random_noise=True)

    # ================================================================================================================ #
    #                                     Analysis of the DEFOCUS term                                                 #
    # ================================================================================================================ #

    """ Impact of Defocus Intensity"""

    GeneratorDefocus = PSFGenerator(PSF)
    N_batches = 5
    min_RMS, max_RMS = 0.10, 0.85
    N_train, N_test = 1500, 500
    # Watch out for the folder Names
    focus_range = [0.25, 0.50, 1.00, 1.50, 2.00]        # [0, 1, 2, 3, 4]
    new_focus_range = [1.25, 1.75]                      # [5, 6]
    path_focus = "Focus Intensity"
    for i, foc in enumerate(focus_range):
    # for i, foc in zip([5, 6], new_focus_range):
        # First we generate the Datasets, which takes a long time
        print("\nFocus Intensity: %.2f" % foc)
        path_to_save = os.path.join(path_focus, "%d" % i)
        GeneratorDefocus.save_dataset_batches(path_to_save, N_train, N_test, N_batches,
                                              min_RMS, max_RMS, foc, path_coef_focus)

    N_waves = 10
    model_options = {"N_WAVES": N_waves, "keep_rate": 0.95}
    for i, foc in enumerate(focus_range):
    # for i, foc in zip([5, 6], new_focus_range):
    # for i, foc in zip([2], [1.00]):
        # After that we can run whichever cases we want, without having to generate the PSF images again
        path_to_save = os.path.join(path_focus, "%d" % i)
        print("\nFocus Intensity: %.2f" % foc)
        dataset = GeneratorDefocus.load_dataset_batches(path_to_load=path_to_save, N_batches=N_batches, N_waves=10)
        batch_trainer_focus = BatchNoiseTraining(PSF, dataset, model_options, batch_size=2500)
        batch_trainer_focus.train_in_batches(loops=1, epochs_per_batch=5, N_noise_copies=5, verbose=1)
        batch_trainer_focus.test_one_iteration()

    ##
    """ Impact of Random Defocus """

    # Use the 'optimum' strength from the previous analysis
    opt_foc = 1.75
    # np.load('master_defocus.npy')
    path_focus = "FocusRandom"
    for k in range(5):
        GeneratorDefocus = PSFGenerator(PSF)
        path_to_save = os.path.join(path_focus, "%d" % k)
        master_defocus = np.random.uniform(low=-1, high=1, size=N_act)
        np.save(os.path.join(path_to_save, 'master_defocus'), master_defocus)
        path_coef_focus = os.path.join(path_to_save, 'master_defocus.npy')
        # path_to_save = path_focus

        GeneratorDefocus.save_dataset_batches(path_to_save, N_train, N_test, N_batches,
                                              min_RMS, max_RMS, opt_foc, path_coef_focus)

        phase = np.dot(PSF.RBF_mat[0], master_defocus)
        phase_f = np.dot(PSF.RBF_flat[0], master_defocus)
        rms_foc = np.std(phase_f)
        plt.figure()
        plt.imshow(phase)
        plt.colorbar()
        plt.title('%.3f' % rms_foc)

        dataset = GeneratorDefocus.load_dataset_batches(path_to_load=path_to_save, N_batches=N_batches, N_waves=10)
        batch_trainer_focus = BatchNoiseTraining(PSF, dataset, model_options, batch_size=2500)
        batch_trainer_focus.train_in_batches(loops=1, epochs_per_batch=5, N_noise_copies=5, verbose=1)
        batch_trainer_focus.test_one_iteration()

    """ Zernike Defocus and other Diversities """
    import zern_core as zern

    pupil = PSF.pupil_masks[0]
    N_zern = 10
    x = np.linspace(-1, 1, N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)

    rho, theta = rho[pupil], theta[pupil]
    zernike = zern.ZernikeNaive(mask=pupil)
    _phase = zernike(coef=np.zeros(N_zern), rho=rho / (RHO_APER), theta=theta, normalize_noll=False,
                     mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix[:, 3:]
    H_matrix = zern.invert_model_matrix(H_flat, pupil)
    N_zern = H_matrix.shape[-1]

    defocus_zern = H_matrix[:, :, 1]
    y_obs = defocus_zern[pupil]
    _H = PSF.RBF_flat[0]
    _Ht = _H.T
    N = np.dot(_Ht, _H)
    invN = np.linalg.inv(N)
    x_fit = np.dot(invN, np.dot(_Ht, y_obs))

    defocus_actu = np.dot(PSF.RBF_mat[0], x_fit)
    plt.figure()
    plt.imshow(defocus_zern)
    plt.colorbar()

    plt.figure()
    plt.imshow(defocus_actu)
    plt.colorbar()

    plt.show()

    opt_foc = 1.75
    path_focus = "FocusZernike"
    np.save('master_defocus', defocus_actu)
    GeneratorDefocus = PSFGenerator(PSF)

    GeneratorDefocus.save_dataset_batches(path_focus, N_train, N_test, N_batches, min_RMS, max_RMS, focus=opt_foc)


    ### Can you brute force the OPTIMUM diversity?

    def create_directory(path_to_save):
        cur_dir = os.getcwd()
        path = os.path.join(cur_dir, path_to_save)
        # Watch out because it won't create a Folder / Subfolder simultaneously
        # if you do os.mkdir(path) and path is cwd\\ Folder \\ Subfolder
        split_paths = os.path.split(path)
        path0 = ''
        # We must loop over the Folder structure, creating one layer at a time
        for _path in split_paths:
            path0 = os.path.join(path0, _path)
            path = os.path.join(cur_dir, path0)
            try:
                os.mkdir(path)
                print("Directory ", path, " Created ")
            except FileExistsError:
                print("Directory ", path, " already exists")

    from time import time
    N_batches = 1
    min_RMS, max_RMS = 0.10, 0.85
    N_train, N_test = 5000, 500
    GeneratorMC = PSFGenerator(PSF)

    path_MC = "BruteForce"
    N_tries = 1000
    opt_foc = 1.75  # Use the 'optimum' strength from the previous analysis
    all_coef = np.random.uniform(low=-1, high=1, size=(N_tries, N_act))
    diversity_coefs, metrics = [], []

    N_waves = 5
    model_options = {"N_WAVES": N_waves, "keep_rate": 0.95}
    start0 = time()

    for i in np.arange(40, 100):
        start = time()
        print("\n====================================================")
        print("Trial %d " % i)
        print("\n====================================================")
        path_to_save = os.path.join(path_MC, "%d" % i)
        create_directory(path_to_save)
        master_defocus = all_coef[i]
        np.save(os.path.join(path_to_save, 'master_defocus'), master_defocus)
        path_coef_focus = os.path.join(os.path.join(path_to_save, 'master_defocus.npy'))


        GeneratorMC.save_dataset_batches(path_to_save, N_train, N_test, N_batches,
                                              min_RMS, max_RMS, opt_foc, path_coef_focus)

        dataset = GeneratorMC.load_dataset_batches(path_to_load=path_to_save, N_batches=N_batches, N_waves=N_waves)

        batch_trainer_MC = BatchNoiseTraining(PSF, dataset, model_options, batch_size=1000)
        batch_trainer_MC.train_in_batches(loops=1, epochs_per_batch=3, N_noise_copies=3, verbose=1)
        metric = batch_trainer_MC.test_one_iteration()
        print("\n====================================================")
        print(metric)
        print("\n====================================================")
        metrics.append(metric)
        np.save(os.path.join(path_to_save, 'metric'), np.array(metric))

        end = time()
        time_iter = (end - start) / 60      # in min
        total_time = (end - start0) / 60      # in min
        remaining_iter = N_tries - (i + 1)
        estimate = remaining_iter * total_time / (i + 1) / 60
        print("Time for iteration: %.2f minutes" % time_iter)
        print("Total time: %.2f minutes" % total_time)
        print("ETA: -%.2f hours" % estimate)

    all_coef = []
    metrics = []
    rms_c = []
    for i in np.arange(40):
        path_to_save = os.path.join(path_MC, "%d" % i)
        all_coef.append(np.load(os.path.join(path_to_save, 'master_defocus.npy')))
        metrics.append(np.load(os.path.join(path_to_save, 'metric.npy')))
    all_coef = np.stack(all_coef)
    metrics = np.stack(metrics)

    norm_c = norm(all_coef, axis=1)


    plt.figure()
    plt.scatter(norm_c, metrics[:,2])
    plt.show()

    plt.figure()
    plt.plot(np.sort(metrics[:,2]))
    plt.show()
    ##### Run iterations




    # ================================================================================================================ #
    #                                      SHAP Values - Model Interpretability                                        #
    # ================================================================================================================ #

    import shap

    def generate_pixel_ids(N_pixels):
        """
        Generate Labels for each pixel according to their (i,j) index
        """
        central_pix = N_pixels // 2
        x = list(range(N_pixels))
        xx, yy = np.meshgrid(x, x)
        xid = xx.reshape((N_pixels * N_pixels))
        yid = yy.reshape((N_pixels * N_pixels))

        labels = ["(%d, %d)" % (x - central_pix, y - central_pix) for (x, y) in zip(xid, yid)]
        return labels

    pix_label = generate_pixel_ids(pix)

    """
    No Noise
    """

    keep_rate = 0.95
    clean_model = create_model_dropout(waves=N_WAVES, keep_rate=dropout)
    train_history = clean_model.fit(x=training_PSF, y=training_coef,
                                      validation_data=(test_PSF, test_coef),
                                      epochs=30, batch_size=32, shuffle=True, verbose=1)

    N_background = 250
    N_shap_samples = 250
    clean_background = training_PSF[np.random.choice(training_PSF.shape[0], N_background, replace=False)]
    clean_explainer = shap.DeepExplainer(clean_model, clean_background)

    # Select only the first N_shap from the test set
    test_shap = test_PSF[:N_shap_samples]
    test_shap_coef = test_coef[:N_shap_samples]
    clean_shap_values = clean_explainer.shap_values(test_shap)
    # Save it because it sometimes crashes and it takes forever to run
    np.save('clean_shap', clean_shap_values)
    np.save('test_shap', test_shap)
    np.save('test_shap_coef', test_shap_coef)

    ### Show the Summary Plot for a given Actuator Command and a set of Wavelength Channels
    j_act = 1
    # Loop over the last 2 channels (Nominal and Defocused, longest wavelength)
    for k_chan in [-2, -1]:
        shap_val_chan = [x[:, :, :, k_chan].reshape((N_shap_samples, pix*pix)) for x in clean_shap_values]
        features_chan = test_shap[:, :, :, k_chan].reshape((N_shap_samples, pix*pix))

        shap.summary_plot(shap_values=shap_val_chan[j_act], features=features_chan,
                          feature_names=pix_label)

    ### Dependence plot
    for k in range(3*pix):
        f, ax = plt.subplots(figsize=(10, 10))
        ind = pix*pix//2 - pix + k
        shap.dependence_plot(ind=ind, shap_values=shap_val_chan[j_act], features=features_chan,
                             xmin=-3/100, xmax=0.5,feature_names=pix_label, ax=ax)
        f.savefig(pix_label[ind])
        plt.close(f)


    act_coef = np.zeros(N_act)
    P0, _s0 = PSF.compute_PSF(0*act_coef, wave_idx=9)

    act_coef[j_act] = 1.0
    P, _s = PSF.compute_PSF(act_coef, wave_idx=9)
    plt.figure()
    plt.imshow(P - P0, cmap='bwr', origin='lower')
    plt.colorbar()
    plt.title('Differential PSF | Actuator #%d' %j_act)
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.show()

    # ================================================================================================================ #
    #                                           SHAP Values - NOISY Images                                             #
    # ================================================================================================================ #



    ### CAUTION! You probably want to do some Memory Cleanup before creating such a large training set

    import sys

    for var, obj in locals().items():
        print(var, sys.getsizeof(obj))

    RMS_READ = 1./500
    N_copies = 5

    read_train_PSF, read_train_coef = readout_noise_images(training_PSF, training_coef, RMS_READ, N_copies)
    read_test_PSF, read_test_coef = readout_noise_images(test_PSF, test_coef, RMS_READ, N_copies)

    noisy_model = create_model(waves=N_WAVES, name='SHAP_NOISY')

    train_history = noisy_model.fit(x=read_train_PSF, y=read_train_coef,
                                      validation_data=(read_test_PSF, read_test_coef),
                                      epochs=10, batch_size=32, shuffle=True, verbose=1)

    N_background = 250
    N_shap_samples = 250

    noisy_background = read_train_PSF[np.random.choice(read_train_PSF.shape[0], N_background, replace=False)]
    noisy_explainer = shap.DeepExplainer(noisy_model, noisy_background)

    # Select only the first N_shap from the test set
    read_test_shap = read_train_PSF[:N_shap_samples]
    read_test_shap_coef = read_test_coef[:N_shap_samples]
    noisy_shap_values = noisy_explainer.shap_values(read_test_shap)
    # Save it because it sometimes crashes and it takes forever to run
    np.save('noisy_shap', noisy_shap_values)
    np.save('read_test_shap', read_test_shap)
    np.save('read_test_shap_coef', read_test_shap_coef)


    N_cases = 250
    test_shap = read_test_PSF[:N_cases]
    test_shap_coef = read_test_coef[:N_cases]
    shap_values = e.shap_values(test_shap)
    np.save('shap', shap_values)
    n_shap = len(shap_values)

    shap.summary_plot(shap_values=shap_val_chan[j_act], features=features_chan,
                      feature_names=pix_label, plot_type='bar')

    shap.summary_plot(shap_values=shap_val_chan, features=features_chan,
                      feature_names=pix_label, plot_type='bar')


    ##
    # shap.dependence_plot
    ##


    j_act = 0
    for i_exa in np.arange(10 + 5*4, 10 + 5*4 + 5, 1):
        # i_exa = 1
        # print(test_shap_coef[i_exa, j_act])
        cmap = 'hot'

        k = -1
        chan = test_shap[i_exa,:,:,2*k]
        coef = test_shap_coef[i_exa, j_act]

        shap_chan = shap_values[j_act][i_exa, :, :, 2*k]
        print(coef)
        smax = max(-np.min(shap_chan), np.max(shap_chan))

        chan_foc = test_shap[i_exa,:,:,2*k+1]
        shap_chan_foc = shap_values[j_act][i_exa, :, :, 2*k+1]
        smax_foc = max(-np.min(shap_chan_foc), np.max(shap_chan_foc))

        maxmax = max(smax, smax_foc)

        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        im1 = ax1.imshow(chan, cmap=cmap)
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(r'Nominal PSF')

        im2 = ax2.imshow(np.log10(np.abs(chan)), cmap=cmap)
        plt.colorbar(im2, ax=ax2)
        ax2.set_title(r'Nominal PSF [log10]')

        im3 = ax3.imshow(shap_chan, cmap='bwr')
        im3.set_clim(-smax, smax)
        ax3.set_title(r'SHAP map [Actuator #%d]' %j_act)
        plt.colorbar(im3, ax=ax3)

        # --------

        im4 = ax4.imshow(chan_foc, cmap=cmap)
        ax4.set_title(r'Defocused PSF')
        plt.colorbar(im4, ax=ax4)

        im5 = ax5.imshow(np.log10(np.abs(chan_foc)), cmap=cmap)
        ax5.set_title(r'Defocused PSF [log10]')
        plt.colorbar(im5, ax=ax5)

        im6 = ax6.imshow(shap_chan_foc, cmap='bwr')
        im6.set_clim(-smax_foc, smax_foc)
        ax6.set_title(r'SHAP map [Actuator #%d]' %j_act)
        plt.colorbar(im6, ax=ax6)

    plt.show()

    ###
    i_exa = 3

    cmap = 'hot'
    f, (ax1, ax2) = plt.subplots(1, 2)

    chan = test_shap[i_exa, :, :, -2]
    im4 = ax1.imshow(chan, cmap=cmap)
    ax1.set_title(r'Nominal PSF')

    im5 = ax2.imshow(np.log10(np.abs(chan)), cmap=cmap)
    ax2.set_title(r'Nominal PSF [log10]')


    f, (ax1, ax2) = plt.subplots(1, 2)

    chan_foc = test_shap[i_exa, :, :, -1]
    im4 = ax1.imshow(chan_foc, cmap=cmap)
    ax1.set_title(r'Defocused PSF')

    im5 = ax2.imshow(np.log10(np.abs(chan_foc)), cmap=cmap)
    ax2.set_title(r'Defocused PSF [log10]')



    n_rows = 5
    n_columns = 8
    f, axes = plt.subplots(n_rows, n_columns)
    for i in range(n_rows):
        for j in range(n_columns):
            k = n_columns * i + j
            shap_chan = shap_values[k][i_exa, :, :, -1]
            smax = max(-np.min(shap_chan), np.max(shap_chan))

            ax = plt.subplot(n_rows, n_columns, k + 1)
            im3 = ax.imshow(shap_chan, cmap='bwr')
            im3.set_clim(-smax, smax)
            # ax.set_title(r'SHAP map [Actuator #%d]' % k)
            # plt.colorbar(im3, ax=ax)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

    shap.decision_plot(e.expected_value[0], shap_val_chan[0][:25], features_chan[:25],
                       feature_order='hclust',
                       feature_display_range=slice(None, -25, -1),
                       ignore_warnings=True)

    shap.image_plot(shap_values, -test_PSF[1:5])

    row_index = 1
    shap.multioutput_decision_plot(list(e.expected_value), shap_val_chan,
                                   row_index=row_index,
                                   feature_names=pix_label,
                                   legend_labels=['Act %d' %x for x in range(N_act)],
                                   legend_location='lower right',
                                   feature_display_range=slice(None, -900, -1),
                                   ignore_warnings=True)




    def percent_mat(residual):
        mat = np.empty((N_act, N_act))
        for i in range(5):
            ri = residual[:, i]
            for j in np.arange(i+1, 5):
                rj = residual[:, j]
                ratio = rj / ri
                plt.scatter(ri, rj, s=3)
                plt.show()
    plt.ylim([-1, 1])
    percent_mat(residual)
    plt.show()

    cov_mat = np.cov(residual.T)
    covm = max(-np.min(cov_mat), np.max(cov_mat))

    plt.figure()
    plt.imshow(cov_mat, cmap='bwr')
    plt.clim(-covm, covm)
    plt.colorbar()
    plt.xlabel('Actuator')
    plt.ylabel('Actuator')
    plt.show()

    plt.figure()
    plt.imshow(dist_mat, cmap='Reds')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.grid(True)
    # plt.axvline(delta0, 0.5, 1, color='black', linestyle='--')
    plt.scatter(dist_mat.flatten(), cov_mat.flatten(), s=5)
    # plt.axhline(0, color='black', linestyle='--')

    plt.ylim([-1.1*covm, 1.1*covm])
    plt.xlabel(r'Actuator distance [1 / D]')
    plt.show()

    """ Is it even worth it to use multiwavelength """

    ### Train with 3 channels
    rms0, rms = [], []

    # for N_chan in [2, training_PSF.shape[-1], -2]:
    for N_chan in [2, training_PSF.shape[-1], -2]:
        input_shape = (pix, pix, 2,)
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(N_act))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')

        train_history = model.fit(x=training_PSF[:,:,:,-2:], y=training_coef, validation_data=(test_PSF[:,:,:,-2:], test_coef),
                                  epochs=25, batch_size=32, shuffle=True, verbose=1)

        guess = model.predict(test_PSF[:,:,:,-2:])
        residual = test_coef - guess

        _rms0, _rms = [], []
        rbf_mat = PSF.RBF_mat[0]
        pupil_mask = PSF.pupil_masks[0]
        for k in range(test_PSF.shape[0]):
            phase = np.dot(rbf_mat, residual[k])
            phase_f = phase[pupil_mask]
            _rms.append(np.std(phase_f))
            _rms0.append(np.std(np.dot(rbf_mat, test_coef[k])[pupil_mask]))
        rms.append(_rms)

    plt.figure()
    plt.hist(_rms0, bins=20, histtype='step', label='Initial')
    # plt.hist(rms[0], bins=20, histtype='step', label='1 Channel')
    plt.hist(rms[1], bins=20, histtype='step', label='5 Channels')
    plt.hist(rms[2], bins=20, histtype='step', label='Last Channel [Long]')
    plt.xlim([0, 1.25])
    plt.xlabel(r'RMS wavefront [$\lambda$]')
    plt.legend()
    plt.show()



    N_channels = training_PSF.shape[-1]
    input_shape = (pix, pix, N_channels,)
    vals = []
    plt.figure()
    for k in range(training_PSF.shape[0] // 1000):
        print(k)
        model = Sequential()
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(N_act))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Slice Channels
        train_wave = training_PSF[:(k+1)*1000]
        # test_wave = test_PSF[k*N_test : (k+1)*N_test]
        train_c = training_coef[:(k+1)*1000]
        # test_c = test_coef[k*N_test : (k+1)*N_test]

        train_history = model.fit(x=train_wave, y=train_c, validation_data=(test_PSF, test_coef),
                                  epochs=25, batch_size=32, shuffle=True, verbose=1)
        val_hist = train_history.history['val_loss']
        vals.append(np.mean(val_hist[-5:]))
        plt.plot(val_hist, label=k+1)
    plt.legend('Batches')
    plt.show()

    n_ex = np.arange(1, 26)
    s = vals[0] / np.sqrt(n_ex)
    plt.figure()
    plt.plot(n_ex, vals)
    plt.plot(n_ex, s)
    plt.xlabel(r'x1000 training examples)')
    plt.ylabel(r'Final validation loss')
    plt.show()



    """ What if we include noise in the training? """

    training_PSF, test_PSF, training_coef, test_coef = load_dataset(N_batches=2, load_waves=10)

    RMS_READ = 1./500
    N_copies = 5
    read_train_PSF, read_train_coef = readout_noise_images(training_PSF, training_coef, RMS_READ, N_copies)
    read_test_PSF, read_test_coef = readout_noise_images(test_PSF, test_coef, RMS_READ, N_copies)

    def select_waves(train_dataset, test_dataset, M_waves=2):
        """
        Out of datacubes that cover [lambda_1, ..., lambda_N]
        with a total of N waves
        slice the waves to get a certain number of Wavelengths (M_waves)
        covering that range

        The problem is that in the datacubes, we organize the channels as
        [lambda_1 (nominal), lambda_1 (defocus), ..., lambda_j (nom), lambda_j (foc), ...]
        which makes it harder than slicing like [wave1 : waveN : n]
        :param train_dataset:
        :param test_dataset:
        :return:
        """
        if M_waves % 2 != 0:
            raise AssertionError('M_waves should be even')

        print("\nSelecting %d wavelengths" % M_waves)
        data_list = []
        for data in [train_dataset, test_dataset]:

            # If we want 4 waves we do: Lambda_1, Lambda_2 & Lambda_(N-1), Lambda_N
            first_chunk = data[:, :, :, :M_waves]
            last_chunck = data[:, :, :, -M_waves:]
            new_data = np.concatenate([first_chunk, last_chunck], axis=-1)
            print(new_data.shape)
            data_list.append(new_data)

        return data_list

    select_waves(read_train_PSF, read_test_PSF, M_waves=4)

    def select_specific_range(train_dataset, test_dataset, which_range):
        print("\nSelecting ", which_range)
        data_list = []
        for data in [train_dataset, test_dataset]:
            new_data = []
            for k in which_range:
                print("\nWavelength: ", k)
                chunk = data[:, :, :, k:k+2]
                print(chunk.shape)
                new_data.append(chunk)
            new_data = np.concatenate(new_data, axis=-1)
            print(new_data.shape)
            data_list.append(new_data)

        return data_list

    _a, _b = select_specific_range(read_train_PSF, read_test_PSF, [0, 5, 7, 9])


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
        for k in range(N_test):
            phase0 = np.dot(rbf_mat, test_coef[k])
            phase0_f = phase0[pupil_mask]
            rms0.append(np.std(phase0_f))
        guessed_coef = []

        # for waves in list_waves:
        # for waves in [2, 4, 6, 8, 10, 16, 20]:
        for waves in [[0, 1, 2, 3, 4],
                      [0, 2, 4, 6, 9],
                      [0, 4, 7, 10, 14],
                      [0, 5, 9, 14, 19]]:
            # print("\nConsidering %d Wavelengths" %waves)

            # N_channels = 2 * waves
            N_channels = 2 * 5
            input_shape = (pix, pix, N_channels,)
            model = Sequential()
            model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
            # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(Conv2D(8, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(N_act))
            model.summary()
            model.compile(optimizer='adam', loss='mean_squared_error')
            # Slice Channels


            # train_wave = training_PSF[:, :, :, :N_channels]
            # test_wave = test_PSF[:, :, :, :N_channels]
            # train_wave, test_wave = select_waves(training_PSF, test_PSF, M_waves=waves)
            train_wave, test_wave = select_specific_range(training_PSF, test_PSF, which_range=waves)

            train_history = model.fit(x=train_wave, y=training_coef, validation_data=(test_wave, test_coef),
                                      epochs=10, batch_size=32, shuffle=True, verbose=1)
            guess = model.predict(test_wave)
            guessed_coef.append(guess)
            residual = test_coef - guess

            # Check the RMS wavefront error for each case
            _rms = []
            for k in range(N_test):
                phase = np.dot(rbf_mat, residual[k])
                phase_f = phase[pupil_mask]
                _rms.append(np.std(phase_f))
            rms.append(_rms)

        return guessed_coef, rms0, rms

    def batch_noise_testing(PSF_model, training_PSF, test_PSF, training_coef, test_coef, N_copies=3):

        N_iter = 10     # How many times to create a new instance of Noisy data
        N_waves = training_PSF.shape[-1] // 2
        list_waves = list(np.arange(1, N_waves + 1))
        rms0, rms = [], []
        rbf_mat = PSF_model.RBF_mat[0]
        pupil_mask = PSF_model.pupil_masks[0]
        for waves in list_waves:
            print("\nConsidering %d Wavelengths" %waves)

            N_channels = 2 * waves
            input_shape = (pix, pix, N_channels,)
            model = Sequential()
            model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
            model.add(Dropout(0.15))
            # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(Dropout(0.15))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(Dropout(0.15))
            model.add(Conv2D(8, (3, 3), activation='relu'))
            model.add(Dropout(0.15))
            model.add(Flatten())
            model.add(Dropout(0.15))
            model.add(Dense(N_act))
            model.summary()
            model.compile(optimizer='adam', loss='mean_squared_error')
            # Slice Channels
            train_wave = training_PSF[:, :, :, :N_channels]
            test_wave = test_PSF[:, :, :, :N_channels]

            for k in range(N_iter):
                ### Include Noise
                read_train_PSF, read_train_coef = readout_noise_images(train_wave, training_coef, RMS_READ, N_copies)
                read_test_PSF, read_test_coef = readout_noise_images(test_wave, test_coef, RMS_READ, N_copies)

                train_history = model.fit(x=read_train_PSF, y=read_train_coef, validation_data=(read_test_PSF, read_test_coef),
                                          epochs=2, batch_size=32, shuffle=True, verbose=1)

            guess = model.predict(read_test_PSF)
            guessed_coef.append(guess)
            residual = read_test_coef - guess


            _rms, _rms0 = [], []
            for k in range(read_test_PSF.shape[0]):
                phase = np.dot(rbf_mat, residual[k])
                phase_f = phase[pupil_mask]
                phase0 = np.dot(rbf_mat, read_test_coef[k])
                phase0_f = phase0[pupil_mask]
                _rms.append(np.std(phase_f))
                _rms0.append(np.std(phase0_f))
            rms.append(_rms)
            rms0.append(_rms0)

        return guessed_coef, rms0, rms

    ### Batch train
    _gc, rms0_batch, rms_batch = batch_noise_testing(PSF, training_PSF, test_PSF, training_coef, test_coef, N_copies=3)

    ### Check the performance on noisy data
    guessed_coef, rms0, rms = test_models(PSF, read_train_PSF, read_test_PSF,
                                          read_train_coef, read_test_coef)

    n_wave = training_PSF.shape[-1] // 2
    # list_waves = list(np.arange(1, n_wave + 1))
    list_waves = [2, 4, 6, 8, 10, 16, 20]

    # plot_waves = n_wave
    # plot_waves = len(list_waves)
    list_waves = [1.875, 2.250, 2.625, 3.00]
    plot_waves = 4
    n_columns = 2
    n_rows = 2
    f, axes = plt.subplots(n_rows, n_columns)
    mus = []
    rmses = []
    for i in range(n_rows):
        for j in range(n_columns):
            k = n_columns * i + j
            if k >= plot_waves:
                break
            rms_wave = rms[k]
            # rms_wave = rms_batch[k]
            avg_rms = np.mean(rms_wave)
            med_rms = np.median(rms_wave)
            std_rms = np.std(rms_wave)
            mus.append(med_rms)
            rmses.append(std_rms)
            print(k)
            ax = plt.subplot(n_rows, n_columns, k + 1)
            ax.hist(rms0, bins=20, color='lightgreen')
            ax.hist(rms_wave, bins=20, color='darkturquoise')
            ax.axvline(med_rms, linestyle='--', color='black')
            ax.set_ylim([0, 700])
            # ax.set_title(r'Channels %d | WFE %.3f ($\sigma$=%.3f)' % (list_waves[k], avg_rms, std_rms))
            ax.set_title(r'$\lambda$ range 1.50-%.3f $\mu m$ | WFE %.3f ($\sigma$=%.3f)' % (list_waves[k], avg_rms, std_rms))
            ax.set_xlim([0, 1.25])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_rows - 1:
                ax.get_xaxis().set_visible(True)
                ax.set_xlabel(r'RMS Wavefront [$\lambda$]')
            if j == 0:
                ax.get_yaxis().set_visible(True)

    ax = plt.subplot(n_rows, n_columns, n_rows*n_columns)
    ax = plt.subplot(1, 1, 1)
    ax.errorbar(list_waves, mus, yerr=rmses, fmt='o')
    ax.set_xlabel('Waves considered')
    ax.set_title(r'RMS Wavefront [$\lambda$]')
    ax.set_xticks(list_waves)
    # ax.set_ylim([0, 0.40])
    plt.show()



    ### Does it saturate because of the defocus range?
    # We put a defocus in [nm] that affects the PSF at each wavelength different
    # For very long wavelengths, the defocus makes almost no difference in intensity
    # Could it be that at such point we do not gain from "diversity" but from error statistics?
    # more samples of the readout noise...

    """ Impact of Defocus range """

    " (1) Show how the defocus affects the PSF "

    training_PSF, test_PSF, training_coef, test_coef = load_dataset(N_batches=2, load_waves=N_WAVES)

    k_train = 2
    f, axes = plt.subplots(2, 3)

    for i, k in enumerate([0, -2]):

        PSF_nom = training_PSF[k_train, :, :, k]
        PSF_foc = training_PSF[k_train, :, :, k+1]
        res = PSF_nom - PSF_foc
        rm = max(-np.min(res), np.max(res))

        ax = plt.subplot(2, 3, 3*i + 1)
        img = ax.imshow(PSF_nom)
        ax.set_title('Nominal PSF')
        plt.colorbar(img)

        ax = plt.subplot(2, 3, 3*i + 2)
        img = ax.imshow(PSF_foc)
        ax.set_title('Defocus PSF')
        plt.colorbar(img)

        ax = plt.subplot(2, 3, 3*i + 3)
        img = ax.imshow(res, cmap='bwr')
        ax.set_title('Difference [Nominal - Defocus]')
        plt.colorbar(img)
        img.set_clim(-rm, rm)

    plt.show()



    def test_defocus(PSF_model):


        # Evaluate the Initial RMS and Wavefronts
        rbf_mat = PSF_model.RBF_mat[0]
        pupil_mask = PSF_model.pupil_masks[0]
        rms = []
        guessed_coef = []
        for focus in [1.0, 1.25, 1.5, 1.75, 2.0]:
            print("\nTesting with a Defocus of %.2f" %focus)
            training_PSF, test_PSF, training_coef, test_coef = generate_training_set(PSF_model, 100, 100, foc=focus)
            read_train_PSF, read_train_coef = readout_noise_images(training_PSF, training_coef, RMS_READ, N_copies=3)
            N_channels = training_PSF.shape[-1]
            del training_PSF
            del training_coef
            read_test_PSF, read_test_coef = readout_noise_images(test_PSF, test_coef, RMS_READ, N_copies=3)
            del test_PSF
            del test_coef

            # print("\nConsidering %d Wavelengths" %waves)

            # N_channels = 2 * waves

            input_shape = (pix, pix, N_channels,)
            print(input_shape)
            model = Sequential()
            model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
            # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(Conv2D(8, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(N_act))
            model.summary()
            model.compile(optimizer='adam', loss='mean_squared_error')

            train_history = model.fit(x=read_train_PSF, y=read_train_coef, validation_data=(read_test_PSF, read_test_coef),
                                      epochs=10, batch_size=32, shuffle=True, verbose=1)
            guess = model.predict(read_test_PSF)
            guessed_coef.append(guess)
            residual = read_test_coef - guess

            if focus == 1.0:
                rms0 = []
                for k in range(read_test_PSF.shape[0]):
                    phase0 = np.dot(rbf_mat, read_test_coef[k])
                    phase0_f = phase0[pupil_mask]
                    rms0.append(np.std(phase0_f))

            # Check the RMS wavefront error for each case
            _rms = []
            for k in range(read_test_PSF.shape[0]):
                phase = np.dot(rbf_mat, residual[k])
                phase_f = phase[pupil_mask]
                _rms.append(np.std(phase_f))
            rms.append(_rms)

        return guessed_coef, rms0, rms

    guessed_coef, rms0_focus, rms_focus = test_defocus(PSF)

    import shap

    shap_values = shap.TreeExplainer(model).shap_values(read_train_PSF)

    list_focus = [1.0, 1.25, 1.5, 1.75, 2.0]
    plot_waves = len(list_focus)
    n_columns = 3
    n_rows = 2
    f, axes = plt.subplots(n_rows, n_columns)
    mus = []
    rmses = []
    for i in range(n_rows):
        for j in range(n_columns):
            k = n_columns * i + j
            if k >= plot_waves:
                break
            rms_wave = rms[k]
            # rms_wave = rms_batch[k]
            avg_rms = np.mean(rms_wave)
            med_rms = np.median(rms_wave)
            std_rms = np.std(rms_wave)
            mus.append(med_rms)
            rmses.append(std_rms)
            print(k)
            ax = plt.subplot(n_rows, n_columns, k + 1)
            ax.hist(rms0, bins=20, color='lightgreen')
            ax.hist(rms_wave, bins=20, color='darkturquoise')
            ax.axvline(med_rms, linestyle='--', color='black')
            # ax.set_ylim([0, 700])
            # ax.set_title(r'Channels %d | WFE %.3f ($\sigma$=%.3f)' % (list_waves[k], avg_rms, std_rms))
            ax.set_title(r'$\lambda$ range 1.50-%.3f $\mu m$ | WFE %.3f ($\sigma$=%.3f)' % (list_waves[k], avg_rms, std_rms))
            ax.set_xlim([0, 1.25])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_rows - 1:
                ax.get_xaxis().set_visible(True)
                ax.set_xlabel(r'RMS Wavefront [$\lambda$]')
            if j == 0:
                ax.get_yaxis().set_visible(True)

    ax = plt.subplot(n_rows, n_columns, n_rows*n_columns)
    ax = plt.subplot(1, 1, 1)
    ax.errorbar(list_waves, mus, yerr=rmses, fmt='o')
    ax.set_xlabel('Waves considered')
    ax.set_title(r'RMS Wavefront [$\lambda$]')
    ax.set_xticks(list_waves)
    # ax.set_ylim([0, 0.40])
    plt.show()




    """ Good Question """
    ### Is the improvement because of HAVING EXTRA IMAGES
    # or is it because we have one image at a higher sampling??

    # To test that, train only on the longest wavelength

    N_channels = 2
    input_shape = (pix, pix, N_channels,)
    print(input_shape)
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
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
                              epochs=10, batch_size=32, shuffle=True, verbose=1)

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
