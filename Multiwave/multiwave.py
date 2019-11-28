from __future__ import print_function  # for Python2

import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from keras import models
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from numpy.linalg import norm as norm

# PARAMETERS
Z = 1.75                    # Strength of the aberrations
pix = 30                    # Pixels to crop the PSF
N_PIX = 256
RHO_APER = 0.5
RHO_OBSC = 0.15
N_WAVES = 10
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

# def flat_field_training(N_flats=5, sigma=0.15):

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
    #                              UNCERTAINTY - Bayesian Networks approx with Dropout                                 #
    # ================================================================================================================ #

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
        # model.add(Conv2D(32, (3, 3), activation='relu'))
        # model.add(Dropout(rate=1 - p))
        # model.add(Conv2D(8, (3, 3), activation='relu'))
        # model.add(Dropout(rate=1 - p))
        model.add(Flatten())
        model.add(Dropout(rate=drop_rate))
        model.add(Dense(N_act))
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

    im_bias = draw_actuator_commands(mean_abs_bias_per_act, centers[0])
    plt.imshow(im_bias, cmap='Reds')
    plt.colorbar()
    plt.show()

    ### Try to fit a Gaussian to the Dropout predictions
    from scipy.stats import norm as gaussian
    from scipy.stats import normaltest
    from matplotlib import cm
    # colors = ['red', 'blue', 'black', 'green', 'pink']
    N_show = 10
    colors = cm.coolwarm(np.linspace(0, 1, N_show))
    print("\nThe True Coefficient was within [x] SIGMA of the average prediction")
    k_act = 5
    for i_ex in range(N_show):

        predictions = result[:, i_ex, k_act]
        _mean, _std = avg_pred[i_ex, k_act], unc[i_ex, k_act]

        # Normal Test
        statistic, p_value = normaltest(predictions)
        # print(p_value)

        # Fit a Gaussian profile
        mean, std = gaussian.fit(predictions)

        # True value
        true_value = test_coef[i_ex, k_act]
        sigmas = np.abs(true_value - _mean) / std

        print("%.2f" % sigmas)

        plt.figure()
        plt.hist(predictions, bins=20, histtype='step', color=colors[i_ex])

        plt.axvline(true_value, linestyle='--', color=colors[i_ex])
        plt.xlim([-2, 2])
        plt.xlabel("Actuator Command")
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

    N_models = 10
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
    list_guesses, list_uncertain = [], []
    for k in range(N_models):
        _model = create_model_dropout(waves=N_WAVES, keep_rate=keep_rate)
        train_history = _model.fit(x=training_PSF, y=training_coef,
                                      validation_data=(test_PSF, test_coef),
                                      epochs=30, batch_size=32, shuffle=True, verbose=1)
        list_models.append(_model)
        # Combine the guesses
        f = K.function([_model.layers[0].input, K.learning_phase()],
                       [_model.layers[-1].output])
        # Use the Uncertain predictions
        result, avg_pred, unc = predict_with_uncertainty(f, test_PSF[:500], N_classes=N_act, N_samples=500)
        list_guesses.append(avg_pred)
        list_uncertain.append(unc)

    many_guesses = np.mean(np.concatenate(list_guesses, axis=0), axis=0)
    many_residual = test_coef - many_guesses



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



    background = read_train_PSF[np.random.choice(read_train_PSF.shape[0], 250, replace=False)]
    e = shap.DeepExplainer(model, background)
    # shap_interaction_values = e.shap_interaction_values(background)

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



    act_coef = np.zeros(N_act)
    defocus = np.load('defocus.npy')
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

    def readout_noise_images(dataset, coef, RMS_READ, N_copies=3):
        N_PSF, pix, _pix, N_chan = dataset.shape
        N_act = coef.shape[-1]
        new_data = np.empty((N_copies * N_PSF, pix, pix, N_chan))
        new_coef = np.empty((N_copies * N_PSF, N_act))

        for k in range(N_PSF):
            # if k %100 == 0:
                # print(k)
            PSF = dataset[k].copy()
            coef_copy = coef[k].copy()
            for i in range(N_copies):
                read_out = np.random.normal(loc=0, scale=RMS_READ, size=(pix, pix, N_chan))
                new_data[N_copies * k + i] = PSF + read_out
                new_coef[N_copies * k + i] = coef_copy
        ### Remove clean PSF to save memory
        del dataset
        new_data += 5*RMS_READ
        return new_data, new_coef

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




    # =------------------------------------------


    k_train = 0
    plot_waves = N_WAVES
    f, axes = plt.subplots(plot_waves, 2)
    for i in range(plot_waves):
        ax = plt.subplot(plot_waves, 2, 2*i + 1)
        PSF_nom = training_PSF[k_train, :,:,2*i]
        img = ax.imshow(PSF_nom)
        ax.set_title('Nominal PSF [Wave %.2f]' %waves_ratio[i])

        ax = plt.subplot(plot_waves, 2, 2*i + 2)
        PSF_foc = training_PSF[k_train, :,:,2*i+1]
        img = ax.imshow(PSF_foc)
        ax.set_title('Defocus PSF [Wave %.2f]' %waves_ratio[i])
    plt.show()

    waves_considered = 5
    N_channels = 2*waves_considered
    input_shape = (pix, pix, N_channels,)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(N_act))
    model.summary()

    model = Sequential()
    ks = 7
    model.add(Conv2D(256, kernel_size=(ks, ks), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(128, (ks, ks), activation='relu'))

    model.add(Flatten())
    # model.add(Dense(2 * N_act))
    model.add(Dense(N_act))

    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    # train_history = model.fit(x=training_PSF[:,:,:,:2*waves_considered], y=training_coef,
    #                           validation_data=(test_PSF[:,:,:,:2*waves_considered], test_coef),
    #                           epochs=50, batch_size=32, shuffle=True, verbose=1)
    train_history = model.fit(x=read_train_PSF[:,:,:,:2*waves_considered], y=read_train_coef,
                              validation_data=(read_test_PSF[:,:,:,:2*waves_considered], read_test_coef),
                              epochs=5, batch_size=32, shuffle=True, verbose=1)
    loss_hist = train_history.history['loss']
    val_hist = train_history.history['val_loss']


    # guess = model.predict(test_PSF[:,:,:,:2*waves_considered])
    # residual = test_coef - guess
    guess = model.predict(read_test_PSF[:, :, :, :2 * waves_considered])
    residual = read_test_coef - guess
    print(np.mean(np.abs(residual)))

    #### Hyperparameter Optim
    k_sizes = [4, 5, 6, 7]
    N_filters = [16, 32, 64]


    for Nf1 in [128]:
        for Nf2 in N_filters:
            for ks in k_sizes:

                model = Sequential()
                model.add(Conv2D(Nf1, kernel_size=(ks, ks), strides=(1, 1),
                                 activation='relu',
                                 input_shape=input_shape))
                model.add(Conv2D(Nf2, (ks, ks), activation='relu'))
                model.add(Flatten())
                model.add(Dense(N_act))
                model.compile(optimizer='adam', loss='mean_squared_error')
                train_history = model.fit(x=read_train_PSF[:, :, :, :2 * waves_considered], y=read_train_coef,
                                          validation_data=(
                                          read_test_PSF[:, :, :, :2 * waves_considered], read_test_coef),
                                          epochs=5, batch_size=32, shuffle=True, verbose=0)
                loss_hist = train_history.history['loss']
                val_hist = train_history.history['val_loss']
                print("\nHyperparams: %d Kernel Size | %d Filters 1 | %d Filters 2" % (ks, Nf1, Nf2))
                print(val_hist[-1])
                guess = model.predict(read_test_PSF[:, :, :, :2 * waves_considered])
                residual = read_test_coef - guess
                print(np.mean(np.abs(residual)))




    validation_losses, guessed_coef, rms0, rms, wavefr0, wavefronts = test_models(PSF, training_PSF, test_PSF,
                                                                             training_coef, test_coef)



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
