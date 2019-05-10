"""
Environment for Reinforcement Learning using OpenAI gym and Q-learning
applied to NCPA calibration


"""
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import zern_core as zern

class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    ### Parameters
    rho_aper = 0.25         # Size of the aperture relative to 1.0
    N_pix = 512
    pix = 25                # Number of pixels for the Zoom of the PSF
    minPix, maxPix = (N_pix - pix) // 2, (N_pix + pix) // 2

    def __init__(self, N_zern):

        ### Zernike Wavefront
        x = np.linspace(-1, 1, self.N_pix, endpoint=True)
        xx, yy = np.meshgrid(x, x)
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        self.pupil = rho <= self.rho_aper
        rho, theta = rho[self.pupil], theta[self.pupil]
        zernike = zern.ZernikeNaive(mask=self.pupil)
        _phase = zernike(coef=np.zeros(N_zern + 3), rho=rho/self.rho_aper, theta=theta, normalize_noll=False,
                         mode='Jacobi', print_option='Silent')
        H_flat = zernike.model_matrix[:, 3:]  # remove the piston and tilts
        self.H_matrix = zern.invert_model_matrix(H_flat, self.pupil)

        # Update the number of aberrations to match the dimensions of H
        self.N_zern = self.H_matrix.shape[-1]

        self.PEAK = self.peak_PSF()

    def peak_PSF(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """

        im, strehl = self.compute_PSF(np.zeros(self.N_zern))

        return strehl

    def compute_PSF(self, zern_coef):
        """
        Compute the PSF and the Strehl ratio
        """

        if zern_coef.shape != self.N_zern:      # Zero-pad to match dimensions
            extra_zeros = np.zeros((self.N_zern - zern_coef.shape[0]))
            zern_coef = np.concatenate([zern_coef, extra_zeros])

        phase = np.dot(self.H_matrix, zern_coef)
        pupil_function = self.pupil * np.exp(1j * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2

        try:
            image /= self.PEAK

        except AttributeError:
            # If self.PEAK is not defined, self.compute_PSF will compute the peak
            pass

        strehl = np.max(image)

        return image, strehl

    def plot_PSF(self, zern_coef, i):
        """
        Plot an image of the PSF
        :param zern_coef:
        :param i: iteration (for labelling purposes)
        """

        PSF, strehl = self.compute_PSF(zern_coef)
        PSF_zoom = PSF[self.minPix:self.maxPix, self.minPix:self.maxPix]

        plt.figure()
        plt.imshow(PSF_zoom)
        plt.title('Iter: %d Strehl: %.3f' %(i, strehl))
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

class FocalPlaneSharpening(object):

    def __init__(self, coef0):

        self.N_zern = coef0.shape[0]
        self.PSF = PointSpreadFunction(N_zern=self.N_zern)

    def policy(self, strehls, j, silent=True):

        i_nom = len(strehls) // 2

        # Positive Stroke GOOD and Negative Stroke BAD
        if strehls[-1] > strehls[i_nom] and strehls[i_nom] > strehls[0]:
            if not silent:
                print('Aberration: %d | (+)' %j)
            return 1.

        # Negative Stroke GOOD and Positive Stroke BAD
        if strehls[0] > strehls[i_nom] and strehls[i_nom] > strehls[-1]:
            if not silent:
                print('Aberration: %d | (-)' % j)
            return -1.

        # Both Positive and Negative are BAD
        if strehls[i_nom] > strehls[0] and strehls[i_nom] > strehls[-1]:
            if not silent:
                print('Aberration: %d | (?)' % j)
            return 0.

        # Both Positive and Negative are GOOD
        if strehls[0] > strehls[i_nom] and strehls[-1] > strehls[i_nom]:
            if strehls[0] > strehls[-1]:
                if not silent:
                    print('Aberration: %d | +(-)' % j)
                return -1.
            if strehls[-1] > strehls[0]:
                if not silent:
                    print('Aberration: %d | (+)-' % j)
                return 1.

        else:
            if not silent:
                print('Aberration: %d | (??)' % j)
            return 0.


    def run(self, coef0, stroke=0.01, max_iter=50, threshold=0.90, statistics=False, silent=True):

        ### Initialize the iteration
        self.states = [coef0.copy()]
        self.strehl_evolution = []
        self.images = []


        ### Global Iteration
        for i in range(max_iter):

            ### Evaluate current state
            current_state = self.states[i].copy()
            psf_nom, strehl_nom = self.PSF.compute_PSF(current_state)
            self.strehl_evolution.append(strehl_nom)
            self.images.append(psf_nom[self.PSF.minPix:self.PSF.maxPix, self.PSF.minPix:self.PSF.maxPix])

            if not silent:
                print("--------------------------------------------------")
                print("\nIteration :", i)
                print("Strehl: %.3f" %strehl_nom)

            if strehl_nom > threshold:
                if statistics:
                    self.statistics()
                return self.states, self.strehl_evolution, self.images

            ### Nested iteration over the aberrations
            for j in range(self.N_zern):


                correction = np.zeros(self.N_zern)
                correction[j] = stroke

                minus = current_state - correction
                psf_minus, strehl_minus = self.PSF.compute_PSF(minus)
                plus = current_state + correction
                psf_plus, strehl_plus = self.PSF.compute_PSF(plus)
                strehls = [strehl_minus, strehl_nom, strehl_plus]

                decision = self.policy(strehls, j, silent)
                current_state += decision * correction

            # Update the state list
            updated_state = current_state
            self.states.append(updated_state)

        if statistics:
            self.statistics()

        return self.states, self.strehl_evolution, self.images

    def run_batch(self, rand_coef, options):

        N_runs = rand_coef.shape[0]
        stroke, max_iter, threshold= options

        for i in range(N_runs):
            print("\nRun: ", i)
            states, strehls, _images = self.run(rand_coef[i], stroke, max_iter,
                                                threshold, statistics=False, silent=True)


    def statistics(self):

        n_iters = len(self.strehl_evolution)

        ### Strehl ratio
        plt.figure()
        plt.scatter(range(n_iters), self.strehl_evolution)
        plt.plot(range(n_iters), self.strehl_evolution)
        plt.xlabel('Iteration')
        plt.ylabel('Strehl')

        ### Norm of aberrations
        aberr = [np.linalg.norm(x) for x in self.states]
        plt.figure()
        plt.scatter(range(n_iters), aberr, color='red')
        plt.plot(range(n_iters), aberr, color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Norm |Aberrations|')

        ### PSF images
        n_images = len(self.images)
        M = int(np.sqrt(n_images)) + 1
        plt.figure()
        for i, img in enumerate(self.images):

            ax = plt.subplot(M + 1, M, i + 1)
            plt.imshow(img)
            plt.colorbar()
            plt.clim(vmin=0, vmax=1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title("i:%d (s:%.3f)" %(i, self.strehl_evolution[i]))


if __name__ == "__main__":

    N_zern = 15
    coef = np.random.uniform(-1.2, 1.2, size=N_zern)

    FPS = FocalPlaneSharpening(coef)
    states, strehls, images = FPS.run(coef, stroke=0.05, max_iter=50,
                                      threshold=0.90, statistics=True, silent=True)

    ### Multiple runs
    N_runs = 5
    rand_coef = np.random.uniform(-1.2, 1.2, size=(N_runs, N_zern))
    FPS.run_batch(rand_coef, options=(0.05, 25, 0.90))

    plt.show()




