
import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D
from keras.models import Model, Sequential, Input
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from numpy.linalg import norm as norm

# PARAMETERS
Z = 1.5                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 30                    # Pixels to crop the PSF
N_PIX = 1024                 # Pixels for the Fourier arrays
RHO_APER = 0.5 / 2
RHO_OBSC = 0.15 / 2           # Central obscuration

# SPAXEL SCALE
wave = 1.5                 # 1 micron
ELT_DIAM = 39
MILIARCSECS_IN_A_RAD = 206265000
SPAXEL_RAD = RHO_APER * wave / ELT_DIAM * 1e-6
SPAXEL_MAS = SPAXEL_RAD * MILIARCSECS_IN_A_RAD
print('%.2f mas spaxels at %.2f microns' %(SPAXEL_MAS, wave))

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

def actuator_matrix(centres, alpha=0.75, rho_aper=RHO_APER, rho_obsc=RHO_OBSC):
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
    #TODO: Update the Model to something other than a Gaussian

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

# ==================================================================================================================== #
#                                          Point Spread Function
# ==================================================================================================================== #


class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    N_pix = N_PIX             # Number of pixels for the FFT computations
    minPix, maxPix = (N_pix + 1 - pix) // 2, (N_pix + 1 + pix) // 2

    def __init__(self, RBF_matrices):

        self.N_act = RBF_matrices[0].shape[-1]
        self.RBF_mat = RBF_matrices[0]
        self.pupil_mask = RBF_matrices[1]
        self.RBF_flat = RBF_matrices[2]

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

        phase = np.dot(self.RBF_mat, coef)


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

    def plot_PSF(self, coef, cmap='hot'):
        """
        Plot an image of the PSF
        """
        PSF, strehl = self.compute_PSF(coef)

        plt.figure()
        plt.imshow(PSF, cmap=cmap)
        plt.title('Strehl: %.3f' %strehl)
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)

def generate_PSF_images(PSF_model, eps, N_samples):

    N_act = PSF_model.N_act
    coef = eps * np.random.uniform(low=-1, high=1, size=(N_samples, N_act))
    dataset = np.empty((N_samples, pix, pix))

    for i in range(N_samples):
        if i % 100 == 0:
            print(i)

        im0, _s = PSF_model.compute_PSF(coef[i])
        dataset[i, :, :] = im0

    return dataset, coef

def crop_array(array, crop=25):
    PIX = array.shape[0]
    min_crop = PIX // 2 - crop // 2
    max_crop = PIX // 2 + crop // 2
    array_crop = array[min_crop:max_crop, min_crop:max_crop]
    return array_crop


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    """ (1) Define the ACTUATOR model for the WAVEFRONT """

    N_actuators = 20 * 2
    centers, MAX_FREQ = actuator_centres(N_actuators)
    N_act = len(centers[0])
    plot_actuators(centers)

    rbf_mat = actuator_matrix(centers)        # Actuator matrix

    PSF = PointSpreadFunction(rbf_mat)

    p, _s = PSF.compute_PSF(np.zeros(N_act))
    plt.figure()
    plt.imshow(p)
    plt.show()

    """ Generate Images """
    images, coef = generate_PSF_images(PSF, eps=0.5e-1, N_samples=50)
    plt.figure()
    plt.imshow(images[0])
    plt.colorbar()
    plt.show()

    def construct_F_matrix(PSF):
        H = PSF.RBF_mat
        F_pi = fftshift(fft2(PSF.pupil_mask))[:,:,np.newaxis]
        F_pi_conj = np.conj(F_pi)
        F_H = fftshift(fft2(H, axes=(0, 1)), axes=(0, 1))
        F_H_conj = np.conj(F_H)
        F = np.real(1j * (F_pi_conj * F_H - F_pi * np.conj(F_H))) / PSF.PEAK

        FF = np.real(F_H * F_H_conj) / PSF.PEAK

        F_H2 = fftshift(fft2(H**2, axes=(0, 1)), axes=(0, 1))
        F_H2_conj = np.conj(F_H2)
        FH2 = - np.real(F_pi * (F_H2 + F_H2_conj)) / PSF.PEAK

        return F, FF, FH2

    F, FF, FH2 = construct_F_matrix(PSF)

    k = 1
    dIm = images[k] - p
    d0 = np.dot(F, coef[k])
    d1 = np.dot(FF, coef[k]**2)
    d2 = 0.5 * np.dot(FH2, coef[k]**2 / 2)
    delta = crop_array(d0 + d2, pix)

    error = dIm - delta
    print(np.mean(np.abs(error)))

    m = min(np.min(dIm), -np.max(dIm))
    plt.figure()
    plt.imshow(dIm, cmap='RdBu')
    plt.clim(-m, m)
    plt.colorbar()

    plt.figure()
    plt.imshow(delta, cmap='RdBu')
    plt.clim(-m, m)
    plt.colorbar()


    plt.figure()
    plt.imshow(error, cmap='RdBu')
    plt.clim(-m, m)
    plt.colorbar()
    plt.show()

    print()

    plt.imshow(F[:,:,0])
    plt.colorbar()
    plt.show()