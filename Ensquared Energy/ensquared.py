"""

Investigate the effect of RMS NCPA on Ensquared Energy for each SPAXEL SCALE

"""

import numpy as np
from numpy.linalg import norm as norm
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import zern_core as zern


# PARAMETERS
Z = 1.25                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 50                    # Pixels to crop the PSF
N_PIX = 1024                 # Pixels for the Fourier arrays
minPix, maxPix = (N_PIX + 1 - pix) // 2, (N_PIX + 1 + pix) // 2

# SPAXEL SCALE
wave = 1.0                 # 1 micron
SPAXEL_MAS = 0.5            # mas
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


RHO_APER = rho_spaxel_scale(spaxel_scale=SPAXEL_MAS, wavelength=wave)
RHO_OBSC = 0.3*RHO_APER     # Central obscuration (30% of ELT)
check_spaxel_scale(RHO_APER, wave)

def triangular_numbers(N_levels):
    """
    Returns a Dictionary for the triangular numbers
    associated with the Zernike pyramid

    key: Zernike Level
    value: Total number of Zernikes up to a "key" layer
    """
    zernike_rows = list(np.arange(1, N_levels + 1))
    triangular = {}
    for i, zernike_per_row in enumerate(zernike_rows):
        total = np.sum(zernike_rows[:i+1])
        triangular[zernike_per_row] = total

    return triangular


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
    N_in_D = 2*RHO_APER/delta
    print('%.2f actuators in D' %N_in_D)
    max_freq = N_in_D / 2                   # Max spatial frequency we can sense
    xx, yy = np.meshgrid(x0, x0)
    x_f = xx.flatten()
    y_f = yy.flatten()

    act = []
    for x_c, y_c in zip(x_f, y_f):
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        if r < 0.97 * rho_aper and r > 1.05 * rho_obsc:
            act.append([x_c, y_c])
    total_act = len(act)
    print('Total Actuators: ', total_act)
    return [act, delta], max_freq

def plot_actuators(centers):
    N_act = len(centers[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circ1 = Circle((0,0), RHO_APER, linestyle='--', fill=None)
    circ2 = Circle((0,0), RHO_OBSC, linestyle='--', fill=None)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    for c in centers[0]:
        ax.scatter(c[0], c[1], color='red', s=3)
    ax.set_aspect('equal')
    plt.xlim([-RHO_APER, RHO_APER])
    plt.ylim([-RHO_APER, RHO_APER])
    plt.title('%d actuators' %N_act)

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
        matrix[:, :, k] = pupil * np.exp(-r2 / (0.75 * delta) ** 2)

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

        pupil_function = self.pupil_mask * np.exp(2 * np.pi * 1j * phase)
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

def ensquared_one_pix(array, new_scale=40, plot=True):

    n = int(new_scale // SPAXEL_MAS)
    minPix, maxPix = (pix + 1 - n) // 2, (pix + 1 + n) // 2
    ens = array[minPix:maxPix, minPix:maxPix]
    # print(ens.shape)
    energy = np.sum(ens)

    if plot:
        mapp = 'viridis'
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = plt.subplot(1, 2, 1)
        square = Rectangle((minPix-0.5, minPix-0.5), n, n, linestyle='--', fill=None, color='white')
        ax1.add_patch(square)
        img1 = ax1.imshow(array, cmap=mapp)
        ax1.set_title('%.1f mas pixels' % (SPAXEL_MAS))
        img1.set_clim(0, 1)
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 2, 2)
        img2 = ax2.imshow(ens, cmap=mapp)
        ax2.set_title('%d mas window' %new_scale)
        img1.set_clim(0, 1)
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

    return energy


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    """ Zernike """

    x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x0, x0)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    pupil = (rho <= RHO_APER) & (rho >= RHO_OBSC)
    rho = rho[pupil]
    theta = theta[pupil]

    triang = triangular_numbers(N_levels=15)
    Level = 11
    exp = Level - 1
    N_zern = triang[Level]
    _coef = np.zeros(N_zern)
    z = zern.ZernikeNaive(mask=pupil)
    _phase = z(coef=_coef, rho=rho/RHO_APER, theta=theta, normalize_noll=False, mode='Jacobi',
               print_option=None)
    model_matrix_flat = z.model_matrix
    model_matrix = zern.invert_model_matrix(z.model_matrix, pupil)
    model_matrix = model_matrix[:, :, 3:]   # remove piston and tilts
    model_matrix_flat = model_matrix_flat[:, 3:]
    N_zern = model_matrix.shape[-1]

    PSF_zern = PointSpreadFunction([model_matrix, pupil, model_matrix_flat])
    c_zern = np.random.uniform(-0.10, 0.10, size=N_zern)
    phase0 = np.dot(model_matrix, c_zern)
    p0 = min(phase0.min(), -phase0.max())

    # plt.figure()
    # plt.imshow(phase0, extent=(-1, 1, -1, 1), cmap='coolwarm')
    # plt.colorbar()
    # plt.clim(p0, -p0)
    # plt.show()
    #
    # p0, s0 = PSF_zern.compute_PSF(c_zern)
    # EE0 = ensquared_one_pix(p0, new_scale=20)         # Perfect EE for central pixel
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(p0)
    # plt.colorbar()
    # plt.show()

    colors = ['blue', 'green', 'red']
    scales = [4, 10, 20]

    for scale, color in zip(scales, colors):
        N_trials = 10
        N_rms = 50
        data = np.zeros((2, N_trials*N_rms))
        amplitudes = np.linspace(0.0, 0.08, N_rms)
        i = 0
        # scale = 20
        p0, s0 = PSF_zern.compute_PSF(np.zeros(N_zern))
        EE0 = ensquared_one_pix(p0, new_scale=scale, plot=False)
        # plt.figure()
        for amp in amplitudes:
            print(amp)
            for k in range(N_trials):
                c_act = np.random.uniform(-amp, amp, size=N_zern)
                phase_flat = np.dot(model_matrix_flat, c_act)
                rms = wave * 1e3 * np.std(phase_flat)
                p, s = PSF_zern.compute_PSF(c_act)
                EE = ensquared_one_pix(p, new_scale=scale, plot=False)
                dEE = EE / EE0 * 100
                data[:, i] = [rms, dEE]
                i += 1

        plt.scatter(data[0], data[1], color=color, s=3, label=scale)
    # plt.xlabel(r'RMS wavefront [$\lambda$]')
    plt.xlabel(r'RMS wavefront [nm]')
    plt.ylabel(r'Relative Encircled Energy [per cent]')
    plt.legend(title='Spaxel [mas]', loc=3)
    plt.ylim([80, 100])
    plt.xlim([0, 125])
    plt.title(r'%d Zernike ($\rho^{%d}$)' % (N_zern, exp))
    plt.savefig('%d Zernike' % N_zern)


    plt.show()




    """ (1) Define the ACTUATORS """

    N_actuators = 100
    centers, MAX_FREQ = actuator_centres(N_actuators)
    print(MAX_FREQ)
    # centers = radial_grid(N_radial=5)
    N_act = len(centers[0])
    plot_actuators(centers)

    rbf_mat = rbf_matrix(centers)        # Actuator matrix

    c_act = np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(rbf_mat[0], c_act)
    p0 = min(phase0.min(), -phase0.max())

    plt.figure()
    plt.imshow(phase0, extent=(-1, 1, -1, 1), cmap='coolwarm')
    plt.colorbar()
    plt.clim(p0, -p0)
    for c in centers[0]:
        plt.scatter(c[0], c[1], color='black', s=3)
    plt.xlim([-RHO_APER, RHO_APER])
    plt.ylim([-RHO_APER, RHO_APER])
    plt.show()

    PSF = PointSpreadFunction(rbf_mat)
    p0, s0 = PSF.compute_PSF(0*c_act)

    EE0 = ensquared_one_pix(p0, new_scale=20)         # Perfect EE for central pixel
    plt.show()

    N_trials = 5
    N_rms = 40
    data = np.zeros((2, N_trials*N_rms))
    amplitudes = np.linspace(0.01, 0.15, N_rms)
    i = 0
    scale = 4
    EE0 = ensquared_one_pix(p0, new_scale=scale, plot=True)
    plt.figure()
    for amp in amplitudes:
        print(amp)
        for k in range(N_trials):
            c_act = np.random.uniform(-amp, amp, size=N_act)
            phase_flat = np.dot(rbf_mat[-1], c_act)
            rms = wave * 1e3 * np.std(phase_flat)
            p, s = PSF.compute_PSF(c_act)
            EE = ensquared_one_pix(p, new_scale=scale, plot=False)
            dEE = EE / EE0 * 100
            data[:, i] = [rms, dEE]
            i += 1

    plt.scatter(data[0], data[1], color='blue', s=3, label=scale)
    # plt.xlabel(r'RMS wavefront [$\lambda$]')
    plt.xlabel(r'RMS wavefront [nm]')
    plt.ylabel(r'Relative Encircled Energy [per cent]')
    plt.legend()
    plt.ylim([80, 100])


    plt.show()

    # Using the Actuator model for the Wavefront leads to almost identical results
    # across 4-10-20 scales... weird


