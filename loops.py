"""

Naive Focal Plane Sharpening

We want to show the difficulties of this method,
because when we vary the intensity of a pair of Zernikes
we can see spurious peaks of Strehl ratio that confuse FPS

"""

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import zern_core as zern


pix = 25                    # Pixels to crop the PSF
N_PIX = 1024                 # Pixels for the Fourier arrays
minPix, maxPix = (N_PIX + 1 - pix) // 2, (N_PIX + 1 + pix) // 2

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


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # SPAXEL SCALE
    SPAXEL_MAS = 4.0    # [mas] spaxel scale at wave0
    RHO_APER = rho_spaxel_scale(spaxel_scale=SPAXEL_MAS, wavelength=WAVE)
    RHO_OBSC = 0.3 * RHO_APER  # Central obscuration (30% of ELT)

    check_spaxel_scale(RHO_APER, WAVE)

    # Define the Pupil Masks and the Rho, Thetha (masked) arrays
    x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x0, x0)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    pupil = (rho <= RHO_APER) & (rho >= RHO_OBSC)
    rho = rho[pupil]
    theta = theta[pupil]

    triang = triangular_numbers(N_levels=15)
    level = 6
    # Calculate how many Zernikes we need to have up to a certain Radial level
    exp = level - 1                         # Exponent of the Zernike radial polynomial
    N_zern = triang[level]                  # How many Zernikes in total to ask zern to use
    _coef = np.zeros(N_zern)
    z = zern.ZernikeNaive(mask=pupil)
    _phase = z(coef=_coef, rho=rho/RHO_APER, theta=theta, normalize_noll=False, mode='Jacobi',
               print_option=None)
    model_matrix_flat = z.model_matrix
    model_matrix = zern.invert_model_matrix(z.model_matrix, pupil)

    model_matrix = model_matrix[:, :, 3:]   # remove piston and tilts
    model_matrix_flat = model_matrix_flat[:, 3:]

    N_zern = model_matrix.shape[-1]  # Update the N_zern after removing Piston and Tilts

    PSF_zern = PointSpreadFunction([model_matrix, pupil, model_matrix_flat])

    """ Random Wavefront Map """
    Z = 0.2
    coef = np.random.uniform(low=-Z, high=Z, size=N_zern)

    # Select which pair of aberrations you want to investigate
    # [0] Astigmatism and [1] Defocus
    i_aber, j_aber = 0, 1
    coef[i_aber], coef[j_aber] = 0.0, 0.0
    im0, s0 = PSF_zern.compute_PSF(coef)

    # plt.figure()
    # plt.imshow(im0)
    # plt.colorbar()
    # plt.show()

    # Loop over the pair of aberrations
    # Calculate how much the Strehl ratio varies when we
    # change the aberrations
    N_samples = 21
    PEAKS = np.empty((N_samples, N_samples))
    values = np.linspace(-Z, Z, N_samples, endpoint=True)
    for i in range(N_samples):
        print(i)
        astig = values[i]
        coef[i_aber] = astig
        for j in range(N_samples):
            defocus = values[j]
            coef[j_aber] = defocus
            im, s = PSF_zern.compute_PSF(coef)
            PEAKS[i, j] = s
    xx, yy = np.meshgrid(values, values)

    # Contour Plot of the aberrations
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(im0)
    ax1.set_xlabel(r'Pixel [ ]')
    ax1.set_ylabel(r'Pixel [ ]')
    plt.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 2, 2)
    img = plt.contourf(xx, yy, PEAKS, levels=25)
    ax2.axhline(0.0, color='black', linestyle='-.', alpha=0.75)
    ax2.axvline(0.0, color='black', linestyle='-.', alpha=0.75)
    ax2.set_xlabel(r'Aberration 1 [$\lambda$]')
    ax2.set_ylabel(r'Aberration 2 [$\lambda$]')
    ax2.set_aspect('equal')
    plt.colorbar(img, ax=ax2, orientation='horizontal')
    plt.show()

    plt.figure()
    # plt.imshow(PEAKS, extent=[-Z, Z, -Z, Z], origin='lower')
    img = plt.contourf(xx, yy, PEAKS, levels=25)
    plt.axhline(0.0, color='black', linestyle='-.', alpha=0.75)
    plt.axvline(0.0, color='black', linestyle='-.', alpha=0.75)
    plt.xlabel(r'Aberration 1 [$\lambda$]')
    plt.ylabel(r'Aberration 2 [$\lambda$]')
    plt.colorbar(img)
    plt.show()
    #
    # w = 0.4
    # ymax = 0.15 + 0.20
    # ymin = 0.10
    # xmax = 0.10 + 0.20
    #
    # plt.axvline(x=-0.10, ymin=ymin/w, ymax=ymax/w, color='red', linestyle='--')
    # plt.axhline(y=-0.10, xmin=ymin/w, xmax=xmax/w, color='red', linestyle='--')
    # plt.scatter(-0.10, 0.15, color='red')
    # plt.scatter(-0.10, -0.10, color='red')
    # plt.scatter(0.10, -0.10, color='red')

    """ FPS algorithm """
    # Simulate what happens when we ran multiple trials
    # of the FPS algorithm from random wavefront maps

    triang = triangular_numbers(N_levels=15)
    level = 14
    # Calculate how many Zernikes we need to have up to a certain Radial level
    exp = level - 1                         # Exponent of the Zernike radial polynomial
    N_zern = triang[level]                  # How many Zernikes in total to ask zern to use
    _coef = np.zeros(N_zern)
    z = zern.ZernikeNaive(mask=pupil)
    _phase = z(coef=_coef, rho=rho/RHO_APER, theta=theta, normalize_noll=False, mode='Jacobi',
               print_option=None)
    model_matrix_flat = z.model_matrix
    model_matrix = zern.invert_model_matrix(z.model_matrix, pupil)

    model_matrix = model_matrix[:, :, 3:]   # remove piston and tilts
    model_matrix_flat = model_matrix_flat[:, 3:]

    N_zern = model_matrix.shape[-1]  # Update the N_zern after removing Piston and Tilts

    PSF_zern = PointSpreadFunction([model_matrix, pupil, model_matrix_flat])


    """ Approach A - Loop / Select / Correct All """
    # For each aberration we try multiple corrections.
    # We decide the best one
    # We apply all corrections simultaneously at the end of the FPS loop
    Z = 0.15
    DM_STEPS = 11
    dm = np.linspace(-Z, Z, DM_STEPS, endpoint=True)

    N_trials = 20
    N_runs = 2
    STREHLS = []

    for k in range(N_trials):

        # Initial State
        coef0 = np.random.uniform(low=-Z, high=Z, size=N_zern)
        im0, s0 = PSF_zern.compute_PSF(coef0)
        _strehl = [s0]
        print("\nInitial Strehl ratio: %.3f" % s0)

        for j in range(N_runs):
            total_correction = []
            for i_aberr in range(N_zern):
                print("Correcting Aberration %d / %d" % (i_aberr + 1, N_zern))
                strehls = []
                for correction in dm:
                    new_coef = coef0.copy()
                    new_coef[i_aberr] += correction
                    im, s = PSF_zern.compute_PSF(new_coef)
                    strehls.append(s)
                best_corr = dm[np.argmax(strehls)]
                # print(best_corr)
                total_correction.append(best_corr)
            total_correction = np.array(total_correction)
            # print("Correction: ", total_correction)
            residual = coef0 + total_correction
            im_final, s_final = PSF_zern.compute_PSF(residual)
            _strehl.append(s_final)
            print("Final Strehl ratio: %.3f" % s_final)
            coef0 = residual

        STREHLS.append(_strehl)

    ss = np.array(STREHLS)

    iters = list(range(N_runs + 1))
    # ss = np.array([[0.1, 0.2, 0.3],
    #                [0.15, 0.17, 0.5]])

    plt.figure()
    # for i in range(ss.shape[0]):
    #     plt.scatter(iters, ss[i])
    plt.plot(ss.T)
    plt.xlabel(r'FPS iteration')
    plt.ylabel(r'Strehl ratio [ ]')
    plt.ylim([0.0, 1.0])
    plt.xticks(iters)
    plt.show()

    """ Approach B - Loop / Correct / Continue """
    # For each aberration we try multiple corrections.
    # We decide the best correction and APPLY it directly
    # We move to the next aberration

    N_trials = 20
    N_runs = 2
    STREHLS_B = []

    for k in range(N_trials):

        # Initial State
        coef0 = np.random.uniform(low=-Z, high=Z, size=N_zern)
        im0, s0 = PSF_zern.compute_PSF(coef0)
        _strehl = [s0]
        print("\nInitial Strehl ratio: %.3f" % s0)

        for j in range(N_runs):
            for i_aberr in range(N_zern):
                print("Correcting Aberration %d / %d" % (i_aberr + 1, N_zern))
                strehls = []
                for correction in dm:
                    new_coef = coef0.copy()
                    new_coef[i_aberr] += correction
                    im, s = PSF_zern.compute_PSF(new_coef)
                    strehls.append(s)
                best_corr = dm[np.argmax(strehls)]
                coef0 += best_corr      # Apply the correction
            im_final, s_final = PSF_zern.compute_PSF(coef0)
            _strehl.append(s_final)
            print("Final Strehl ratio: %.3f" % s_final)
        STREHLS_B.append(_strehl)






    """ Hessian Calculation """

    def hessian_diagonal(coef):

        delta = 1e-3
        N_aberr = coef.shape[0]
        img_nom, strehl_nom = PSF_zern.compute_PSF(coef)
        f_nom = -strehl_nom

        hess_diag = np.zeros(N_aberr)
        for k in range(N_aberr):
            print(k)
            delta_vect = np.zeros(N_aberr)
            delta_vect[k] = delta

            img_plus, strehl_plus = PSF_zern.compute_PSF(coef + delta_vect)
            f_plus = -strehl_plus
            img_minus, strehl_minus = PSF_zern.compute_PSF(coef - delta_vect)
            f_minus = -strehl_minus

            hess_diag[k] = (f_plus + f_minus - 2*f_nom) / (delta)**2
        return hess_diag

    Z = 0.2
    coef = np.random.uniform(low=-Z, high=Z, size=N_zern)
    hd = hessian_diagonal(coef)


