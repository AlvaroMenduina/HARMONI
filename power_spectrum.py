"""
Analysis of the radial PSF for Zernike wavefronts
"""

import numpy as np
from numpy.random import RandomState
from numpy.fft import fft, fft2, fftshift
import matplotlib.pyplot as plt
import zern_core as zern

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


def solve_triangular(N_zern):
    """
    Solves the equation of Triangular Numbers T_n
    T_n = n (n + 1) / 2
    for T_n = N_zern

    It returns the Radial Order associated with that many Zernikes
    """

    n = int(0.5 * (-1 + np.sqrt(1 + 8 * N_zern)))

    return n

def decay_factor(n, ratio=100):
    """
    Computes how much you need to rescale the coefficients
    at each Zernike row such that after "n" levels your magnitude
    has been reduced by a "ratio"
    """

    log_10_alfa = -np.log10(ratio) / (n - 1)
    alfa = 10 ** (log_10_alfa)

    return alfa


def generate_decay_vector(n, decay_alfa):
    """
    Create a vector of length Z containing
    the required scaling
    """
    vec = [1]
    for i in range(1, n):
        new_vec = [decay_alfa ** i for _ in range(i + 1)]
        vec.extend(new_vec)
    return np.array(vec)

plt.rc('font', family='serif')
plt.rc('text', usetex=False)

# Parameters
N = 1024
rho_aper = 0.25

x = np.linspace(-1., 1., N, endpoint=True)
xx, yy = np.meshgrid(x, x)
r = np.sqrt(xx ** 2 + yy ** 2)
angle = np.arctan2(yy, xx)
aperture_mask = r <= rho_aper

rf, af = (aperture_mask*r).flatten(), (aperture_mask*angle).flatten()

def compute_peak():

    P = aperture_mask * np.exp(1j * np.zeros_like(aperture_mask))
    im = (np.abs(fftshift(fft2(P))))**2

    return np.max(im)

PEAK = compute_peak()

def compute_strehl(zern_model, coef):

    phase = zern_model(coef=coef, rho=rf/rho_aper, theta=af, normalize_noll=False, mode='Jacobi', print_option=None)
    phase_map = phase.reshape((N, N))

    P = aperture_mask * np.exp(1j * phase_map)
    im = (np.abs(fftshift(fft2(P)))) ** 2 / PEAK

    return im.max()


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    class ZernikeWavefront(object):

        def __init__(self, N_PIX, N_zern):
            self.N_PIX = N_PIX
            rho_aper = 0.95
            _rho = np.linspace(0.0, rho_aper, self.N_PIX, endpoint=True)
            _theta = np.linspace(0.0, 2 * np.pi, self.N_PIX)

            rr, tt = np.meshgrid(_rho, _theta)
            self.aperture_mask = rr <= rho_aper
            self.rho = (self.aperture_mask * rr).flatten()
            self.theta = (self.aperture_mask * tt).flatten()

            self.xx = rr * np.cos(tt)
            self.yy = rr * np.sin(tt)

            self.create_model_matrix(N_zern)

        def create_model_matrix(self, N_zern):
            """
            Watch out because the matrices are in polar coordinates
            :param N_zern:
            :return:
            """

            _coef = np.zeros(N_zern)
            z = zern.ZernikeNaive(mask=np.ones((N, N)))
            _phase = z(coef=_coef, rho=self.rho, theta=self.theta, normalize_noll=False, mode='Jacobi',
                      print_option=None)
            self.model_matrix_flat = z.model_matrix
            self.model_matrix = zern.invert_model_matrix(z.model_matrix, np.ones((N, N)))
            self.N_zern = self.model_matrix.shape[-1]

            # phase_map = phase.reshape((N, N))

        def compute_polar_phasemap(self, coef):
            """

            :param coef:
            :return:
            """

            N = coef.shape[0]
            if N < self.N_zern:    # Zeropad the coef
                new_coef = np.zeros(self.N_zern)
                new_coef[:N] = coef
            if N > self.N_zern:     # Update the model Matrix
                self.create_model_matrix(N_zern=N)
                new_N_zern = self.N_zern
                new_coef = np.zeros(new_N_zern)
                new_coef[:N] = coef
            elif N == self.N_zern:
                new_coef = coef

            phase_map = np.dot(self.model_matrix, new_coef)
            return phase_map

        def plot_polar_phasemap(self, phase_map):

            z_min = min(np.min(phase_map), -np.max(phase_map))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            im1 = ax.imshow(phase_map, origin='lower', cmap='RdBu', extent=[0, rho_aper, 0, 2 * np.pi])
            im1.set_clim(z_min, -z_min)
            plt.colorbar(im1, ax=ax)
            ax.set_aspect('auto')
            plt.xlabel(r'Radius $\rho$')
            plt.ylabel(r'Angle $\theta$')
            plt.title('Wavefront map in Polar Coordinates')
            return

        def plot_cartesian_phasemap(self, phase_map):

            z_min = min(np.min(phase_map), -np.max(phase_map))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            im1 = ax.contourf(self.xx, self.yy, phase_map, 100, cmap='RdBu')
            ax.set_aspect('equal')
            im1.set_clim(z_min, -z_min)
            plt.colorbar(im1, ax=ax)
            plt.title('Wavefront map in Cartesian Coordinates')
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            return

        def compute_PSD(self, phase_map, N_samples=50):
            """
            Compute the Power Spectral Density (PSD) for a given phase map
            The PSD is computed radially for N_samples and averaged
            """
            profile = []
            for i in np.arange(0, self.N_PIX, self.N_PIX//N_samples):
                fft_r = fft(phase_map[i], norm='ortho')
                PSD = fftshift((np.abs(fft_r)) ** 2)
                PSD_sym = PSD[self.N_PIX//2 + 1:]
                profile.append(PSD_sym)
            prof = np.array(profile)
            return np.mean(prof, axis=0)


    N_PIX = 1024
    N_zern = 20
    randgen = RandomState(1234)
    coef = randgen.normal(size=N_zern)/2
    print('First 10 Zernike coefficients')
    print(coef[:10])



    wavefronts = ZernikeWavefront(N_PIX, 20)
    phase_map = wavefronts.compute_polar_phasemap(coef)

    """ Plot the Wavefront in Polar coordinates """
    wavefronts.plot_polar_phasemap(phase_map)

    """ Plot the Wavefront in Cartesian coordinates """
    wavefronts.plot_cartesian_phasemap(phase_map)

    plt.show()

    # Impact of Zernike strength
    # Changing the intensity of the Zernike coefficients does NOT change the shape of the PSD
    # but it changes the area under the curve
    PSD = wavefronts.compute_PSD(phase_map)
    PSD_2 = wavefronts.compute_PSD(2*phase_map)

    plt.figure()
    plt.loglog(PSD, label=r'x')
    plt.loglog(PSD_2, label=r'2x')
    plt.legend()
    plt.title(r'Number of Zernikes: %d' %N_zern)
    plt.show()

    # --------------------------------------------------------------------

    plt.figure()
    for Z in [10, 20, 50]:
        coef = np.random.uniform(low=-1, high=1, size=Z)
        phase_map = wavefronts.compute_polar_phasemap(coef)
        PSD = wavefronts.compute_PSD(phase_map)

        plt.loglog(PSD)
    # plt.legend()
    # plt.title(r'Number of Zernikes: %d' % N_zern)
    plt.show()

    """ Coefficient decay with Zernike order """
    N_levels = 5
    triangular = triangular_numbers(N_levels=N_levels)
    alpha_decay = decay_factor(N_levels, ratio=10000)
    decay_scales = generate_decay_vector(N_levels, alpha_decay)

    ones = 0.1*np.ones(triangular[N_levels])
    decayed_ones = decay_scales * ones

    phase_ones = wavefronts.compute_polar_phasemap(ones)
    phase_decayed = wavefronts.compute_polar_phasemap(decayed_ones)

    PSD_ones = wavefronts.compute_PSD(phase_ones)
    PSD_decayed = wavefronts.compute_PSD(phase_decayed)

    wavefronts.plot_polar_phasemap(phase_ones)
    wavefronts.plot_polar_phasemap(phase_decayed)

    wavefronts.plot_cartesian_phasemap(phase_ones)
    wavefronts.plot_cartesian_phasemap(phase_decayed)

    plt.figure()
    plt.loglog(PSD_ones)
    plt.loglog(PSD_decayed)
    plt.show()



    nn = PSD_sym.shape[0]
    f = np.linspace(1, N // 2, N // 2)
    f2 = f ** (-2)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for i in range(len(profile)):
        ax.plot(profile[i], color='red')

    ax.plot(f, f2, color='black', linestyle='--', label=r'$f^{-2}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1, N // 2])
    # ax.set_ylim([1e-9, 1])
    ax.set_xlabel('Spatial Frequency')
    ax.legend()
    # plt.title('Radial PSD')
    plt.show()
