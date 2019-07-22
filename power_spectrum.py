import numpy as np
from numpy.random import RandomState
from numpy.fft import fft, fft2, fftshift
import matplotlib.pyplot as plt
import zern_core as zern



def solve_triangular(Z):
    """
    Solves the equation of Triangular Numbers T_n
    T_n = n (n + 1) / 2
    for T_n = N_zern
    """

    n = int(0.5 * (-1 + np.sqrt(1 + 8 * Z)))

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


    N_zern = 20
    randgen = RandomState(1234)

    N_crop = int(N*rho_aper)

    coef = randgen.normal(size=N_zern)/2
    print('First 10 Zernike coefficients')
    print(coef[:10])

    """ Plot the Wavefront in Polar coordinates """

    rho_1 = np.linspace(0.0, 1.0, N, endpoint=True)
    theta_1 = np.linspace(0.0, 2 * np.pi, N)

    rr, tt = np.meshgrid(rho_1, theta_1)
    aper = rr <= rho_aper
    rho, theta = (aper*rr).flatten(), (aper*tt).flatten()

    z = zern.ZernikeNaive(mask=np.ones((N, N)))
    phase = z(coef=coef, rho=rho/rho_aper, theta=theta, normalize_noll=False, mode='Jacobi', print_option=None)
    phase_map = phase.reshape((N, N))
    # phase_map = phase_map[aper].reshape((N_crop, N_crop))

    compute_strehl(z, coef)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(phase_map, origin='lower', cmap='viridis', extent=[0, 1, 0, 2 * np.pi])
    ax.set_aspect('auto')
    plt.xlabel(r'Radius $\rho$')
    plt.ylabel(r'Angle $\theta$')
    # plt.title('Wavefront map in Polar Coordinates')

    # --------------------------------------------------------------------
    """ Plot the Wavefront in Cartesian coordinates """

    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = ax.contourf(xx, yy, phase_map, 100, cmap='seismic')
    ax.set_aspect('equal')
    # plt.title('Wavefront map in Cartesian Coordinates')
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    # plt.colorbar(ax1)

    # --------------------------------------------------------------------
    """ Compute radial PSD for the WAVEFRONT """

    def compute_PSD(phase_map, N_samples=50):
        """
        Compute the Power Spectral Density (PSD) for a given phase map
        The PSD is computed radially for N_samples and averaged
        """
        profile = []
        for i in np.arange(0, N, N_samples):
            fft_r = 1. / N_crop * fft(phase_map[i, :N_crop])
            PSD = fftshift((np.abs(fft_r)) ** 2)
            PSD_sym = PSD[N_crop // 2 + 1:]
            profile.append(PSD_sym)
        prof = np.array(profile)
        return np.mean(prof, axis=0)

    # Impact of Zernike strength
    # Changing the intensity of the Zernike coefficients does NOT change the shape of the PSD
    # but it changes the area under the curve
    p = compute_PSD(phase_map)
    p2 = compute_PSD(2*phase_map)

    plt.figure()
    plt.loglog(p, label=r'x')
    # plt.loglog(p2, label=r'2x')
    plt.legend()
    plt.title(r'Number of Zernikes: %d' %N_zern)
    # plt.show()


    N_trials = 50
    coeffs = randgen.normal(size=(N_trials, N_zern))
    strehls = []
    for c in coeffs:
        strehls.append(compute_strehl(z, c))



    phase = z(coef=coef, rho=rho/rho_aper, theta=theta, normalize_noll=False, mode='Jacobi', print_option=None)
    phase_map = phase.reshape((N, N))
    # phase_map = phase_map[aper].reshape((N_crop, N_crop))








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
