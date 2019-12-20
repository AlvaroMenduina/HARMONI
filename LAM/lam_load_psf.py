"""
Author: Alvaro Menduina
Date: April 2019

PSF data for different Zernike aberrations

Details:
    - PSF arrays are 14 x 14 pixels wide, considering 7 slices (2 pixels per slice) in the 4x4 mas scale
    - All arrays are normalized by the peak intensity of the unaberrated PSF, such that the nominal PSF
    has a peak of 1.0

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

""" PARAMETERS """
N_pix = 14
N_slices = 7
width = 0.130                   # Physical length of a slice
length = width * N_slices
resampled_extend = (-length / 2, length / 2, -length / 2, length / 2)
ls = '-.'

def plot_slices(ls, color='white'):
    # Overlays the slicer footprints on the PSF plots
    plt.axhline(y=width / 2, color=color, linestyle=ls, alpha=0.7)
    plt.axhline(y=-width / 2, color=color, linestyle=ls, alpha=0.7)
    plt.axhline(y=3 * width / 2, color=color, linestyle=ls, alpha=0.5)
    plt.axhline(y=-3 * width / 2, color=color, linestyle=ls, alpha=0.5)
    plt.axhline(y=5 * width / 2, color=color, linestyle=ls, alpha=0.2)
    plt.axhline(y=-5 * width / 2, color=color, linestyle=ls, alpha=0.2)

if __name__ == "__main__":

    #FIXME: adapt the path to the files location
    path_data = os.path.abspath('H:\POP\IFU CLEAN\B WIDE\DATA LAM')

    # Load and show the nominal (unaberrated) PSF
    perfect_psf = np.load(os.path.join(path_data, 'perfect_psf.npy'))

    plt.figure()
    plt.imshow(perfect_psf, extent=resampled_extend, origin='lower')
    plt.colorbar()
    plot_slices(ls)
    plt.title('Perfect PSF')

    # Aberration coefficients: ranging from -0.25 to 0.25 waves
    coef = np.load(os.path.join(path_data, 'aberration_coef.npy'))
    N_points = coef.shape[0]

    # List of the aberrations in the PSF array
    aberrations = ['DEFOCUS', 'ASTIG X', 'ASTIG Y', 'COMA X', 'SPHERICAL', 'TREFOIL X']
    N_aberr = len(aberrations)

    # The PSFs are saved into an array of shape (N_aberr, N_points, N_pix, N_pix)
    # where N_aberr is the Zernike aberrations (defocus, astigmatism, ...)
    # and N_points is the number of aberration intensity values in [coef]
    aberrated_psf = np.load(os.path.join(path_data, 'aberrated_psfs.npy'))

    print('\nLoading aberrated PSFs')
    print(aberrated_psf.shape)

    # Plot some examples for each aberration
    for i, fold in enumerate(aberrations):
        print('\nAberration: ' + fold)

        for j in [0, N_points//2, -1]:
            print(coef[j])
            image = aberrated_psf[i][j]

            plt.figure()
            plt.imshow(image, extent=resampled_extend, origin='lower')
            plt.colorbar()
            plot_slices(ls)
            plt.title(fold + r' ($a =$%.3f $\lambda$)' %coef[j])

    # Plot residuals
    for i, fold in enumerate(aberrations):
        print('\nAberration: ' + fold)

        for j in [-1]:
            print(coef[j])
            image = aberrated_psf[i][j] - aberrated_psf[i][N_points//2]
            cmin = min(image.min(), -image.max())

            plt.figure()
            plt.imshow(image, extent=resampled_extend, origin='lower', cmap='seismic')
            plt.colorbar()
            plt.clim(cmin, -cmin)
            plt.title(fold + r' ($a =$%.3f $\lambda$)' %coef[j])

    # ============================================================================== #
    # Plot the Peak Intensity vs Aberration Coefficient
    peaks = np.max(aberrated_psf, axis=(2, 3))

    colors = cm.rainbow(np.linspace(0, 1, len(aberrations)))
    plt.figure()
    for i in range(len(aberrations)):
        plt.plot(coef, peaks[i], label=aberrations[i], color=colors[i])
        plt.scatter(coef, peaks[i], s=10, color=colors[i])
    plt.legend()
    plt.xlim([coef[0], coef[-1]])
    plt.xlabel('Aberration [waves]')
    plt.ylabel('Peak ratio')
    plt.grid(True)
    plt.title('Evolution of Peak Intensity with aberrations')
    plt.show()

    # ============================================================================== #
    # Phase Diversity PSFs

    # 5 PSFs containing random aberrations of ['DEFOCUS', 'ASTIG X', 'ASTIG Y', 'COMA X', 'SPHERICAL', 'TREFOIL X']
    defocus = 0.15          # Intensity of the extra defocus [waves]

    random_coef = np.load(os.path.join(path_data, 'coef_random.npy'))
    random_psfs_nom = np.load(os.path.join(path_data, 'random_psfs_nominal.npy'))
    random_psfs_foc = np.load(os.path.join(path_data, 'random_psfs_defocused.npy'))

    for i in range(random_psfs_nom.shape[0]):
        im = np.concatenate((random_psfs_nom[i], random_psfs_foc[i]), axis=1)
        plt.figure()
        plt.imshow(im, extent=(-length, length, -length/2, length/2), origin='lower')
        plt.colorbar()
        plot_slices(ls)
    plt.show()
