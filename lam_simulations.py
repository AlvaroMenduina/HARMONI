"""
==========================================================
                  LAM Simulations
==========================================================

Some analysis for the people at LAM to try and use
Phase Diversity with our POP results

Influence of Zernike aberrations on the PSF features
to try to understanding the Image Slicer better
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyzdde.zdde import readBeamFile
from astropy.io import fits

# Slices
list_slices = [17, 19, 21, 53, 55, 57, 59]
# list_slices = [17, 19, 21, 55]
i_central = 19

# POP arrays - Nyquist sampled PSF
x_size = 2.08           # Physical size of array at Image Plane
N_pix = 128              # Number of pixels in the Zemax BFL
N_crop = 128             # Crop to a smaller region around the PSF
min_pix = N_pix//2 - N_crop//2
max_pix = N_pix//2 + N_crop//2
extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]

# Zernikes: Defocus, Astigmatism x2, Coma x2
zern_list_low = ['Defocus', 'Astigmatism X', 'Astigmatism Y', 'Coma X', 'Coma Y']
zern_list_high = ['Trefoil X', 'Trefoil Y', 'Quatrefoil X', 'Quatrefoil Y']


""" HELPER FUNCTIONS """


def reshape_flattened(array, pix=N_crop):
    """
    It takes a PSF composed of a flattened nominal
    and a flattened defocused PSF and rebuilds them
    and concatenates them into a 2D image
    """
    n_flat = array.shape[0]
    nominal = array[:n_flat//2].reshape((pix, pix))
    defocus = array[n_flat//2:].reshape((pix, pix))
    return np.concatenate((nominal, defocus), axis=1)

# ============================================================================== #
#                                ZEMAX INTERFACE                                 #
# ============================================================================== #

def read_beam_file(file_name):
    """
    Reads a Zemax Beam File and returns the Irradiance
    of the Magnetic field E
    """
    beamData = readBeamFile(file_name)
    (version, (nx, ny), ispol, units, (dx, dy), (zposition_x, zposition_y),
     (rayleigh_x, rayleigh_y), (waist_x, waist_y), lamda, index, re, se,
     (x_matrix, y_matrix), (Ex_real, Ex_imag, Ey_real, Ey_imag)) = beamData

    E_real = np.array([Ex_real, Ey_real])
    E_imag = np.array([Ex_imag, Ey_imag])

    re = np.linalg.norm(E_real, axis=0)
    im = np.linalg.norm(E_imag, axis=0)

    irradiance = (re ** 2 + im ** 2).T
    power = np.sum(irradiance)
    print('Total Power: ', power)
    return (nx, ny), (dx, dy), irradiance, power

def read_all_zemax_files(path_zemax, name_convention, file_list):
    """
    Goes through the ZBF Zemax Beam Files of all Slices and
    extracts the beam information (X_size, Y_size) etc
    as well as the Irradiance distribution
    """
    info, data, powers = [], [], []

    for k in file_list:
        print('\n======================================')

        if k < 10:
            file_id = name_convention + ' ' + str(k) + '_POP.ZBF'
        else:
            file_id = name_convention + str(k) + '_POP.ZBF'
        file_name = os.path.join(path_zemax, file_id)

        print('Reading Beam File: ', file_id)

        NM, deltas, beam_data, power = read_beam_file(file_name)
        Dx, Dy = NM[0] * deltas[0], NM[1] * deltas[1]
        info.append([k, Dx, Dy])
        data.append(beam_data)
        powers.append(power)

    beam_info = np.array(info)
    irradiance_values = np.array(data)
    powers = np.array(powers)

    return beam_info, irradiance_values, powers


def load_files(path, file_list, N, defocus=False):
    """
    Loads the Zemax beam files, constructs the PSFs
    and normalizes everything by the intensity of the PSF
    at i_norm (the Nominal PSF)
    """

    pop_slicer_nom = POP_Slicer()

    if defocus==False:
        flat_PSFs = np.empty((N, N_crop * N_crop))
        PSFs = np.empty((N, N_crop, N_crop))

        for k in range(N):
            if k < 10:
                # We have to adjust for the ZBF format. Before 10 it adds a space []3
                # name_nominal = 'IFU_TopAB_HARMONI_light' + '% d_' % k
                name_nominal = 'IFUAB_FDR_1_04clean_Tolerance' + '% d_' % k
            else:
                # name_nominal = 'IFU_TopAB_HARMONI_light' + '%d_' % k
                name_nominal = 'IFUAB_FDR_1_04clean_Tolerance' + '%d_' % k

            pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
            slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

            PSFs[k, :, :] = slicers_nom
            flat_PSFs[k, :] = slicers_nom.flatten()

    if defocus:

        pop_slicer_foc = POP_Slicer()
        flat_PSFs = np.empty((N, 2 * N_crop * N_crop))
        PSFs = np.empty((N, 2, N_crop, N_crop))

        for k in range(N):
            if k < 10:
                # We have to adjust for the ZBF format. Before 10 it adds a space []3
                # name_nominal = 'IFU_TopAB_HARMONI_light' + '% d_' % k
                name_nominal = 'IFUAB_FDR_1_04clean_Tolerance' + '% d_' % k
                name_defocus = 'IFUAB_FDR_1_04clean_Tolerance' + '% d_FOC_' % k
            else:
                # name_nominal = 'IFU_TopAB_HARMONI_light' + '%d_' % k
                name_nominal = 'IFUAB_FDR_1_04clean_Tolerance' + '%d_' % k
                name_defocus = 'IFUAB_FDR_1_04clean_Tolerance' + '%d_FOC_' % k

            pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
            slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

            pop_slicer_foc.get_zemax_files(path, name_defocus, file_list)
            slicers_foc = np.sum(pop_slicer_foc.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

            PSFs[k, 0, :, :], PSFs[k, 1, :, :] = slicers_nom, slicers_foc
            flat_PSFs[k, :] = np.concatenate((slicers_nom.flatten(), slicers_foc.flatten()))

    return [flat_PSFs, PSFs]

class POP_Slicer(object):
    """
    Physical Optics Propagation (POP) analysis of an Image Slicer
    """
    def __init__(self):
        pass

    def get_zemax_files(self, zemax_path, name_convention, file_list):
        _info, _data, _power = read_all_zemax_files(zemax_path, name_convention, file_list)
        self.beam_info = _info
        self.beam_data = _data
        self.powers = _power

def downsample_slicer_pixels(square_PSFs):
    """
    Raw PSF files sample the slice width with 2 pixels that can take different values
    This is not exactly true, as the detector pixels are elongated at the slicer,
    with only 1 true value covering 2 square pixels

    This functions fixes that issue by taking the average value pairwise
    :param array: PSF array
    :return:
    """

    n_psf, n_pix = square_PSFs.shape[0], square_PSFs.shape[-1]
    downsampled_PSFs = np.zeros_like(square_PSFs)
    flat_PSFs = np.empty((n_psf, 2 * n_pix * n_pix))
    for k in range(n_psf):
        for i in np.arange(1, n_pix-1, 2):
            # print(i)
            row_foc = square_PSFs[k, 0, i, :]
            next_row_foc = square_PSFs[k, 0, i+1, :]
            mean_row_foc = 0.5*(row_foc + next_row_foc)

            row_defoc = square_PSFs[k, 1, i, :]
            next_row_defoc = square_PSFs[k, 1, i+1, :]
            mean_row_defoc = 0.5*(row_defoc + next_row_defoc)

            downsampled_PSFs[k, 0, i, :] = mean_row_foc
            downsampled_PSFs[k, 0, i + 1, :] = mean_row_foc

            downsampled_PSFs[k, 1, i, :] = mean_row_defoc
            downsampled_PSFs[k, 1, i + 1, :] = mean_row_defoc

        flat_PSFs[k] = np.concatenate((downsampled_PSFs[k, 0].flatten(), downsampled_PSFs[k, 1].flatten()))

    return square_PSFs, downsampled_PSFs, flat_PSFs

# ============================================================================== #
#                              SAMPLING FUNCTIONS                                #
# ============================================================================== #

class Resampler_Zemax(object):
    """
    Created to fix the fact that Zemax POP always forces the arrays to have a CENTRAL pixel
    for the PSF peak, and does not like even numbers for Pixels across a Slice

    In order to properly model the 4 x 4 mas scale (with 2 pixels per slice) we need to
    resample the POP arrays and adjust the
    """
    def __init__(self):
        pass

    def resample_odd_to_2pix(self, array):
        """
        Mapping from an (N_odd, N) array to a 2 pix grid (2, N)
        It takes the N_odd array and divides it in half.
        It distributes one half of the lowest pixel and one half of the middle pixel
        and all pixels in between to the new lower pixel in the 2pix array.
        Equivalently, it takes half the upper pixel and half the middle pixel
        and all pixels in between and puts that sum in the upper pixel.
        """
        N_odd, N = array.shape
        N_middle = N_odd//2
        two_pix_array = np.zeros((2, N))
        lower_end = array[0, :]
        upper_end = array[-1, :]
        middle = array[N_odd//2, :]

        lower_pix = lower_end / 2 + np.sum(array[1:N_middle, :], axis=0) + middle / 2
        upper_pix = middle / 2 + np.sum(array[N_middle+1:-1, :], axis=0) + upper_end / 2

        two_pix_array[0, :] = lower_pix
        two_pix_array[1, :] = upper_pix

        return two_pix_array

    def resample_direction(self, PSF_array, N_slices):
        """
        Resamples the PSF along ONE direction, by calling resample_odd_to_2pix
        for each slice (N_slices considered)
        :param PSF_array: array to resample along the first dimension
        :param N_slices: Number of slices to consider in the PSF
        :return: an array of size (2*N_slices, N_pix)
        """
        physical_length = 2.08      # Length of the PSF arrays [mm]
        slicer_width = 0.130        # Size of a slice in the image plane [mm]
        N_pix = PSF_array.shape[0]
        pixels_per_slice = N_pix / (physical_length / slicer_width)
        p_s = pixels_per_slice

        N_odd = p_s + 1             # The number of pixels (odd) in a Zemax slice

        # Start with the Central Slice
        low_central = int(N_pix//2 - p_s//2)
        up_central = int(N_pix//2 + 1 + p_s//2)
        central_slice = PSF_array[low_central:up_central, :]

        new_central = self.resample_odd_to_2pix(central_slice)

        # Continue with Lower Slices
        low_slices = []
        for i in range((N_slices - 1)//2)[::-1]:
            low_lim = int(low_central - (i + 1) * N_odd)
            up_lim = int(up_central - (i + 1) * N_odd)
            # print(low_lim)
            # print(up_lim)
            current_slice = PSF_array[low_lim:up_lim, :]
            new_slice = self.resample_odd_to_2pix(current_slice)
            low_slices.append(new_slice)

        # Add the Central Slice to the list
        low_slices.append(new_central)

        # Upper slices
        for i in range((N_slices - 1) // 2):
            low_lim = int(low_central + (i + 1) * N_odd)
            up_lim = int(up_central + (i + 1) * N_odd)
            # print(low_lim)
            # print(up_lim)
            current_slice = PSF_array[low_lim:up_lim, :]
            new_slice = self.resample_odd_to_2pix(current_slice)
            low_slices.append(new_slice)

        # join everything together
        new_PSF = np.concatenate(low_slices, axis=0)
        # print(new_PSF.shape)
        return new_PSF

    def resample_PSF(self, PSF, N_slices):
        """
        The method self.resample_direction only fixes 1 direction.
        If we want to obtain a square PSF we have to resample along the slice.

        We can achieve this by doing self.resample_direction accounting for an axis flip
        :param PSF: original PSF to resample
        :param N_slices: number of slices to consider
        :return: a resamples PSF of size (2*N_slices, 2*N_slices)
        """

        PSF_across = self.resample_direction(PSF, N_slices)
        # Flip so that we can now resample along the slice length
        PSF_flipped = PSF_across.T
        PSF_resample = self.resample_direction(PSF_flipped, N_slices)
        PSF_unflipped = PSF_resample.T
        return PSF_unflipped

    def resample_all_PSFs(self, PSFs, N_slices):
        """
        Repeats self.resample_PSF across a whole array of PSFs
        :param PSFs: an (N_PSF, N_pix, N_pix) to resample
        :param N_slices:
        :return:
        """
        N_PSFs = PSFs.shape[0]
        new_PSFs = np.empty((N_PSFs, 2*N_slices, 2*N_slices))
        for i in range(N_PSFs):
            new_PSFs[i] = self.resample_PSF(PSFs[i], N_slices=N_slices)
        return new_PSFs

def plot_slices(ls, color='white'):
    plt.axhline(y=width / 2, color=color, linestyle=ls, alpha=0.7)
    plt.axhline(y=-width / 2, color=color, linestyle=ls, alpha=0.7)
    plt.axhline(y=3 * width / 2, color=color, linestyle=ls, alpha=0.5)
    plt.axhline(y=-3 * width / 2, color=color, linestyle=ls, alpha=0.5)
    plt.axhline(y=5 * width / 2, color=color, linestyle=ls, alpha=0.2)
    plt.axhline(y=-5 * width / 2, color=color, linestyle=ls, alpha=0.2)


if __name__ == "__main__":

    path_files_slicer = os.path.abspath('H:\POP\IFU CLEAN\_NO SLICER APERTURE')

    N_points_slicer = 21
    max_wave = 0.25
    coef_slicer = np.linspace(-max_wave, max_wave, N_points_slicer, endpoint=True)
    # np.save(os.path.join(path_files, 'coef_slicer'), coef)
    # np.savetxt(os.path.join(path_files, 'coef_slicer.txt'), coef, fmt='%.5f')

    # Perfect PSF as a reference
    path_perfect_slicer = os.path.join(path_files_slicer, 'PERFECT')
    PSFs_perfect_slicer = load_files(path_perfect_slicer, N=1, file_list=list_slices)

    # ============================================================================== #
    #                  RESAMPLE THE PSF TO MATCH THE 4 X 4 MAS SCALE                 #
    # ============================================================================== #

    N_slices = 7
    width = 0.130
    length = width * N_slices
    resampled_extend = (-length/2, length/2, -length/2, length/2)
    ls = '-.'

    resampler = Resampler_Zemax()
    P = PSFs_perfect_slicer[1][0]
    PSF_4x4_slicer = resampler.resample_PSF(P, N_slices=N_slices)
    PEAK_slicer = np.max(PSF_4x4_slicer)

    # Rescale so that the peak of the PSF is 1.0
    PSF_4x4_slicer /= PEAK_slicer

    plt.figure()
    plt.imshow(PSF_4x4_slicer, extent=resampled_extend, origin='lower')
    plt.colorbar()
    plot_slices(ls)
    plt.show()

    # ============================================================================== #
    path_files = os.path.abspath('H:\POP\IFU CLEAN\B WIDE')
    N_points = 41
    coef = np.linspace(-max_wave, max_wave, N_points, endpoint=True)
    path_perfect = os.path.join(path_files, 'PERFECT')
    PSFs_perfect = load_files(path_perfect, N=1, file_list=list_slices)

    P = PSFs_perfect[1][0]
    PSF_4x4 = resampler.resample_PSF(P, N_slices=N_slices)
    PEAK = np.max(PSF_4x4)

    # Rescale so that the peak of the PSF is 1.0
    PSF_4x4 /= PEAK

    # ============================================================================== #
    """ Compute the evolution of the PEAK vs aberration intensity """

    # folders = ['DEFOCUS', 'ASTIG X', 'ASTIG Y', 'COMA X', 'SPHERICAL', 'TREFOIL X']
    folders = ['COMA Y']
    path_results = os.path.join('Results', 'LAM', 'B WIDE')

    PSF_list = []
    PSF_list_slicer = []
    list_of_peaks = []
    p_max = 1
    for i, fold in enumerate(folders):
        print('\nAberration: ' + fold)

        # Load all the PSFs for each aberration
        path_aberr = os.path.join(path_files, fold)
        PSFs_aberr = load_files(path_aberr, N=N_points, file_list=list_slices)

        PSFs_resampled = resampler.resample_all_PSFs(PSFs_aberr[1], N_slices=N_slices)
        PSFs_resampled /= PEAK

        PSF_list.append(PSFs_resampled)

        # No Slicer
        path_aberr_slicer = os.path.join(path_files_slicer, fold)
        PSFs_aberr_slicer = load_files(path_aberr_slicer, N=N_points_slicer, file_list=list_slices)

        PSFs_resampled_slicer = resampler.resample_all_PSFs(PSFs_aberr_slicer[1], N_slices=N_slices)
        PSFs_resampled_slicer /= PEAK_slicer

        PSF_list_slicer.append(PSFs_resampled_slicer)

        # Compute the PEAK ratio
        peak_aberr = np.max(PSFs_resampled, axis=(1, 2))
        p_copy = PSFs_resampled.copy()
        mm = np.sort(p_copy.reshape((PSFs_resampled.shape[0], -1)), axis=-1)
        pp = np.mean(mm[:, -p_max:], axis=-1)
        list_of_peaks.append(peak_aberr)

    peaks = np.array(list_of_peaks)

    # Plot the PEAK vs ABERRATION
    colors = cm.rainbow(np.linspace(0, 1, len(folders)))
    plt.figure()
    for i in range(len(folders)):
        plt.plot(coef, peaks[i], label=folders[i], color=colors[i])
        plt.scatter(coef, peaks[i], s=10, color=colors[i])
    plt.legend()
    plt.xlim([-max_wave, max_wave])
    # plt.ylim([0.75, 1])
    plt.xlabel('Aberration [waves]')
    plt.ylabel('Peak ratio')
    plt.show()

    # Show how the PSFs look like for each ABERRATION type
    map2 = 'bwr'
    for k in range(len(folders)):
        for j in np.arange(N_points//2, N_points, 10):
            im = PSF_list[k][j]
            # plt.figure()
            # plt.imshow(im, extent=resampled_extend, origin='lower')
            # plt.colorbar()
            # plot_slices(ls)
            # plt.title(folders[k] + r' ($a =$%.3f $\lambda$)' %coef[j])
            # plt.savefig(os.path.join(path_results, folders[k], '%d' %j))

            residual = im - PSF_list[k][N_points//2]
            vmax = max(-residual.min(), residual.max())

            plt.figure()
            plt.imshow(residual, cmap=map2, extent=resampled_extend, origin='lower')
            plt.colorbar()
            plt.clim(-vmax, vmax)
            plot_slices(ls, color='black')
            plt.title('Residual ' + folders[k] + r' ($a =$%.3f $\lambda$)' %coef[j])
            plt.savefig(os.path.join(path_results, folders[k], 'res_%d' % j))

            # NO Slicer
            c = coef[j]
            j_slicer = np.argwhere(coef_slicer == c)[0][0]
            im_slicer = PSF_list_slicer[k][j_slicer]
            residual_slicer = im_slicer - PSF_list[k][N_points_slicer//2]

            # vmax = max(-residual_slicer.min(), residual_slicer.max())
            plt.figure()
            plt.imshow(im_slicer - im, cmap=map2, extent=resampled_extend, origin='lower')
            plt.colorbar()
            # plt.clim(-vmax, vmax)
            plot_slices(ls, color='black')
            plt.title('No Slicer Diff: ' + folders[k] + r' ($a =$%.3f $\lambda$)' %coef[j])
            plt.savefig(os.path.join(path_results, folders[k], 'res_diff_%d' % j))

    plt.show()


    # ============================================================================== #
    # Quadratic fit
    from scipy.optimize import least_squares as lsq

    def quadratic_residual(x, y_data, aberr_coef):
        model = quadratic(x, aberr_coef)
        residual = y_data - model
        return residual

    def quadratic(x, aberr_coef):
        a, x_shift = x[0], x[1]
        f = 1 + a * (aberr_coef + x_shift * np.ones_like(aberr_coef))**2
        return f

    def r2(x_fit, y_data, aberr_coef):
        mean_data = np.mean(y_data)
        ss_total = np.sum((y_data - mean_data)**2)
        ss_res = np.sum((quadratic_residual(x_fit, y_data, aberr_coef))**2)
        R2 = 1 - ss_res / ss_total
        return R2

    # Aberrations that behave like a parabola
    parab_indices = [0, 1, 2, 4]

    parab_fit = []
    parab_shift = []
    for k in parab_indices:
        y_data = peaks[k]
        x_solve = lsq(quadratic_residual, x0=[-0.1 , 0.0], args=(y_data, coef,))
        a_fit, x_shift = x_solve['x']

        print(a_fit, x_shift)
        parab_fit.append(a_fit)
        parab_shift.append(x_shift)

        R2 = r2([a_fit, x_shift], y_data, coef)

        plt.figure()
        lab = label=r'Fit: $f(x)=1 - %.3f (x + %.3f)^2$' % (-a_fit, x_shift)
        plt.scatter(coef, y_data, label='Data', s=10)
        plt.plot(coef, quadratic([a_fit, x_shift], coef), label=lab,
                 color='black', linestyle='--')
        plt.title(folders[k])
        plt.legend()
        plt.savefig(os.path.join(path_results, folders[k], 'quad_fit'))
    plt.show()

    # ============================================================================== #
    #                        SAVE DATA FOR PHASE DIVERSITY                           #
    # ============================================================================== #

    path_data = os.path.abspath('H:\POP\IFU CLEAN\B WIDE\DATA LAM')

    # PERFECT PSF
    plt.figure()
    plt.imshow(PSF_4x4, extent=resampled_extend, origin='lower')
    plt.colorbar()
    plot_slices(ls)
    plt.savefig(os.path.join(path_data, 'perfect_psf'))
    plt.show()

    hdu_perfect = fits.PrimaryHDU(PSF_4x4)
    hdu_perfect.writeto(os.path.join(path_data, 'perfect_psf.fits'))

    np.save(os.path.join(path_data, 'perfect_psf'), PSF_4x4)
    np.savetxt(os.path.join(path_data, 'perfect_psf.txt'), PSF_4x4)

    folders = ['DEFOCUS', 'ASTIG X', 'ASTIG Y', 'COMA X', 'SPHERICAL', 'TREFOIL X']

    PSF_list = []
    for i, fold in enumerate(folders):
        print('\nAberration: ' + fold)

        # Load all the PSFs for each aberration
        path_aberr = os.path.join(path_files, fold)
        PSFs_aberr = load_files(path_aberr, N=N_points, file_list=list_slices)

        PSFs_resampled = resampler.resample_all_PSFs(PSFs_aberr[1], N_slices=N_slices)
        PSFs_resampled /= PEAK

        print(PSFs_resampled.shape)

        PSF_list.append(PSFs_resampled)

    PSFs_array = np.array(PSF_list)

    # Check a random PSF
    plt.figure()
    plt.imshow(PSFs_array[0, 0], extent=resampled_extend, origin='lower')
    plt.colorbar()
    plot_slices(ls)
    plt.show()

    # Save the PSFs in a (N_aberr, N_points, 2*N_slices, 2*N_slices) array
    file_name = os.path.join(path_data, 'aberrated_psfs')
    np.save(file_name, PSFs_array)
    hdu_aberr = fits.PrimaryHDU(PSFs_array)
    hdu_aberr.writeto(os.path.join(path_data, 'aberrated_psfs.fits'))

    np.save(os.path.join(path_data, 'aberration_coef'), coef)
    np.savetxt(os.path.join(path_data, 'aberration_coef.txt'), coef, fmt='%.5f')

    # ============================================================================== #
    #                        LOAD DATA FOR PHASE DIVERSITY                           #
    # ============================================================================== #

    N_examples = 5
    factor = max_wave
    coef_random = np.random.uniform(low=-factor, high=factor, size=(N_examples, len(folders)))
    np.savetxt(os.path.join(path_data, 'coef_random.txt'), coef_random, fmt='%.5f')
    np.save(os.path.join(path_data, 'coef_random'), coef_random)

    PSFs_PD = load_files(os.path.join(path_files, 'DATA LAM'), N=N_examples,
                         file_list=list_slices, defocus=True)
    # PSFs_PD[1] is the Square N_pix, N_pix set, and it contains both NOM and FOC
    PSFs_PD_nom = resampler.resample_all_PSFs(PSFs_PD[1][:,0,:,:], N_slices=N_slices)
    PSFs_PD_nom /= PEAK

    # DEFOCUSED version
    PSFs_PD_foc = resampler.resample_all_PSFs(PSFs_PD[1][:,1,:,:], N_slices=N_slices)
    PSFs_PD_foc /= PEAK

    for i in range(N_examples):
        im = np.concatenate((PSFs_PD_nom[i], PSFs_PD_foc[i]), axis=1)
        plt.figure()
        plt.imshow(im, extent=(-length, length, -length/2, length/2), origin='lower')
        plt.colorbar()
        plot_slices(ls)
    plt.show()

    np.save(os.path.join(path_data, 'random_psfs_nominal'), PSFs_PD_nom)
    np.save(os.path.join(path_data, 'random_psfs_defocused'), PSFs_PD_foc)

    def save_fits(nominal, defocused, zern_coef):
        # Saves a FITS file for each random PSF
        # The primary HDU contains the nominal PSF, the 1st extension
        # contains the defocused version
        # Zernike coefficients are saved in the HEADER
        n_files = nominal.shape[0]
        slicer_width = 0.130     #[mm]
        for i in range(n_files):
            hdu = fits.PrimaryHDU([nominal[i], defocused[i]])
            hdr = hdu.header
            hdr['COMMENT'] = 'All Zernike intensities defined in [waves] at 1.5 um, following the Zernike Fringe definition from Zemax. Pixel scale [mm]. Auxiliary defocus [waves]'
            hdr['PXLSCALE'] = slicer_width/2   # Pixels per slice
            hdr['AUXFOC'] = 0.15
            for j, aberr in enumerate(folders):
                hdr[aberr] = zern_coef[i, j]

            hdu.writeto(os.path.join(path_data, 'random_psf_%d.fits') %i, overwrite=True)
            print('Saving file %d' %i)

    save_fits(PSFs_PD_nom, PSFs_PD_foc, coef_random)

    hdu_PD_nom = fits.PrimaryHDU(PSFs_PD_nom)
    hdu_PD_nom.writeto(os.path.join(path_data, 'random_psfs_nominal.fits'))

    hdu_PD_foc = fits.PrimaryHDU(PSFs_PD_foc)
    hdu_PD_foc.writeto(os.path.join(path_data, 'random_psfs_defocused.fits'))

















