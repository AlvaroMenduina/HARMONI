"""
==========================================================
                  LAM Simulations
==========================================================

 --- Preliminary Tasks ---

 (1) Find the Zemax Setting that removes the aliasing effect
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyzdde.zdde import readBeamFile
import zern_core as zern
from astropy.io import fits

# Slices
list_slices = list(np.arange(1, 77, 2))
i_central = 19

# POP arrays - Nyquist sampled PSF
x_size = 4*2.08           # Physical size of array at Image Plane
N_pix = 128              # Number of pixels in the Zemax BFL
N_crop = 128             # Crop to a smaller region around the PSF
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


def load_files(path, file_list, N, N_pix=N_pix, N_crop=N_crop, defocus=False):
    """
    Loads the Zemax beam files, constructs the PSFs
    and normalizes everything by the intensity of the PSF
    at i_norm (the Nominal PSF)
    """
    min_pix = N_pix // 2 - N_crop // 2
    max_pix = N_pix // 2 + N_crop // 2

    pop_slicer_nom = POP_Slicer()

    if defocus==False:
        flat_PSFs = np.empty((N, N_crop * N_crop))
        PSFs = np.empty((N, N_crop, N_crop))

        for k in range(N):
            if k < 10:
                # We have to adjust for the ZBF format. Before 10 it adds a space []3
                name_nominal = 'IFU_TopAB_HARMONI_LAM' + '% d_' % k
                # name_nominal = 'IFUAB_FDR_1_04clean_Tolerance' + '% d_' % k
            else:
                name_nominal = 'IFU_TopAB_HARMONI_LAM' + '%d_' % k
                # name_nominal = 'IFUAB_FDR_1_04clean_Tolerance' + '%d_' % k

            pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
            slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

            PSFs[k, :, :] = slicers_nom
            flat_PSFs[k, :] = slicers_nom.flatten()
            info_nom = pop_slicer_nom.beam_info
            info = [info_nom]

    if defocus:

        pop_slicer_foc = POP_Slicer()
        flat_PSFs = np.empty((N, 2 * N_crop * N_crop))
        PSFs = np.empty((N, 2, N_crop, N_crop))

        for k in range(N):
            if k < 10:
                # We have to adjust for the ZBF format. Before 10 it adds a space []3
                # name_nominal = 'IFU_TopAB_HARMONI_light' + '% d_' % k
                name_nominal = 'IFU_TopAB_HARMONI_LAM' + '% d_' % k
                name_defocus = 'IFU_TopAB_HARMONI_LAM' + '% d_FOC_' % k
            else:
                # name_nominal = 'IFU_TopAB_HARMONI_light' + '%d_' % k
                name_nominal = 'IFU_TopAB_HARMONI_LAM' + '%d_' % k
                name_defocus = 'IFU_TopAB_HARMONI_LAM' + '%d_FOC_' % k

            pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
            slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

            pop_slicer_foc.get_zemax_files(path, name_defocus, file_list)
            slicers_foc = np.sum(pop_slicer_foc.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

            PSFs[k, 0, :, :], PSFs[k, 1, :, :] = slicers_nom, slicers_foc
            flat_PSFs[k, :] = np.concatenate((slicers_nom.flatten(), slicers_foc.flatten()))
            info_foc = pop_slicer_foc.beam_info

            info = [info_foc, info_foc]

    return [flat_PSFs, PSFs, info]

def load_slices(path, name_zemax='IFU_TopAB_HARMONI_LAM', file_list=[19], N_pix=N_pix, N_crop=N_crop, defocus=False):
    """
    Loads the Zemax beam files, constructs the PSFs
    and normalizes everything by the intensity of the PSF
    at i_norm (the Nominal PSF)
    """
    min_pix = N_pix // 2 - N_crop // 2
    max_pix = N_pix // 2 + N_crop // 2

    pop_slicer_nom = POP_Slicer()
    N = len(file_list)

    if defocus==False:
        # flat_PSFs = np.empty((N, N_crop * N_crop))
        PSFs = np.empty((N, N_crop, N_crop))

        # name_nominal = 'IFU_TopAB_HARMONI_LAM' + ' 0_'
        name_nominal = name_zemax + ' 0_'

        pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
        for k in range(N):
            PSFs[k, :, :] = pop_slicer_nom.beam_data[k][min_pix:max_pix, min_pix:max_pix]

    if defocus:

        pop_slicer_foc = POP_Slicer()
        # PSFs = np.empty((N, 2, N_crop, N_crop))
        PSFs = np.empty((N, N_crop, N_crop))

        # name_nominal = 'IFU_TopAB_HARMONI_LAM' + ' 0_'
        # name_defocus = 'IFU_TopAB_HARMONI_LAM' + ' 0_FOC_'
        name_defocus = name_zemax + ' 0_FOC_'

        # pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
        pop_slicer_foc.get_zemax_files(path, name_defocus, file_list)
        for k in range(N):
            # PSFs[k, :, :] = pop_slicer_nom.beam_data[k][min_pix:max_pix, min_pix:max_pix]
            PSFs[k, :, :] = pop_slicer_foc.beam_data[k][min_pix:max_pix, min_pix:max_pix]

    return PSFs

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

def plot_slices(ls, width=0.130, color='white'):
    plt.axhline(y=width / 2, color=color, linestyle=ls, alpha=0.7)
    plt.axhline(y=-width / 2, color=color, linestyle=ls, alpha=0.7)
    plt.axhline(y=3 * width / 2, color=color, linestyle=ls, alpha=0.5)
    plt.axhline(y=-3 * width / 2, color=color, linestyle=ls, alpha=0.5)
    plt.axhline(y=5 * width / 2, color=color, linestyle=ls, alpha=0.2)
    plt.axhline(y=-5 * width / 2, color=color, linestyle=ls, alpha=0.2)
    plt.axhline(y=7 * width / 2, color=color, linestyle=ls, alpha=0.15)
    plt.axhline(y=-7 * width / 2, color=color, linestyle=ls, alpha=0.15)
    plt.axhline(y=9 * width / 2, color=color, linestyle=ls, alpha=0.1)
    plt.axhline(y=-9 * width / 2, color=color, linestyle=ls, alpha=0.1)

def crop_array(array, crop=25):
    PIX = array.shape[0]
    min_crop = PIX // 2 - crop // 2
    max_crop = PIX // 2 + crop // 2
    array_crop = array[min_crop:max_crop, min_crop:max_crop]
    return array_crop


if __name__ == "__main__":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    path_prelim = os.path.abspath('D:/Thesis/LAM/POP/Slicer/0 Preliminary Tasks')
    N_slices = len(list_slices)

    # ===================

    x = np.linspace(-1, 1, 1024, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho_circ, theta_circ = np.sqrt((xx)**2 + yy**2), np.arctan2(xx, yy)
    rho_elli, theta_elli = np.sqrt((xx)**2 + (2*yy)**2), np.arctan2(xx, yy)
    pupil_circ = rho_circ <= 1.0
    pupil_elli = rho_elli <= 1.0

    ### Clipped Defocus
    rho_circ, theta_circ = rho_circ[pupil_circ], theta_circ[pupil_circ]
    zernike = zern.ZernikeNaive(mask=pupil_circ)
    _phase = zernike(coef=np.zeros(50), rho=rho_circ, theta=theta_circ, normalize_noll=False, mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix   # remove the piston and tilts
    H_matrix = zern.invert_model_matrix(H_flat, pupil_circ)
    defocus_circ = H_matrix[:,:,4].copy()

    ### Elliptic
    rho_elli, theta_elli = rho_elli[pupil_elli], theta_elli[pupil_elli]
    zernike = zern.ZernikeNaive(mask=pupil_elli)
    _phase = zernike(coef=np.zeros(25), rho=rho_elli, theta=theta_elli, normalize_noll=False, mode='Jacobi', print_option='Silent')
    H_flat_elli = zernike.model_matrix   # remove the piston and tilts
    H_matrix_elli = zern.invert_model_matrix(H_flat_elli, pupil_elli)
    piston_elli = H_matrix_elli[:,:,0].copy()

    defocus_circ = piston_elli * defocus_circ
    defocus_elli = H_matrix_elli[:,:,4].copy()

    plt.figure()
    plt.imshow(defocus_circ)
    plt.colorbar()

    plt.figure()
    plt.imshow(defocus_elli)
    plt.colorbar()
    plt.show()

    # Least Squares fit
    y_obs = defocus_circ[pupil_elli]
    H = H_flat_elli
    Ht = H.T
    Hy = np.dot(Ht, y_obs)
    N = np.dot(Ht, H)
    invN = np.linalg.inv(N)
    x_LS = np.dot(invN, Hy)

    i_LS = list(np.argwhere(np.abs(x_LS) > 0.1)[:, 0])

    for i in i_LS:
        plt.figure()
        plt.imshow(H_matrix_elli[:,:, i])
        plt.title(x_LS[i])
    plt.show()

    cm = 'jet'
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    ax1 = plt.subplot(3, 3, 1)
    img1 = ax1.imshow(defocus_circ, cm)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Clipped Circular Defocus')

    ax2 = plt.subplot(3, 3, 2)
    img2 = ax2.imshow(defocus_elli, cm)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Anamorphic Defocus')

    res1 = defocus_circ - defocus_elli
    ax3 = plt.subplot(3, 3, 3)
    img3 = ax3.imshow(res1, cm, vmin=-1, vmax=1)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title('Residual')
    # ----------------------------------- #

    astig = H_matrix[:,:, i_LS[2]]
    piston = H_matrix[:,:, i_LS[0]]
    new = x_LS[i_LS[0]]*piston + x_LS[i_LS[2]]*astig + x_LS[i_LS[1]] * defocus_elli
    res2 = defocus_circ - new
    ax4 = plt.subplot(3, 3, 4)
    img4 = ax4.imshow(astig, cm)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title('Anamorphic Astigmatism')

    ax5 = plt.subplot(3, 3, 5)
    img5 = ax5.imshow(new, cm, vmin=-1, vmax=1)
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax5.set_title('Astigmatism + Defocus')

    ax6 = plt.subplot(3, 3, 6)
    img6 = ax6.imshow(res2, cm, vmin=-1, vmax=1)
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax6.set_title('Residual')

    # ------------------------------------- #

    tref = H_matrix[:, :, i_LS[3]]
    new2 = new + x_LS[i_LS[3]]*tref
    res3 = defocus_circ - new2
    ax7 = plt.subplot(3, 3, 7)
    img7 = ax7.imshow(tref, cm)
    ax7.get_xaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)
    ax7.set_title('Anamorphic Quatrefoil')

    ax8 = plt.subplot(3, 3, 8)
    img8 = ax8.imshow(new, cm, vmin=-1, vmax=1)
    ax8.get_xaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)
    ax8.set_title('Astig + Defocus + Quatrefoil')

    ax9 = plt.subplot(3, 3, 9)
    img9 = ax9.imshow(res3, cm, vmin=-1, vmax=1)
    ax9.get_xaxis().set_visible(False)
    ax9.get_yaxis().set_visible(False)
    ax9.set_title('Residual')


    plt.show()

    """ The other way around! """
    ### Could we use clipped aberrations to mimic the anamorphic defocus?

    # Least Squares fit
    aberration = H_matrix_elli[:,:,5].copy()
    y_obs = aberration[pupil_elli]
    H_matrix_masked = H_matrix * pupil_elli[:,:,np.newaxis]     # Clip the circular matrix
    H = H_matrix_masked[pupil_elli]     # Flatten it with the elliptical pupil
    Ht = H.T
    Hy = np.dot(Ht, y_obs)
    N = np.dot(Ht, H)
    invN = np.linalg.inv(N)
    x_LS = np.dot(invN, Hy)

    i_LS = list(np.argwhere(np.abs(x_LS) > 0.1)[:, 0])
    s = np.zeros_like(aberration)
    for i in i_LS:
        plt.figure()
        plt.imshow(H_matrix_masked[:,:, i])

        s += x_LS[i] * H_matrix_masked[:,:, i]
        rms = np.std(H_matrix[:,:, i][pupil_circ])
        plt.title(x_LS[i])
        print(rms)
    plt.show()


    # ================================================================================================================ #

    # Autoresample after the Field Splitter blurs away the rings

    """ Resample after the Field Splitter onwards """
    PIX_POP = 1024
    path_1k = os.path.join(path_prelim, '1024_NoAutoResample')
    PSF_1k = load_slices(path_1k, N_pix=1024, N_crop=1024, file_list=list_slices)

    PSF_1kall = np.sum(PSF_1k, axis=0)
    PEAK_1k = np.max(PSF_1kall)
    PSF_1k /= PEAK_1k

    plt.figure()
    plt.imshow(np.log10(PSF_1kall))
    plt.title(r'PSF 1024')
    plt.show()

    f, axes = plt.subplots(1, 5)
    list_titles = ['-2', '-1', '0', '+1', '+2']
    for i, k in enumerate([17, 57, 19, 55, 21]):
        ax = plt.subplot(1, 5, i+1)
        img = ax.imshow(np.log10(PSF_1k[k//2]), vmin=-10, vmax=0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(list_titles[i])
    plt.show()

    # ================================================================================================================ #

    """ 2048 Pixels POP """
    PIX_POP = 2048

    path_2k = os.path.join(path_prelim, '2048_NoAutoResample')
    PSF_2k = load_slices(path_2k, N_pix=PIX_POP, N_crop=PIX_POP, file_list=list_slices)

    PSF_2kall = np.sum(PSF_2k, axis=0)
    PEAK = np.max(PSF_2kall)
    PSF_2k /= PEAK
    plt.figure()
    plt.imshow(np.log10(PSF_2kall))
    plt.title(r'PSF 2048')
    plt.show()

    for k in [59, 17, 57, 19, 55, 21, 53]:
        plt.figure()
        plt.imshow(np.log10(PSF_2k[k//2]))
        plt.title(list_slices[k//2])
    plt.show()


    f, axes = plt.subplots(1, 5)
    list_titles = ['-2', '-1', '0', '+1', '+2']
    for i, k in enumerate([17, 57, 19, 55, 21]):
        ax = plt.subplot(1, 5, i+1)
        img = ax.imshow(np.log10(PSF_2k[k//2]), vmin=-10, vmax=0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(list_titles[i])
    plt.show()

    # ================================================================================================================ #

    """ Compare the two samplings 1024 and 2048 """

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(crop_array(PSF_1kall, 128))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('1024')

    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow(crop_array(PSF_2kall, 256))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('2048')
    plt.show()

    # ================================================================================================================ #
    #                           NO IMAGE SLICER
    # ================================================================================================================ #

    """ Compare to NO SLICER """
    path_perfect = os.path.join(path_2k, 'Perfect PSF')
    PSF_2k_perfect = load_slices(path_perfect, N_pix=2048, N_crop=2048, file_list=[19], defocus=False)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(crop_array(PSF_2kall, 256))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('With Slicer')

    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow(crop_array(PSF_2k_perfect[0], 256))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Without Slicer')


    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = plt.subplot(1, 3, 1)
    img1 = ax1.imshow(crop_array(np.log10(PSF_2kall), 512), vmin=-13, vmax=-3)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('With Slicer')
    plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 3, 2)
    img2 = ax2.imshow(crop_array(np.log10(PSF_2k_perfect[0]), 512), vmin=-13, vmax=-3)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Without Slicer')
    plt.colorbar(img2, ax=ax2, orientation='horizontal')

    diff = np.log10(np.abs(PSF_2kall - PSF_2k_perfect[0]))

    ax3 = plt.subplot(1, 3, 3)
    img3 = ax3.imshow(crop_array(diff, 512), vmin=-13, vmax=-3)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title('Absolute Residual Difference')
    plt.colorbar(img3, ax=ax3, orientation='horizontal')

    plt.show()

    # ------------------------------------
    # Save the NO SLICER PSF

    # Fix the anamorphism

    def downsample(array, p=2):
        PIX = array.shape[0]
        pix = PIX//2

        print("\nInput array with dimensions: ", array.shape)

        # (1) Fix the anamorphism
        new_array = np.empty((pix, pix))
        min_crop = PIX // 2 - pix // 2
        max_crop = PIX // 2 + pix // 2
        for k in range(pix):
            crop = array[2 * k:2 * k + 2, min_crop:max_crop]
            new_array[k] = np.mean(crop, axis=0)

        print("Fixed Anamorphism. New array: ", new_array.shape)
        print("Downsample by %dX" %p)

        down_array = np.empty((pix//p, pix//p))
        aux_array = np.empty((pix//p, pix))
        for k in range(pix//p):     # Downsample X axis
            crop = new_array[p * k:p * k + p, :]
            aux_array[k] = np.mean(crop, axis=0)
        aux_array = aux_array.T
        for k in range(pix//p):     # Downsample Y axis
            crop = aux_array[p * k:p * k + p, :]
            down_array[k] = (np.mean(crop, axis=0))
        down_array = down_array.T
        print("Downsampled array: ", down_array.shape)

        return down_array

    PSF_anam = PSF_2k_perfect[0].copy()
    PSF_round = downsample(PSF_anam)                # Square

    plt.figure()
    plt.imshow(np.log10(PSF_round))
    plt.show()

    def save_fits(data, path, filename):
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(os.path.join(path, filename + '.fits'), overwrite=True)

    save_fits([PSF_round], path=path_perfect, filename='no_slicer_PSF')

    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #

    """ FIXED THE ANAMORPHIC DEFOCUS """

    path = os.path.abspath('D:/Thesis/LAM/POP/Slicer/9 Other Wavelengths/2.25 um/1024')
    zemax_name = 'IFU_TopAB_HARMONI_LAM_FASTPOP'
    PIX = 1024
    PSF = load_slices(path, name_zemax=zemax_name, N_pix=PIX, N_crop=PIX, file_list=list_slices, defocus=False)
    PSF_all = np.sum(PSF, axis=0)

    ### Loop Over all Defocus values
    defocus_values = [0.05, 0.10, 0.20, 0.30]
    # defocus_values = [0.10]
    path_fits = os.path.abspath('D:/Thesis/LAM/POP/Slicer/2 No Aberrations/Anamorphic Defocus')
    for focus in defocus_values:

        # Load the Defocused PSF
        path_defocus = os.path.join(path, 'Defocus %.2f' % focus)
        PSF_defocus = load_slices(path_defocus, name_zemax=zemax_name, N_pix=PIX, N_crop=PIX, file_list=list_slices,
                                  defocus=True)
        PSF_defocus_all = np.sum(PSF_defocus, axis=0)

        # Load the No Slicer Defocus
        path_defocus_no_slicer = os.path.join(path_defocus, 'No Slicer')
        PSF_defocus_no_slicer = load_slices(path_defocus_no_slicer, name_zemax=zemax_name,
                                            N_pix=PIX, N_crop=PIX, file_list=[19], defocus=True)[0]

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = plt.subplot(1, 3, 1)
        img1 = ax1.imshow(crop_array((PSF_defocus_all), 1024))
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title('With Slicer')
        plt.colorbar(img1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 3, 2)
        img2 = ax2.imshow(crop_array((PSF_defocus_no_slicer), 1024))
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title('Without Slicer')
        plt.colorbar(img2, ax=ax2, orientation='horizontal')

        diff = np.log10(np.abs(PSF_2kall - PSF_2k_perfect[0]))

        ax3 = plt.subplot(1, 3, 3)
        img3 = ax3.imshow(crop_array(diff, 512), vmin=-13, vmax=-3)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title('Absolute Residual Difference')
        plt.colorbar(img3, ax=ax3, orientation='horizontal')

        plt.show()

        # Fix the anamorphic scale and down sample
        by_pix = [1, 2, 4, 8, 16, 32]
        for p in by_pix:

            PSF_round = downsample(PSF_all, p)
            PSF_defocus_round = downsample(PSF_defocus_all, p)
            resolution = PSF_round.shape[0]

            # Save as .fits
            filename = 'Perfect_PSF_%dPIX_%.2fFOC' % (resolution, focus)
            save_fits([PSF_round, PSF_defocus_round], path=path_fits, filename=filename)






    defocus = 0.10
    path_defocus = os.path.join(path, 'Defocus %.2f' %defocus)
    PSF_defocus = load_slices(path_defocus, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=True)
    PSF_defocus_all = np.sum(PSF_defocus, axis=0)

    defocus2 = 0.20
    path_defocus2 = os.path.join(path, 'Defocus %.2f' %defocus2)
    PSF_defocus2 = load_slices(path_defocus2, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=True)
    PSF_defocus_all2 = np.sum(PSF_defocus2, axis=0)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1 = plt.subplot(2, 2, 1)
    img1 = ax1.imshow(crop_array(PSF_all, 512))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(r'Nominal PSF [Zoom 512 pix]')
    # plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(2, 2, 2)
    img2 = ax2.imshow(crop_array(PSF_defocus_all2, 512))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus %.2f $\lambda$' %defocus2)
    # plt.colorbar(img2, ax=ax2, orientation='horizontal')

    # diff = np.log10(np.abs(PSF_2kall - PSF_2k_perfect[0]))

    ax3 = plt.subplot(2, 2, 3)
    img3 = ax3.imshow(np.log10(PSF_all))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'Nominal PSF Log10 [2048 pix]')
    # plt.colorbar(img3, ax=ax3, orientation='horizontal')

    ax4 = plt.subplot(2, 2, 4)
    img4 = ax4.imshow(np.log10(PSF_defocus_all2))
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title(r'Defocus %.2f $\lambda$' %defocus2)

    plt.show()

    path_defocus_no_slicer2 = os.path.join(path_defocus2, 'No Slicer')
    PSF_defocus_no_slicer2 = load_slices(path_defocus_no_slicer2, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=[19], defocus=True)[0]

    #### Compare Defocus With and Without

    pp = np.max(PSF_defocus_no_slicer2)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1 = plt.subplot(2, 2, 1)
    img1 = ax1.imshow(crop_array(PSF_defocus_all2/pp, 512))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(r'Nominal PSF [Zoom 512 pix]')
    plt.colorbar(img1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(2, 2, 2)
    img2 = ax2.imshow(crop_array(PSF_defocus_no_slicer2/pp, 512))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus %.2f $\lambda$' %defocus2)
    plt.colorbar(img2, ax=ax2, orientation='horizontal')

    diff = PSF_defocus_no_slicer2/pp - PSF_defocus_all2/pp

    ax3 = plt.subplot(2, 2, 3)
    img3 = ax3.imshow(crop_array(diff,512))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'Nominal PSF Log10 [2048 pix]')
    plt.colorbar(img3, ax=ax3, orientation='horizontal')

    ax4 = plt.subplot(2, 2, 4)
    img4 = ax4.imshow(np.log10(PSF_defocus_all2))
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title(r'Defocus %.2f $\lambda$' %defocus2)

    plt.show()

    # ================================================================================================================ #
    #                       NO ABERRATIONS - Different Resolutions
    # ================================================================================================================ #
    path_resolution = os.path.abspath('D:/Thesis/LAM/POP/Slicer/2 No Aberrations/Different Resolutions')
    zemax_name = 'IFU_TopAB_HARMONI_LAM_FASTPOP'

    # Load the 2048 pixels PSF
    path_res = os.path.join(path_resolution, '2048')
    PSF = load_slices(path_res, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=False)
    PSF_all = np.sum(PSF, axis=0)

    defocus = 0.10
    path_defocus = os.path.join(path_res, 'Defocus %.2f' %defocus)
    PSF_defocus = load_slices(path_defocus, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=True)
    PSF_defocus_all = np.sum(PSF_defocus, axis=0)

    # Fix the anamorphic scale and down sample
    by_pix = [1, 2, 4, 8, 16, 32]
    for p in by_pix:

        PSF_round = downsample(PSF_all, p)
        PSF_defocus_round = downsample(PSF_defocus_all, p)
        resolution = PSF_round.shape[0]

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1 = plt.subplot(1, 3, 1)
        img1 = ax1.imshow(np.log10(PSF_round))
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title('Nominal [%d pix]' %resolution)

        ax2 = plt.subplot(1, 3, 2)
        img2 = ax2.imshow(np.log10(PSF_defocus_round))
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(r'Defocus $%.2f \lambda$' % defocus)

        diff = PSF_round - PSF_defocus_round
        cmin = min(np.min(diff), -np.max(diff))

        ax3 = plt.subplot(1, 3, 3)
        img3 = ax3.imshow(diff, cmap='seismic', vmin=cmin, vmax=-cmin)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(r'Difference: Nominal - Defocus')

        filename = 'PSF_Resolution_%dpix_foc%.2f' % (resolution, defocus)
        save_fits([PSF_round, PSF_defocus_round], path=path_resolution, filename=filename)

    plt.show()

    defocus2 = 0.20
    path_defocus2 = os.path.join(path_res, 'Defocus %.2f' %defocus2)
    PSF_defocus2 = load_slices(path_defocus2, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=True)
    PSF_defocus_all2 = np.sum(PSF_defocus2, axis=0)

    defocus3 = 0.30
    path_defocus3 = os.path.join(path_res, 'Defocus %.2f' %defocus3)
    PSF_defocus3 = load_slices(path_defocus3, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=True)
    PSF_defocus_all3 = np.sum(PSF_defocus3, axis=0)

    p = 32
    PSF_round = downsample(PSF_all, p)
    PSF_defocus_round = downsample(PSF_defocus_all, p)
    PSF_defocus_round2 = downsample(PSF_defocus_all2, p)
    PSF_defocus_round3 = downsample(PSF_defocus_all3, p)
    p_pix = PSF_round.shape[0]
    k_pix = p_pix//2

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1 = plt.subplot(1, 4, 1)
    img1 = ax1.imshow(crop_array(PSF_round, k_pix))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Nominal [%d pix]' %p_pix)

    ax2 = plt.subplot(1, 4, 2)
    img2 = ax2.imshow(crop_array(PSF_defocus_round, k_pix))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus $%.2f \lambda$' % defocus)

    ax3 = plt.subplot(1, 4, 3)
    img3 = ax3.imshow(crop_array(PSF_defocus_round2, k_pix))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'Defocus $%.2f \lambda$' % defocus2)

    ax4 = plt.subplot(1, 4, 4)
    img4 = ax4.imshow(crop_array(PSF_defocus_round3, k_pix))
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title(r'Defocus $%.2f \lambda$' % defocus3)

    plt.show()

    diff = PSF_round - PSF_defocus_round
    cmin = min(np.min(diff), -np.max(diff))

    # ----------------------------------------------------------------------------------- #

    """ Compare With / Without Slicer for each Defocus """

    path_defocus_no_slicer = os.path.join(path_defocus, 'No Slicer')
    PSF_defocus_no_slicer = load_slices(path_defocus_no_slicer, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=[19], defocus=True)[0]
    PSF_defocus_round_no_slicer = downsample(PSF_defocus_no_slicer, p=1)

    path_defocus_no_slicer2 = os.path.join(path_defocus2, 'No Slicer')
    PSF_defocus_no_slicer2 = load_slices(path_defocus_no_slicer2, name_zemax=zemax_name, N_pix=2048, N_crop=2048, file_list=[19], defocus=True)[0]
    PSF_defocus_round_no_slicer2 = downsample(PSF_defocus_no_slicer2, p=1)

    k_pix = 256
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    ax1 = plt.subplot(2, 4, 1)
    img1 = ax1.imshow(crop_array(PSF_round, k_pix))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Nominal [1024 pix]')

    ax2 = plt.subplot(2, 4, 2)
    img2 = ax2.imshow(crop_array(PSF_defocus_round, k_pix))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus $%.2f \lambda$ [Slicer]' % defocus)

    ax3 = plt.subplot(2, 4, 3)
    img3 = ax3.imshow(crop_array(PSF_defocus_round_no_slicer, k_pix))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'Defocus $%.2f \lambda$ [No Slicer]' % defocus)

    diff = PSF_defocus_round - PSF_defocus_round_no_slicer
    cmin = min(np.min(diff), -np.max(diff))

    ax4 = plt.subplot(2, 4, 4)
    img4 = ax4.imshow(crop_array(diff, k_pix), cmap='bwr', vmin=cmin, vmax=-cmin)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title(r'Difference: With - Without Slicer]')

    # ----------------------------------------------------- #

    ax5 = plt.subplot(2, 4, 5)
    img5 = ax5.imshow(crop_array(PSF_round, k_pix))
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax5.set_title('Nominal [1024 pix]')

    ax6 = plt.subplot(2, 4, 6)
    img6 = ax6.imshow(crop_array(PSF_defocus_round2, k_pix))
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax6.set_title(r'Defocus $%.2f \lambda$ [Slicer]' % defocus2)

    ax7 = plt.subplot(2, 4, 7)
    img7 = ax7.imshow(crop_array(PSF_defocus_round_no_slicer2, k_pix))
    ax7.get_xaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)
    ax7.set_title(r'Defocus $%.2f \lambda$ [No Slicer]' % defocus2)

    diff = PSF_defocus_round2 - PSF_defocus_round_no_slicer2
    cmin = min(np.min(diff), -np.max(diff))

    ax8 = plt.subplot(2, 4, 8)
    img8 = ax8.imshow(crop_array(diff, k_pix), cmap='bwr', vmin=cmin, vmax=-cmin)
    ax8.get_xaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)
    ax8.set_title(r'Difference: With - Without Slicer]')

    plt.show()





    # ================================================================================================================ #
    #                       DEFOCUS
    # ================================================================================================================ #

    path_2k_defoc = os.path.join(path_2k, '0.15 Defocus')
    PSF_2k_foc_a = load_slices(path_2k_defoc, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=True)

    PSF_2k_foc_all_a = np.sum(PSF_2k_foc_a, axis=0)
    # PEAK = np.max(PSF_2k_foc_all)
    # PSF_2k_foc_all /= PEAK
    plt.figure()
    plt.imshow(np.log10(PSF_2k_foc_all_a[1]))
    plt.title(r'PSF 2048')
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(np.log10(PSF_2kall), vmin=-10, vmax=-3)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Nominal')

    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow(np.log10(PSF_2k_foc_all_a), vmin=-10, vmax=-3)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus $0.15 \lambda$')
    plt.show()

    ## Linear Scale

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(crop_array(PSF_2kall, 256))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Nominal')

    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow(crop_array(PSF_2k_foc_all_a, 256))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus $0.15 \lambda$')
    plt.show()

    # ================================================================================================================ #
    path_2k_defoc = os.path.join(path_2k, '0.10 Defocus')
    PSF_2k_foc_b = load_slices(path_2k_defoc, N_pix=2048, N_crop=2048, file_list=list_slices, defocus=True)
    PSF_2k_foc_all_b = np.sum(PSF_2k_foc_b, axis=0)

    plt.figure()
    plt.imshow(np.log10(PSF_2k_foc_all_b))
    plt.title(r'Defocus $0.10 \lambda$')
    plt.show()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = plt.subplot(1, 3, 1)
    img1 = ax1.imshow(np.log10(PSF_2kall))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Nominal')

    ax2 = plt.subplot(1, 3, 2)
    img2 = ax2.imshow(np.log10(PSF_2k_foc_all_b))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus $0.10 \lambda$')

    ax3 = plt.subplot(1, 3, 3)
    img3 = ax3.imshow(np.log10(PSF_2k_foc_all_a[1]))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'Defocus $0.15 \lambda$')
    plt.show()

    ### Linear scale

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1 = plt.subplot(1, 3, 1)
    img1 = ax1.imshow(crop_array(PSF_2kall, 256))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Nominal')

    ax2 = plt.subplot(1, 3, 2)
    img2 = ax2.imshow(crop_array(PSF_2k_foc_all_b, 256))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(r'Defocus $0.10 \lambda$')

    ax3 = plt.subplot(1, 3, 3)
    img3 = ax3.imshow(crop_array(PSF_2k_foc_all_a[1], 256))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'Defocus $0.15 \lambda$')
    plt.show()

