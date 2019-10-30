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

def load_slices(path, file_list, N_pix=N_pix, N_crop=N_crop, defocus=False):
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

        name_nominal = 'IFU_TopAB_HARMONI_LAM' + ' 0_'

        pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
        for k in range(N):
            PSFs[k, :, :] = pop_slicer_nom.beam_data[k][min_pix:max_pix, min_pix:max_pix]

    if defocus:

        pop_slicer_foc = POP_Slicer()
        # PSFs = np.empty((N, 2, N_crop, N_crop))
        PSFs = np.empty((N, N_crop, N_crop))

        # name_nominal = 'IFU_TopAB_HARMONI_LAM' + ' 0_'
        name_defocus = 'IFU_TopAB_HARMONI_LAM' + ' 0_FOC_'

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

