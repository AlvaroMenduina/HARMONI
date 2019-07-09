"""
==========================================================
                    SWIFT Slicer POP Analysis
==========================================================

SWIFT Image Slicer simulations using Zemax POP
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import zern_core as zern
from pyzdde.zdde import readBeamFile

wave_nom = 850          # nmn

list_slices = [18, 19, 20, 21, 22]
i_central = 20

# POP arrays - Nyquist sampled PSF
x_size = 3.328           # Physical size of arraxy at Image Plane
N_pix = 512              # Number of pixels in the Zemax BFL
N_crop = 512             # Crop to a smaller region around the PSF
min_pix = N_pix//2 - N_crop//2
max_pix = N_pix//2 + N_crop//2
extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]
xc = np.linspace(-x_size / 2, x_size / 2, 10)

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


def load_files(path, file_list, N):
    """
    Loads the Zemax beam files, constructs the PSFs
    and normalizes everything by the intensity of the PSF
    at i_norm (the Nominal PSF)
    """

    pop_slicer_nom = POP_Slicer()
    pop_slicer_foc = POP_Slicer()

    flat_PSFs = np.empty((N, 2 * N_crop * N_crop))
    PSFs = np.empty((N, 2, N_crop, N_crop))

    for k in range(N):
        if k < 10:
            # We have to adjust for the ZBF format. Before 10 it adds a space []3
            name_nominal = 'SWIFT_SLICER_FINAL_VERSION' + '% d_' % k
            name_defocus = 'SWIFT_SLICER_FINAL_VERSION' + '% d_FOC_' % k
        else:
            name_nominal = 'SWIFT_SLICER_FINAL_VERSION' + '%d_' % k
            name_defocus = 'SWIFT_SLICER_FINAL_VERSION' + '%d_FOC_' % k

        pop_slicer_nom.get_zemax_files(path, name_nominal, file_list)
        slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

        # pop_slicer_foc.get_zemax_files(path, name_defocus, file_list)
        # slicers_foc = np.sum(pop_slicer_foc.beam_data, axis=0)[min_pix:max_pix, min_pix:max_pix]

        # PSFs[k, 0, :, :], PSFs[k, 1, :, :] = slicers_nom, slicers_foc
        PSFs[k, 0, :, :], PSFs[k, 1, :, :] = slicers_nom, slicers_nom
        # flat_PSFs[k, :] = np.concatenate((slicers_nom.flatten(), slicers_foc.flatten()))
        flat_PSFs[k, :] = np.concatenate((slicers_nom.flatten(), slicers_nom.flatten()))

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

if __name__ == "__main__":

    N = 1
    path_pop = os.path.abspath('D:\Research\Experimental\SWIFT\POP')
    
    PSFs = load_files(path_pop, N=N, file_list=list_slices)
    PEAK = np.max(PSFs[1])
    PSFs[0] /= PEAK
    PSFs[1] /= PEAK

    # Show SWIFT PSF
    PSF_swift = PSFs[1][0,0]
    plt.figure()
    plt.imshow(PSF_swift)
    plt.colorbar()
    plt.show()

