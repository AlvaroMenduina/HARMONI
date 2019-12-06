"""
Attempt to simulate semi-realistic Image Slicer effects with Python
Based on a 1D approach


"""

import os
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# SPAXEL SCALE
ELT_DIAM = 39
CENTRAL_OBS = 0.30                  # Central obscuration is ~30% of the diameter
MILIARCSECS_IN_A_RAD = 206265000


def rho_spaxel_scale(spaxel_scale=4, wavelength=1.5):
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

if __name__ == """__main__""":

    # Sanity check on the sizes of the apertures to match a 4 m.a.s. spaxel scale at 1.5 um
    nom_wave = 1.5          # microns
    spaxels = 4             # m.a.s
    _rho = rho_spaxel_scale(spaxel_scale=spaxels, wavelength=nom_wave)
    check_spaxel_scale(rho_aper=_rho, wavelength=nom_wave)

    class SlicerModel(object):

        def __init__(self, N_PIX, spaxel_scale, N_waves=1, wave0=1.5, waveN=1.5):

            self.N_PIX = N_PIX
            self.spaxel_scale = spaxel_scale
            self.create_pupil_masks(spaxel_scale, N_waves, wave0, waveN)


            #TODO: add the possibility of looping over multiple wavelengths


        def create_pupil_masks(self, spaxel_scale, N_waves=1, wave0=1.5, waveN=1.5):
            """
            Creates a dictionary of pupil masks for a set of wavelengths
            The pupil masks change in size to account for the variation of PSF size with wavelength
            We set the spaxel_scale for the first wavelength [wave0]
            :param spaxel_scale: spaxel_scale [mas] for wave0
            :param N_waves: number of wavelengths to consider in the interval [wave0, waveN]
            :param wave0: nominal wavelength in [microns]
            :param waveN: maximum wavelength in [microns]
            :return:
            """

            self.wave_range = np.linspace(wave0, waveN, N_waves, endpoint=True)
            self.waves_ratio = np.linspace(1., waveN / wave0, N_waves, endpoint=True)

            rho_aper = rho_spaxel_scale(spaxel_scale, wavelength=wave0)
            rho_obsc = CENTRAL_OBS * rho_aper

            self.pupil_masks = {}
            x0 = np.linspace(-1., 1., self.N_PIX, endpoint=True)
            xx, yy = np.meshgrid(x0, x0)
            rho = np.sqrt(xx ** 2 + yy ** 2)
            for i, wave in enumerate(self.waves_ratio):
                wavelength = self.wave_range[i]
                pupil = (rho <= rho_aper / wave) & (rho >= rho_obsc / wave)
                self.pupil_masks[wavelength] = pupil
            return

        def create_slicer_masks(self, spaxels_per_slice, N_slices):

            # Check the size of the arrays
            U = self.N_PIX * self.spaxel_scale          # Size in [m.a.s] of the Slicer Plane
            u0 = np.linspace(-U/2., U/2., self.N_PIX, endpoint=True)
            uu, vv = np.meshgrid(u0, u0)

            cent_pix = self.N_PIX // 2          # Central pixel
            slice_width = spaxels_per_slice * self.spaxel_scale
            self.slicer_masks = []
            #TODO: what if N_slices is not odd??
            N_slices_above = (N_slices - 1) // 2
            lower_centre = -N_slices_above * slice_width
            for i in range(N_slices):
                centre = lower_centre + i * slice_width
                up_mask = vv > (centre - slice_width//2)
                low_mask = vv < (centre + slice_width//2)
                mask = up_mask * low_mask
                plt.figure()
                plt.imshow(mask)
            return

        def propagate_pupil_to_slicer(self, wavelength, wavefront):

            # FIXME: possibility of using other wavefronts

            pupil_mask = self.pupil_masks[wavelength]
            wavefront = pupil_mask
            complex_pupil = pupil_mask * np.exp(1j * 2*np.pi * wavefront / wavelength)

            # Propagate to Slicer plane
            complex_slicer = fftshift(fft2(complex_pupil, norm='ortho'))

            slicer_intensity = (np.abs(complex_slicer))**2

            plt.figure()
            plt.imshow(slicer_intensity)
            plt.colorbar()

            return

    slicer = SlicerModel(N_PIX=2048, spaxel_scale=1, N_waves=5, wave0=1.5, waveN=2.5)
    for wave in slicer.wave_range:
        plt.figure()
        plt.imshow(slicer.pupil_masks[wave])
        plt.title('Pupil Mask at %.2f microns' % wave)
    plt.show()

    slicer.propagate_pupil_to_slicer(slicer.wave_range[0], wavefront=0)
    slicer.create_slicer_masks(spaxels_per_slice=10, N_slices=5)
    plt.show()