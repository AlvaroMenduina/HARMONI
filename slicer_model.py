"""
Attempt to simulate semi-realistic Image Slicer effects with Python
Based on a 1D approach


"""

import os
import numpy as np
from numpy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from time import time

import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import skcuda
import skcuda.linalg as clinalg
import skcuda.fft as cu_fft

# SPAXEL SCALE
ELT_DIAM = 39
CENTRAL_OBS = 0.0                  # Central obscuration is ~30% of the diameter for the ELT
MILIARCSECS_IN_A_RAD = 206265000


def crop_array(array, crop):
    """
    Crops an array or datacube for various cases of shapes to a smaller dimesion
    Typically used to zoom in for the PSF arrays
    :param array: can be [Pix, Pix] or [:, Pix, Pix] or [:, Pix, Pix, :]
    :param crop:
    :return:
    """
    shape = array.shape

    if len(shape) == 2:         # Classic [Pix, Pix] array
        PIX = array.shape[0]
        if crop > PIX:
            raise ValueError("Array is only %d x %d pixels. Trying to crop to %d x %d" % (PIX, PIX, crop, crop))
        min_crop = (PIX + 1 - crop) // 2
        max_crop = (PIX + 1 + crop) // 2
        array_crop = array[min_crop:max_crop, min_crop:max_crop]

    if len(shape) == 3:         # [N_PSF, Pix, Pix] array
        N_PSF, PIX = array.shape[0], array.shape[1]
        if crop > PIX:
            raise ValueError("Array is only %d x %d pixels. Trying to crop to %d x %d" % (PIX, PIX, crop, crop))
        min_crop = (PIX + 1 - crop) // 2
        max_crop = (PIX + 1 + crop) // 2
        array_crop = array[:, min_crop:max_crop, min_crop:max_crop]

    if len(shape) == 4:         # [N_PSF, Pix, Pix, N_channels] array
        N_PSF, PIX = array.shape[0], array.shape[1]
        if crop > PIX:
            raise ValueError("Array is only %d x %d pixels. Trying to crop to %d x %d" % (PIX, PIX, crop, crop))
        min_crop = (PIX + 1 - crop) // 2
        max_crop = (PIX + 1 + crop) // 2
        array_crop = array[:, min_crop:max_crop, min_crop:max_crop, :]

    return array_crop


def rho_spaxel_scale(spaxel_scale=4.0, wavelength=1.5):
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


def compute_FWHM(wavelength):
    """
    Compute the Full Width Half Maximum of the PSF at a given Wavelength
    in miliarcseconds, for the ELT
    :param wavelength: [microns]
    :return:
    """
    FWHM_RAD = wavelength * 1e-6 / ELT_DIAM          # radians
    FWHM_MAS = FWHM_RAD * MILIARCSECS_IN_A_RAD
    return FWHM_MAS

def pupil_mask(xx, yy, rho_aper, rho_obsc, anamorphic=False):

    if anamorphic == False:
        rho = np.sqrt(xx ** 2 + yy ** 2)
    elif anamorphic == True:
        rho = np.sqrt(xx ** 2 + (2 * yy) ** 2)
    pupil = (rho <= rho_aper) & (rho >= rho_obsc)
    return pupil


# ============================================================================== #
#                                ZEMAX INTERFACE                                 #
# ============================================================================== #
import pyzdde.zdde
from pyzdde.zfileutils import readBeamFile

class POP_Slicer(object):
    """
    Physical Optics Propagation (POP) analysis of an Image Slicer
    """

    def __init__(self):
        pass

    def read_beam_file(self, file_name):
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

    def read_all_zemax_files(self, path_zemax, name_convention, file_list):
        """
        Goes through the ZBF Zemax Beam Files of all Slices and
        extracts the beam information (X_size, Y_size) etc
        as well as the Irradiance distribution

        :param path_zemax:
        :param name_convention: typically the Zemax file name that generates the POP arrays
        :param file_list: list of indices for the files, typically each index corresponds to one slice
        """
        info, data, powers = [], [], []
        start = time()
        N_slices = len(file_list)
        print("\nReading %d Zemax Beam Files" % N_slices)
        for k in file_list:
            print('\n======================================')

            if k < 10:      # The Zemax naming format changes after 10
                file_id = name_convention + ' ' + str(k) + '_POP.ZBF'
            else:
                file_id = name_convention + str(k) + '_POP.ZBF'
            file_name = os.path.join(path_zemax, file_id)

            print('Reading Beam File: ', file_id)

            NM, deltas, beam_data, power = self.read_beam_file(file_name)
            Dx, Dy = NM[0] * deltas[0], NM[1] * deltas[1]
            # info.append([k, Dx, Dy])
            data.append(beam_data)
            # powers.append(power)

        # beam_info = np.array(info)
        irradiance_values = np.array(data)
        # powers = np.array(powers)

        POP_PSF = np.sum(irradiance_values, axis=0)
        end = time()
        print("Total time: %.2f sec" % (end - start))
        print("1 slice: %.2f sec" % ((end - start)/N_slices))
        return POP_PSF, irradiance_values


class SlicerModel(object):
    """
    Object that models the effect of Image Slicers in light propagation
    """

    def __init__(self, slicer_options, N_PIX, spaxel_scale, N_waves=1, wave0=1.5, waveN=1.5):

        self.N_slices = slicer_options["N_slices"]                                          # Number os Slices to use
        self.spaxels_per_slice = slicer_options["spaxels_per_slice"]                        # Size of the slice in Spaxels
        self.pupil_mirror_aperture = slicer_options["pupil_mirror_aperture"]                # Pupil Mirror Aperture
        N_rings = self.spaxels_per_slice / 2
        self.anamorphic = slicer_options["anamorphic"]                                      # Anamorphic Preoptics?
        self.slice_size_mas = self.spaxels_per_slice * spaxel_scale                         # Size of 1 slice in mas
        self.spaxel_scale = spaxel_scale                                                    # Spaxel scale [mas]
        self.N_PIX = N_PIX

        print("\nCreating Image Slicer Model")
        print("     Nominal Wavelength: %.2f microns" % spaxel_scale)
        print("     Spaxel Scale: %.2f mas" % wave0)
        print("     Number of slices: ", self.N_slices)
        print("     Spaxels per slices: ", self.spaxels_per_slice)
        FWHM = compute_FWHM(wave0)
        self.FWHM_ratio = self.slice_size_mas / FWHM            # How much of the FWHM is covered by one slice
        print("     FWHM: %.2f mas" % FWHM)
        print("     One slice covers: %.2f mas / %.2f FWHM" % (self.slice_size_mas, self.FWHM_ratio))

        self.create_pupil_masks(spaxel_scale, N_waves, wave0, waveN)
        self.create_slicer_masks()
        self.create_pupil_mirror_apertures(N_rings=self.pupil_mirror_aperture * N_rings)

        self.compute_peak_nominal_PSFs()

    def compute_peak_nominal_PSFs(self):
        """
        Compute the PEAK of the PSF without aberrations so that we can
        normalize everything by it
        """
        self.nominal_PSFs, self.PEAKs = {}, {}
        for wavelength in self.wave_range:
            pupil_mask = self.pupil_masks[wavelength]
            nominal_pup = pupil_mask * np.exp(1j * 2 * np.pi * pupil_mask / wavelength)
            nominal_PSF = (np.abs(fftshift(fft2(nominal_pup, norm='ortho')))) ** 2
            self.nominal_PSFs[wavelength] = nominal_PSF
            self.PEAKs[wavelength] = np.max(nominal_PSF)
        return

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
        self.diameters = 1/rho_aper
        # check_spaxel_scale(rho_aper=rho_aper, wavelength=wave0)
        rho_obsc = CENTRAL_OBS * rho_aper

        self.pupil_masks, self.pupil_masks_fft = {}, {}
        x0 = np.linspace(-1., 1., self.N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x0, x0)

        print("\nCreating Pupil Masks:")
        for i, wave in enumerate(self.waves_ratio):
            wavelength = self.wave_range[i]
            print("Wavelength: %.2f microns" % wavelength)
            # pupil = (rho <= rho_aper / wave) & (rho >= rho_obsc / wave)
            _pupil = pupil_mask(xx, yy, rho_aper / wave, rho_obsc / wave, self.anamorphic)
            mask = np.array(_pupil).astype(np.float32)
            self.pupil_masks[wavelength] = mask
            self.pupil_masks_fft[wavelength] = np.stack(mask * self.N_slices)       # For the GPU calculations
        return

    def create_slicer_masks(self):
        """
        Creates a list of slicer masks that model the finite aperture por each slice
        effectively cropping the PSF at the focal plane
        :return:
        """

        # Check the size of the arrays
        U = self.N_PIX * self.spaxel_scale          # Size in [m.a.s] of the Slicer Plane
        u0 = np.linspace(-U/2., U/2., self.N_PIX, endpoint=True)
        uu, vv = np.meshgrid(u0, u0)

        slice_width = self.slice_size_mas
        self.slicer_masks, self.slice_boundaries = [], []
        self.slicer_masks_fftshift = []
        #TODO: what if N_slices is not odd??
        N_slices_above = (self.N_slices - 1) // 2
        lower_centre = -N_slices_above * slice_width
        for i in range(self.N_slices):
            centre = lower_centre + i * slice_width
            bottom = centre - slice_width/2
            self.slice_boundaries.append(bottom)
            up_mask = vv > (centre - slice_width/2)
            low_mask = vv < (centre + slice_width/2)
            mask = up_mask * low_mask

            self.slicer_masks.append(mask)
            self.slicer_masks_fftshift.append(fftshift(mask))           # For the GPU calculations
        self.slicer_masks_fftshift = np.array(self.slicer_masks_fftshift).astype(np.float32)

        return

    def create_pupil_mirror_apertures(self, N_rings):
        """
        Creates a mask to model the finite aperture of the pupil mirror,
        which effectively introduces fringe effects on the PSF at the exit slit
        :param aperture:
        :return:
        """
        # The number of PSF zeros (or rings) that we see in the Pupil Mirror plane is
        # equal to half the number of spaxels per slice we have defined, along each direction
        # i.e. if we have 20 spaxels_per_slice, we see 10 zeros above and 10 below the PSF core

        N_zeros = self.spaxels_per_slice / 2.
        aperture_ratio = N_rings / N_zeros
        x0 = np.linspace(-1., 1., self.N_PIX, endpoint=True)
        xx, yy = np.meshgrid(x0, x0)

        self.pupil_mirror_mask, self.pupil_mirror_masks_fft = {}, {}

        print("\nCreating Pupil Mirror Apertures:")
        for i, wave in enumerate(self.waves_ratio):
            wavelength = self.wave_range[i]
            print("Wavelength: %.2f microns" % wavelength)
            _pupil = np.abs(yy) <= aperture_ratio / wave
            mask = np.array(_pupil).astype(np.float32)
            self.pupil_mirror_mask[wavelength] = mask
            self.pupil_mirror_masks_fft[wavelength] = np.stack(mask * self.N_slices)       # For the GPU calculations

        return

    def propagate_pupil_to_slicer(self, wavelength, wavefront):
        """
        Takes the a given wavefront at the PUPIL plane at a given wavelength
        and propagates it to the focal plane, i.e. SLICER plane
        using standard Fourier transforms
        :param wavelength:
        :param wavefront: a wavefront map
        :return: complex electric field at the slicer
        """

        # print("Pupil Plane -> Image Slicer Plane")

        pupil_mask = self.pupil_masks[wavelength]
        # wavefront = pupil_mask
        complex_pupil = pupil_mask * np.exp(1j * 2*np.pi * wavefront / wavelength)

        # Propagate to Slicer plane
        complex_slicer = fftshift(fft2(complex_pupil, norm='ortho'))

        return complex_slicer

    def propagate_slicer_to_pupil_mirror(self, complex_slicer):
        """
        Using the SLICER MASKS, it masks the complex field at the SLICER plane
        and propagates each slice to the PUPIL MIRROR plane using inverse Fourier transform
        :param complex_slicer: complex electric field at the SLICER plane
        :return:
        """

        # print("Image Slicer Plane -> Pupil Mirror Plane")
        complex_mirror = []
        for i_slice in range(self.N_slices):        # Loop over the Slices
            mask = self.slicer_masks[i_slice]
            masked_complex_slicer = mask * complex_slicer
            # Pre FFT-Shift to put it back to the format the FFT uses
            _shifted = fftshift(masked_complex_slicer)
            # propagate to Pupil Mirror plane
            complex_mirror.append(ifft2(_shifted, norm='ortho'))
        return complex_mirror

    def propagate_pupil_mirror_to_exit_slit(self, complex_mirror, wavelength):
        """
        Using the PUPIL MIRROR MASK, it propagates each slice to the corresponding
        exit slit
        :param complex_mirror: complex electric field at the PUPIL MIRROR plane [a list of slices]
        :param wavelength: wavelength at which to rescale the PUPIL MIRROR apertures
        """

        # print("Pupil Mirror Plane -> Exit Slits")
        exit_slits = []
        for c_mirror in complex_mirror:
            masked_mirror = self.pupil_mirror_mask[wavelength] * c_mirror
            complex_slit = fftshift(fft2(masked_mirror, norm='ortho'))
            exit_slits.append((np.abs(complex_slit))**2)
        image = np.sum(np.stack(exit_slits), axis=0)

        return exit_slits, image

    def propagate_gpu_wavelength(self, wavelength, wavefront, N):
        """
        Propagation from Pupil Plane to Exit Slit on the GPU for a single wavelength

        Repeated N times to show how it runs much faster on the GPU when we want to compute
        many PSF images
        :param wavefront:
        :return:
        """
        # It is a pain in the ass to handle the memory properly on the GPU when you have [N_slices, N_pix, N_pix]
        # arrays
        print("\nPropagating on the GPU")
        # GPU memory management
        free, total = cuda.mem_get_info()
        print("Memory Start | Free: %.2f percent" % (free/total*100))
        slicer_masks_gpu = gpuarray.to_gpu(self.slicer_masks_fftshift)
        mirror_mask_gpu = gpuarray.to_gpu(self.pupil_mirror_masks_fft[wavelength])

        plan_batch = cu_fft.Plan((self.N_PIX, self.N_PIX), np.complex64, np.complex64, self.N_slices)

        # Allocate GPU arrays that will be overwritten with skcuda.misc.set_realloc to save memory
        _pupil = np.zeros((self.N_PIX, self.N_PIX), dtype=np.complex64)
        complex_pupil_gpu = gpuarray.to_gpu(_pupil)

        _slicer = np.zeros((self.N_slices, self.N_PIX, self.N_PIX), dtype=np.complex64)
        complex_slicer_gpu = gpuarray.to_gpu(_slicer)

        PSF_images = []
        for i in range(N):

            # Pupil Plane -> Image Slicer
            pupil_mask = self.pupil_masks[wavelength]
            complex_pupil = pupil_mask * np.exp(1j * 2 * np.pi * pupil_mask / wavelength)
            skcuda.misc.set_realloc(complex_pupil_gpu, np.asarray(complex_pupil, np.complex64))
            cu_fft.fft(complex_pupil_gpu, complex_pupil_gpu, plan_batch)

            # Add N_slices copies to be Masked
            complex_slicer_cpu = complex_pupil_gpu.get()
            complex_slicer_cpu = np.stack([complex_slicer_cpu] * self.N_slices)
            skcuda.misc.set_realloc(complex_slicer_gpu, complex_slicer_cpu)
            clinalg.multiply(slicer_masks_gpu, complex_slicer_gpu, overwrite=True)

            # Image Slicer -> Pupil Mirror
            cu_fft.ifft(complex_slicer_gpu, complex_slicer_gpu, plan_batch, True)
            clinalg.multiply(mirror_mask_gpu, complex_slicer_gpu, overwrite=True)

            # Pupil Mirror -> Exit Slits
            cu_fft.fft(complex_slicer_gpu, complex_slicer_gpu, plan_batch)

            # pycuda.cumath.fabs(complex_slicer_gpu, out=complex_slicer_gpu)

            _slits = complex_slicer_gpu.get()
            slits = np.sum((np.abs(_slits))**2, axis=0)
            PSF_images.append(slits)

            # free, total = cuda.mem_get_info()
            # print("Memory Usage | Free: %.2f percent" % (free / total * 100))

            # free, total = cuda.mem_get_info()
            # print("Memory End | Free: %.2f percent" % (free/total*100))

        # Make sure you clean up the memory so that it doesn't blow up!!
        complex_pupil_gpu.gpudata.free()
        complex_slicer_gpu.gpudata.free()
        slicer_masks_gpu.gpudata.free()
        mirror_mask_gpu.gpudata.free()
        free, total = cuda.mem_get_info()
        print("Memory Final | Free: %.2f percent" % (free / total * 100))

        return fftshift(np.array(PSF_images), axes=(1, 2))

    def propagate_eager(self, wavelength, wavefront):
        """
        'Not-Too-Good' version of the propagation on the GPU (lots of Memory issues...)
        Remove in the future
        :param wavelength:
        :param wavefront:
        :return:
        """

        N = self.N_PIX
        # free, total = cuda.mem_get_info()
        free, total = cuda.mem_get_info()
        print("Free: %.2f percent" %(free/total*100))

        # Pupil Plane -> Image Slicer
        complex_pupil = self.pupil_masks[wavelength] * np.exp(1j * 2 * np.pi * self.pupil_masks[wavelength] / wavelength)
        complex_pupil_gpu = gpuarray.to_gpu(np.asarray(complex_pupil, np.complex64))
        plan = cu_fft.Plan(complex_pupil_gpu.shape, np.complex64, np.complex64)
        cu_fft.fft(complex_pupil_gpu, complex_pupil_gpu, plan, scale=True)

        # Add N_slices copies to be Masked
        complex_slicer_cpu = complex_pupil_gpu.get()
        complex_pupil_gpu.gpudata.free()

        free, total = cuda.mem_get_info()
        print("*Free: %.2f percent" %(free/total*100))

        complex_slicer_cpu = np.stack([complex_slicer_cpu]*self.N_slices)
        complex_slicer_gpu = gpuarray.to_gpu(complex_slicer_cpu)
        slicer_masks_gpu = gpuarray.to_gpu(self.slicer_masks_fftshift)
        clinalg.multiply(slicer_masks_gpu, complex_slicer_gpu, overwrite=True)
        slicer_masks_gpu.gpudata.free()
        free, total = cuda.mem_get_info()
        print("**Free: %.2f percent" %(free/total*100))

       # Slicer -> Pupil Mirror
        plan = cu_fft.Plan((N, N), np.complex64, np.complex64, self.N_slices)
        cu_fft.ifft(complex_slicer_gpu, complex_slicer_gpu, plan, scale=True)
        mirror_mask_gpu = gpuarray.to_gpu(self.pupil_mirror_masks_fft)
        clinalg.multiply(mirror_mask_gpu, complex_slicer_gpu, overwrite=True)

        # Pupil Mirror -> Slits
        cu_fft.fft(complex_slicer_gpu, complex_slicer_gpu, plan)
        slits = complex_slicer_gpu.get()
        complex_slicer_gpu.gpudata.free()
        mirror_mask_gpu.gpudata.free()
        slit = fftshift(np.sum((np.abs(slits))**2, axis=0))

        free, total = cuda.mem_get_info()
        print("***Free: %.2f percent" % (free / total * 100))

        return slit

    def propagate_one_wavelength(self, wavelength, wavefront, plot=False):
        """
        Run the propagation from PUPIL Plane to EXIT SLIT Plane for a given wavelength
        :param wavelength:
        :param wavefront:
        """

        print("\nPropagating Wavelength: %.2f microns" % wavelength)
        complex_slicer = self.propagate_pupil_to_slicer(wavelength=wavelength, wavefront=wavefront)
        complex_mirror = self.propagate_slicer_to_pupil_mirror(complex_slicer)
        exit_slits, image_slit = self.propagate_pupil_mirror_to_exit_slit(complex_mirror, wavelength=wavelength)

        if plot:

            #___________________________________________________________________________________
            # Image Slicer plane
            slicer_size = self.N_PIX * self.spaxel_scale / 2
            slicer_extents = [-slicer_size, slicer_size, -slicer_size, slicer_size]
            zoom_size = self.N_slices * self.slice_size_mas / 2
            slicer_intensity = (np.abs(complex_slicer))**2

            plt.figure()
            plt.imshow(slicer_intensity, extent=slicer_extents)
            self.plot_slicer_boundaries()
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            plt.colorbar()
            plt.title('Slicer Plane')

            plt.figure()
            plt.imshow(np.log10(slicer_intensity), extent=slicer_extents)
            plt.clim(vmin=-10)
            self.plot_slicer_boundaries()
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            plt.colorbar()
            plt.title('Slicer Plane [log10]')

            #___________________________________________________________________________________
            # Pupil Mirror plane
            central_slice = (self.N_slices - 1)//2
            pupil_mirror = complex_mirror[central_slice]
            pupil_image = (np.abs(pupil_mirror))**2
            plt.figure()
            plt.imshow(pupil_image, extent=[-1, 1, -1, 1])
            plt.axhline(self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.axhline(-self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.title('Pupil Mirror [Central Slice]')

            plt.figure()
            plt.imshow(np.log10(pupil_image), extent=[-1, 1, -1, 1])
            plt.clim(vmin=-10)
            plt.axhline(self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.axhline(-self.pupil_mirror_aperture, linestyle='--', color='white')
            plt.title('Pupil Mirror [Central Slice]')

            #___________________________________________________________________________________
            # Exit Slit plane

            plt.figure()
            plt.imshow(image_slit / self.PEAKs[wavelength], extent=slicer_extents)
            self.plot_slicer_boundaries()
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            plt.colorbar()
            plt.title('Exit Slit')

            residual = (image_slit - self.nominal_PSFs[wavelength]) / self.PEAKs[wavelength]
            m_res = min(np.min(residual), -np.max(residual))
            plt.figure()
            plt.imshow(residual, extent=slicer_extents, cmap='bwr')
            plt.xlim([-zoom_size, zoom_size])
            plt.ylim([-zoom_size, zoom_size])
            self.plot_slicer_boundaries()
            plt.colorbar()
            plt.clim(m_res, -m_res)
            plt.title('Exit Slit - No Slicer')

        return complex_slicer, complex_mirror, image_slit, exit_slits

    def plot_slicer_boundaries(self):
        """
        Overlays the boundaries of each SLICE on the plots
        :return:
        """

        min_alpha, max_alpha = 0.15, 0.85
        half_slices = (self.N_slices - 1)//2
        alphas = np.linspace(min_alpha, max_alpha,  half_slices)
        alphas = np.concatenate([alphas, [max_alpha], alphas[::-1]])
        for y, alpha in zip(self.slice_boundaries, alphas):
            plt.axhline(y, linestyle='--', color='white', alpha=alpha)
        return


class SlicerPSFCalculator(object):

    def __init__(self, slicer_model, N_actuators, radial=True, h_centers=25):
        print("\nCreating PSF Calculator")
        self.slicer_model = slicer_model

        self.actuator_centres = self.create_actuator_centres(N_actuators, radial)
        self.N_act = len(self.actuator_centres[0][0])
        print("Total Number of Actuators: ", self.N_act)
        matrices = self.create_actuator_matrices(self.actuator_centres, h_centers)

        self.actuator_matrices = [x[0] for x in matrices]
        self.pupil_masks = [x[1] for x in matrices]
        self.actuator_flats = [x[2] for x in matrices]

        # Show the actuator positions for each Wavelength
        for i, wave_r in enumerate(self.slicer_model.waves_ratio):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            circ1 = Circle((0, 0), self.rho_aper / wave_r, linestyle='--', fill=None)
            circ2 = Circle((0, 0), self.rho_obsc / wave_r, linestyle='--', fill=None)
            ax.add_patch(circ1)
            ax.add_patch(circ2)
            for c in self.actuator_centres[i][0]:
                ax.scatter(c[0], c[1], color='red', s=10)
                ax.scatter(c[0], c[1], color='black', s=10)
            ax.set_aspect('equal')
            plt.xlim([-self.rho_aper, self.rho_aper])
            plt.ylim([-self.rho_aper, self.rho_aper])
            plt.title('Wavelength: %.2f microns' % self.slicer_model.wave_range[i])
        plt.show()
        return

    def create_actuator_centres(self, N_actuators, radial=True):
        """
        Computes the (Xc, Yc) coordinates of actuator centres
        inside a circle of rho_aper, assuming there are N_actuators
        along the line Diameter

        :param N_actuators: Number of actuators along the Diameter
        :param radial: if True, we add actuators at the boundaries RHO_APER, RHO_OBSC
        :return: [act (list of actuator centres), delta (actuator separation)]
        """
        wave_range = self.slicer_model.wave_range
        wave_ratios = self.slicer_model.waves_ratio
        self.rho_aper = rho_spaxel_scale(self.slicer_model.spaxel_scale, wavelength=wave_range[0])
        self.rho_obsc = CENTRAL_OBS * self.rho_aper

        centres = []
        for wave in wave_ratios:
            x0 = np.linspace(-self.rho_aper / wave, self.rho_aper / wave, N_actuators, endpoint=True)
            delta = x0[1] - x0[0]
            xx, yy = np.meshgrid(x0, x0)
            x_f = xx.flatten()
            y_f = yy.flatten()

            act = []  # List of actuator centres (Xc, Yc)
            for x_c, y_c in zip(x_f, y_f):
                r = np.sqrt(x_c ** 2 + y_c ** 2)
                # Leave some margin close to the boundary
                if r < (self.rho_aper / wave - delta / 2) and r > (self.rho_obsc / wave + delta / 2):
                    act.append([x_c, y_c])

            if radial:  # Add actuators at the boundaries, keeping a constant angular distance
                for r in [self.rho_aper / wave, self.rho_obsc / wave]:
                    N_radial = int(np.floor(2 * np.pi * r / delta))
                    d_theta = 2 * np.pi / N_radial
                    theta = np.linspace(0, 2 * np.pi - d_theta, N_radial)
                    # Super important to do 2Pi - d_theta to avoid placing 2 actuators in the same spot... Degeneracy
                    for t in theta:
                        act.append([r * np.cos(t), r * np.sin(t)])

            centres.append([act, delta])
        return centres

    def create_actuator_matrices(self, centres, h_centers):
        """
        Creates the a list of matrices at a range of Wavelengths useful for Wavefront calculations
        [[Actuator_Matrices, Pupil_Masks, Actuator_Flats]_{wave_1}, ..., [ ]_{wave_N} ]

        (1) Actuator Matrices is a [N_PIX, N_PIX, N_act] array representing the influence matrix of each actuator
        i.e., each slice shows the effect of poking one actuator
        (2) Pupil_Masks: the pupil mask at each wavelength (which varies in relative size)
        (3) Actuator Flats: flattened version of Actuator Matrices that only accounts for point inside the pupil
        this is very useful for RMS wavefront calculations

        :param centres: list of actuator centres from self.create_actuator_centres()
        :param h_centers: height of the Gaussian influence function at the neighbour actuator
        :return:
        """

        wave_ratios = self.slicer_model.waves_ratio
        alpha = 1 / np.sqrt(np.log(100 / h_centers))

        matrices = []
        for i, wave in enumerate(wave_ratios):

            cent, delta = centres[i]
            N_act = len(cent)
            matrix = np.empty((N_PIX, N_PIX, N_act))
            x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
            xx, yy = np.meshgrid(x0, x0)
            rho = np.sqrt(xx ** 2 + yy ** 2)
            pupil = (rho <= self.rho_aper / wave) & (rho >= self.rho_obsc / wave)

            for k in range(N_act):
                xc, yc = cent[k][0], cent[k][1]
                r2 = (xx - xc) ** 2 + (yy - yc) ** 2
                matrix[:, :, k] = pupil * np.exp(-r2 / (alpha * delta) ** 2)

            mat_flat = matrix[pupil]
            matrices.append([matrix, pupil, mat_flat])

        return matrices

    def generate_PSF(self, coef):
        """
        Using the SLICER MODEL, it propagates a set of wavefronts up to the EXIT SLIT
        to model the effects of Image Slicers on the PSF
        """

        print("\nGenerating %d PSF images" % N_PSF)

        PSF_slicer, PSF_no_slicer = [], []
        for i in range(N_PSF):
            print("     PSF #%d" % (i+1))
            PSF_slicer_waves, PSF_no_slicer_waves = [], []     # length is Number of Wavelengths

            for j, wave in enumerate(self.slicer_model.wave_range):
                # With Slicer
                pupil = self.pupil_masks[j]
                wavefront = pupil * np.dot(self.actuator_matrices[j], coef[i])
                _slicer, _mirror, slit, slits = self.slicer_model.propagate_one_wavelength(wave, wavefront, plot=False)
                crop_slit = crop_array(slit, self.slicer_model.N_slices * self.slicer_model.spaxels_per_slice)
                PSF_slicer_waves.append(crop_slit / self.slicer_model.PEAKs[wave])

                # Without Slicer
                pupil_complex = pupil * np.exp(1j * 2 * np.pi * wavefront / wave)
                image_ = (np.abs(fftshift(fft2(pupil_complex, norm='ortho'))))**2
                image = crop_array(image_, self.slicer_model.N_slices * self.slicer_model.spaxels_per_slice)
                PSF_no_slicer_waves.append(image / self.slicer_model.PEAKs[wave])

            PSF_slicer.append(PSF_slicer_waves)
            PSF_no_slicer.append(PSF_no_slicer_waves)
            print("_______________________________________________________")

        return np.array(PSF_slicer), np.array(PSF_no_slicer)


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Sanity check on the sizes of the apertures to match a 4 m.a.s. spaxel scale at 1.5 um
    nom_wave = 1.5          # microns
    spaxels = 4             # m.a.s
    _rho = rho_spaxel_scale(spaxel_scale=spaxels, wavelength=nom_wave)
    check_spaxel_scale(rho_aper=_rho, wavelength=nom_wave)

    # ================================================================================================================ #
    #                           Example of how to create an Image Slicer model                                         #
    # ================================================================================================================ #

    # for wave in slicer.wave_range:
    #     plt.figure()
    #     plt.imshow(slicer.pupil_masks[wave])
    #     plt.title('Pupil Mask at %.2f microns' % wave)
    # plt.show()

    # ================================================================================================================ #
    #                                      Speed Comparison on the GPU                                            #
    # ================================================================================================================ #

    slicer_options = {"N_slices": 38, "spaxels_per_slice": 11,
                      "pupil_mirror_aperture": 0.85, "anamorphic": True}
    N_PIX = 2048
    spaxel_mas = 0.25        # to get a decent resolution

    slicer = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX,
                         spaxel_scale=spaxel_mas, N_waves=2, wave0=1.5, waveN=2.5)

    print("\n---------------------------------------------------------------------")
    print("Running on the CPU")
    N_PSF = 10
    cpu_start = time()
    for i in range(N_PSF):
        complex_slicer, complex_mirror, exit_slit, slits = slicer.propagate_one_wavelength(wavelength=1.5, wavefront=0, plot=False)
    cpu_end = time()
    cpu_time = cpu_end - cpu_start

    print("Time to propagate %d PSFs: %.2f seconds" % (N_PSF, cpu_time))
    print("Time to propagate 1 PSF (%d slices): %.2f seconds" % (slicer.N_slices, cpu_time / N_PSF))
    print("Time to propagate 1 slice: %.2f seconds" % (cpu_time / N_PSF / slicer.N_slices))

    N_PSF = 10
    gpu_start = time()
    exit_slits = slicer.propagate_gpu_wavelength(wavelength=1.5, wavefront=0, N=N_PSF)
    gpu_end = time()
    gpu_time = gpu_end - gpu_start
    print("Time to propagate %d PSFs: %.2f seconds" % (N_PSF, gpu_time))
    print("Time to propagate 1 PSF (%d slices): %.2f seconds" % (slicer.N_slices, gpu_time / N_PSF))
    print("Time to propagate 1 slice: %.2f seconds" % (gpu_time / N_PSF / slicer.N_slices))

    free, total = cuda.mem_get_info()
    print("Free: %.2f percent" % (free / total * 100))

    for i in range(N_PSF):
        plt.figure()
        plt.imshow(exit_slits[i])
        plt.colorbar()
    plt.show()

    # ================================================================================================================ #
    #                                    HARMONI comparison                                                            #
    # ================================================================================================================ #

    # First we have to match the Slice Size to have a proper clipping at the Slicer Plane
    # That comes from the product spaxels_per_slice * spaxel_mas

    N_slices = 37
    spaxels_per_slice = 28
    spaxel_mas = 0.25  # to get a decent resolution

    # HARMONI fits approximately 6 rings (at each side) at the Pupil Mirror at 1.5 microns
    # that would
    # In Python we have 1/2 spaxels_per_slice rings at each side in the Pupil Mirror arrays
    N_rings = spaxels_per_slice / 2
    rings_we_want = 2
    pupil_mirror_aperture = rings_we_want / N_rings

    N_PIX = 2048
    wave0, waveN = 1.5, 3.0

    slicer_options = {"N_slices": N_slices, "spaxels_per_slice": spaxels_per_slice,
                      "pupil_mirror_aperture": pupil_mirror_aperture, "anamorphic": True}

    HARMONI = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX,
                         spaxel_scale=spaxel_mas, N_waves=2, wave0=wave0, waveN=waveN)

    for wave in HARMONI.wave_range:

        complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=wave, wavefront=0)

        masked_slicer = (np.abs(complex_slicer)) ** 2 * HARMONI.slicer_masks[N_slices // 2]
        masked_slicer /= np.max(masked_slicer)
        minPix_Y = (N_PIX + 1 - 2 * spaxels_per_slice) // 2         # Show 2 slices
        maxPix_Y = (N_PIX + 1 + 2 * spaxels_per_slice) // 2
        minPix_X = (N_PIX + 1 - 6 * spaxels_per_slice) // 2
        maxPix_X = (N_PIX + 1 + 6 * spaxels_per_slice) // 2
        masked_slicer = masked_slicer[minPix_Y:maxPix_Y, minPix_X:maxPix_X]

        plt.figure()
        plt.imshow(masked_slicer, cmap='jet')
        plt.colorbar(orientation='horizontal')
        plt.title('HARMONI Slicer: Central Slice @%.2f microns' % wave)
        # plt.show()

        masked_pupil_mirror = (np.abs(complex_mirror[N_slices // 2])) ** 2 * HARMONI.pupil_mirror_mask[wave]
        masked_pupil_mirror /= np.max(masked_pupil_mirror)
        plt.figure()
        plt.imshow(np.log10(masked_pupil_mirror), cmap='jet')
        plt.colorbar()
        plt.clim(vmin=-4)
        plt.title('Pupil Mirror: Aperture %.2f PSF zeros' % rings_we_want)
        # plt.show()

        masked_slit = exit_slit * HARMONI.slicer_masks[N_slices // 2]
        masked_slit = masked_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
        plt.figure()
        plt.imshow(masked_slit, cmap='jet')
        plt.colorbar(orientation='horizontal')
        plt.title('Exit Slit @%.2f microns (Pupil Mirror: %.2f PSF zeros)' % (wave, rings_we_want))
    plt.show()

    # ================================================================================================================ #

    # Compare everything to the Zemax POP PSFs
    cwd = os.getcwd()
    path_zemax = os.path.join(cwd, 'ImageSlicerEffects\\HARMONI\\PupilMirror\\2048\\2Rings\\3 um')

    list_slices = list(np.arange(1, 76, 2))
    central_slice = 19
    pop_slicer = POP_Slicer()
    POP_PSF, POP_slices = pop_slicer.read_all_zemax_files(path_zemax, 'HARMONI_SLICER_EFFECTS 0_', list_slices)
    POP_PSF /= np.max(POP_PSF)

    # HARMONI sampling
    L_harmoni = 4.94                            # Physical size of POP arrays
    harmoni_sampling = L_harmoni / 2048         # mm / pixel
    harmoni_extent = [-L_harmoni/2, L_harmoni/2, -L_harmoni/2, L_harmoni/2]

    physical_slice = 0.13               # mm at Exit Slit
    python_sampling = physical_slice / HARMONI.spaxels_per_slice    # mm / pixel
    new_spaxels_per_slice = physical_slice / harmoni_sampling
    L_python = python_sampling * HARMONI.N_PIX
    python_extent = [-L_python/2, L_python/2, -L_python/2, L_python/2]

    complex_slicer, complex_mirror, exit_slit, slits = HARMONI.propagate_one_wavelength(wavelength=1.5, wavefront=0)
    python_PSF = exit_slit
    python_PSF /= np.max(python_PSF)

    zoom = 0.75
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(POP_PSF, extent=harmoni_extent, cmap='jet')
    ax1.set_xlim(-zoom, zoom)
    ax1.set_ylim(-zoom, zoom)
    ax1.set_title(r'Zemax POP')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    plt.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(python_PSF, extent=python_extent, cmap='jet')
    ax2.set_xlim(-zoom, zoom)
    ax2.set_ylim(-zoom, zoom)
    ax2.set_title('Python')
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    plt.colorbar(im2, ax=ax2, orientation='horizontal')
    plt.show()

    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(np.log10(POP_PSF), extent=harmoni_extent)
    im1.set_clim(vmin=-10)
    ax1.set_title(r'Zemax POP')
    plt.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(np.log10(python_PSF), extent=python_extent)
    ax2.set_xlim(-0.5*L_harmoni, 0.5*L_harmoni)
    ax2.set_ylim(-0.5*L_harmoni, 0.5*L_harmoni)
    im2.set_clim(vmin=-10)
    ax2.set_title('Python')
    plt.colorbar(im2, ax=ax2, orientation='horizontal')
    plt.show()

    # Central Slice
    # masked_slit = exit_slit * HARMONI.slicer_masks[N_slices // 2]
    masked_slit = slits[N_slices // 2]
    # masked_slit = masked_slit[minPix_Y: maxPix_Y, minPix_X: maxPix_X]
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(POP_slices[central_slice//2], extent=harmoni_extent, cmap='jet')
    ax1.set_xlim([-physical_slice, physical_slice])
    ax1.set_ylim([-physical_slice, physical_slice])
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    # im1.set_clim(vmin=-10)
    ax1.set_title(r'Zemax POP | Central Slice')
    plt.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(masked_slit, extent=python_extent,cmap='jet')
    # im2.set_clim(vmin=-10)
    ax2.set_xlim([-physical_slice, physical_slice])
    ax2.set_ylim([-physical_slice, physical_slice])
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_title('Python | Central Slice')
    plt.colorbar(im2, ax=ax2, orientation='horizontal')
    plt.show()

    zoom = 3
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(np.log10(POP_slices[central_slice//2]), extent=harmoni_extent, cmap='jet')
    ax1.set_xlim([-zoom*physical_slice, zoom*physical_slice])
    ax1.set_ylim([-zoom*physical_slice, zoom*physical_slice])
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    im1.set_clim(vmin=-10)
    ax1.set_title(r'Zemax POP | Central Slice')
    plt.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(np.log10(masked_slit), extent=python_extent,cmap='jet')
    im2.set_clim(vmin=-6)
    ax2.set_xlim([-zoom*physical_slice, zoom*physical_slice])
    ax2.set_ylim([-zoom*physical_slice, zoom*physical_slice])
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_title('Python | Central Slice')
    plt.colorbar(im2, ax=ax2, orientation='horizontal')
    plt.show()

    #TODO: Ask about Exit Slit masks

    # ================================================================================================================ #
    #                                    Impact of the PUPIL MIRROR APERTURE                                           #
    # ================================================================================================================ #

    print("\n======================================================")
    print("         Impact of Pupil Mirror Aperture ")
    print("======================================================\n")

    apertures = [0.99, 0.95, 0.90, 0.80, 0.75]

    for aper in apertures:
        print("\n-------------------------------------------------------------")
        print("         Pupil Mirror Aperture: ", aper)
        print("-------------------------------------------------------------\n")
        slicer_options = {"N_slices": 15, "spaxels_per_slice": 11, "pupil_mirror_aperture": aper}
        slicer = SlicerModel(slicer_options=slicer_options,N_PIX=N_PIX,
                             spaxel_scale=0.5, N_waves=2, wave0=1.5, waveN=2.0)
        slicer_size = slicer.N_PIX * slicer.spaxel_scale / 2
        slicer_extents = [-slicer_size, slicer_size, -slicer_size, slicer_size]
        zoom_size = slicer.N_slices * slicer.slice_size_mas / 2

        for wave in slicer.wave_range:
            _complex_slicer, _complex_mirror, image_slit, slits = slicer.propagate_one_wavelength(wavelength=wave, wavefront=0)

            # Pupil Mirror plane
            central_slice = (slicer.N_slices - 1) // 2
            pupil_mirror = _complex_mirror[central_slice]
            pupil_image = (np.abs(pupil_mirror)) ** 2

            # Exit Slit
            residual = (image_slit - slicer.nominal_PSFs[wave]) / slicer.PEAKs[wave]
            m_res = min(np.min(residual), -np.max(residual))

            plt.figure()
            ax1 = plt.subplot(1, 2, 1)
            im1 = ax1.imshow(np.log10(pupil_image), extent=[-1, 1, -1, 1])
            im1.set_clim(vmin=-10)
            ax1.axhline(slicer.pupil_mirror_aperture, linestyle='--', color='white')
            ax1.axhline(-slicer.pupil_mirror_aperture, linestyle='--', color='white')
            ax1.set_title('Pupil Mirror [Central Slice] (%.2f microns) | Aperture: %.2f' % (wave, aper))
            plt.colorbar(im1, ax=ax1, orientation='horizontal')

            ax2 = plt.subplot(1, 2, 2)
            im2 = ax2.imshow(residual, extent=slicer_extents, cmap='bwr')
            im2.set_clim(m_res, -m_res)
            ax2.set_xlim([-zoom_size, zoom_size])
            ax2.set_ylim([-zoom_size, zoom_size])
            ax2.set_xlabel(r'm.a.s')
            ax2.set_ylabel(r'm.a.s')
            slicer.plot_slicer_boundaries()
            plt.colorbar(im2, ax=ax2, orientation='horizontal')
            ax2.set_title('Exit Slit - No Slicer (%.2f microns) | Aperture: %.2f' % (wave, aper))

    plt.show()

    # ================================================================================================================ #
    #                                            Impact of the SLICE size                                              #
    # ================================================================================================================ #

    print("\n======================================================")
    print("         Impact of Slice size     ")
    print("======================================================\n")

    slice_sizes = [7, 11, 15, 19]           # in spaxels of 0.5 mas each
    scale_mas = 0.5
    fwhm_ratios = []
    pup_mirr_aper = 0.90

    for slice_size in slice_sizes:

        slicer_options = {"N_slices": 15, "spaxels_per_slice": slice_size, "pupil_mirror_aperture": pup_mirr_aper}
        slicer = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX,
                             spaxel_scale=scale_mas, N_waves=2, wave0=1.5, waveN=2.0)
        FWHM_ratio = slicer.FWHM_ratio
        fwhm_ratios.append(FWHM_ratio)
        print("\n-----------------------------------------------------------------------------------------")
        print("Slice Size: %d spaxels | %.2f of the FWHM @1.5 microns" % (slice_size, FWHM_ratio))
        print("-----------------------------------------------------------------------------------------\n")
        slicer_size = slicer.N_PIX * slicer.spaxel_scale / 2
        slicer_extents = [-slicer_size, slicer_size, -slicer_size, slicer_size]
        zoom_size = slicer.N_slices * slicer.slice_size_mas / 2


        for wave in slicer.wave_range:
            _complex_slicer, _complex_mirror, image_slit, slits = slicer.propagate_one_wavelength(wavelength=wave, wavefront=0)

            central_slice = (slicer.N_slices - 1) // 2

            # Image Slicer
            slice_image = (np.abs(_complex_slicer)) ** 2

            # Pupil Mirror plane
            pupil_mirror = _complex_mirror[central_slice]
            pupil_image = (np.abs(pupil_mirror)) ** 2

            # Exit Slit
            residual = (image_slit - slicer.nominal_PSFs[wave]) / slicer.PEAKs[wave]
            m_res = min(np.min(residual), -np.max(residual))

            plt.figure()
            ax1 = plt.subplot(1, 3, 1)
            im1 = ax1.imshow(np.log10(slice_image), extent=slicer_extents)
            im1.set_clim(vmin=-10)
            ax1.set_xlim([-zoom_size, zoom_size])
            ax1.set_ylim([-zoom_size, zoom_size])
            slicer.plot_slicer_boundaries()
            ax1.set_title('Slicer (%.2f microns) | Slice Width: %.2f FWHM' % (wave, FWHM_ratio))
            plt.colorbar(im1, ax=ax1, orientation='horizontal')

            ax2 = plt.subplot(1, 3, 2)
            im2 = ax2.imshow(np.log10(pupil_image), extent=[-1, 1, -1, 1])
            im2.set_clim(vmin=-10)
            ax2.axhline(slicer.pupil_mirror_aperture, linestyle='--', color='white')
            ax2.axhline(-slicer.pupil_mirror_aperture, linestyle='--', color='white')
            ax2.set_title('Pupil Mirror [Central Slice] (%.2f microns)' % (wave))
            plt.colorbar(im2, ax=ax2, orientation='horizontal')

            ax3 = plt.subplot(1, 3, 3)
            im3 = ax3.imshow(residual, extent=slicer_extents, cmap='bwr')
            im3.set_clim(m_res, -m_res)
            ax3.set_xlim([-zoom_size, zoom_size])
            ax3.set_ylim([-zoom_size, zoom_size])
            ax3.set_xlabel(r'm.a.s')
            ax3.set_ylabel(r'm.a.s')
            slicer.plot_slicer_boundaries()
            plt.colorbar(im3, ax=ax3, orientation='horizontal')
            ax3.set_title('Exit Slit - No Slicer (%.2f microns)' % (wave))

    plt.show()

    # ================================================================================================================ #
    #              Example of how to generate PSF images with Slicer Effects                                           #
    # ================================================================================================================ #

    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+-+            How to generate PSF images with Slicer Effects")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    """ (1) Create a Slicer Model """
    pup_mirr_aper = 0.70
    scale_mas = 0.5
    slicer_options = {"N_slices": 15, "spaxels_per_slice": 7, "pupil_mirror_aperture": pup_mirr_aper}
    slicer = SlicerModel(slicer_options=slicer_options, N_PIX=N_PIX,
                         spaxel_scale=scale_mas, N_waves=2, wave0=1.5, waveN=2.0)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    """ (2) Create a PSF Calculator based on the Slicer Model """
    # Number of Actuators in the Diameter, height of Gaussian profile at neighbour actuator [%]
    N_act, h_centres = 8, 20
    PSF_generator = SlicerPSFCalculator(slicer_model=slicer, N_actuators=N_act, radial=True, h_centers=h_centres)
    N_PSF = 1
    scale = 0.35        # Scale of the actuator commands
    coef = scale * np.random.uniform(low=-1, high=1, size=(N_PSF, PSF_generator.N_act))
    images_slicer, images_no_slicer = PSF_generator.generate_PSF(coef)

    slicer_size = slicer.N_PIX * slicer.spaxel_scale / 2
    slicer_extents = [-slicer_size, slicer_size, -slicer_size, slicer_size]

    j_image = 0
    for i, wave in enumerate(slicer.wave_range):


        PSF_slicer = images_slicer[j_image, i]
        PSF_no_slicer = images_no_slicer[j_image, i]
        diff = PSF_slicer - PSF_no_slicer
        m_diff = min(np.min(diff), -np.max(diff))

        plt.figure()
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(PSF_slicer)
        # im1.set_clim(vmin=-10)
        # ax1.set_xlim([-zoom_size, zoom_size])
        # ax1.set_ylim([-zoom_size, zoom_size])
        # slicer.plot_slicer_boundaries()
        # ax1.set_title('Slicer (%.2f microns) | Slice Width: %.2f FWHM' % (wave, FWHM_ratio))
        plt.colorbar(im1, ax=ax1, orientation='horizontal')

        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(PSF_no_slicer)
        # im2.set_clim(vmin=-10)
        # ax2.set_xlim([-zoom_size, zoom_size])
        # ax2.set_ylim([-zoom_size, zoom_size])
        # slicer.plot_slicer_boundaries()
        # ax2.set_title('Pupil Mirror [Central Slice] (%.2f microns)' % (wave))
        plt.colorbar(im2, ax=ax2, orientation='horizontal')

        ax3 = plt.subplot(1, 3, 3)
        im3 = ax3.imshow(diff, extent=slicer_extents, cmap='bwr')
        # im3.set_clim(m_diff, -m_diff)
        # ax3.set_xlim([-zoom_size, zoom_size])
        # ax3.set_ylim([-zoom_size, zoom_size])
        # ax3.set_xlabel(r'm.a.s')
        # ax3.set_ylabel(r'm.a.s')
        # slicer.plot_slicer_boundaries()
        plt.colorbar(im3, ax=ax3, orientation='horizontal')
        # ax3.set_title('Exit Slit - No Slicer (%.2f microns)' % (wave))

    plt.show()

    #___________________________________________________________________________________





