"""
Author: Alvaro Menduina
Sanity check on the effect of Defocus intensity on the Peak and Rings of the PSF

It shows that a defocus of PV = 1 wave suppresses the central Peak
"""

import numpy as np
import matplotlib.pyplot as plt
import zern_core as zern
from numpy.fft import fft, fft2, ifft2, fftshift

pix = 30                                # Pixels to crop the PSF
N_PIX = 2048                            # Pixels for the Fourier arrays
wave = 1.5                              # microns
ELT_DIAM = 39                           # meters
MILIARCSECS_IN_A_RAD = 206265000

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
    FWHM_RAD = wavelength * 1e-6 / ELT_DIAM  # radians
    FWHM_MAS = FWHM_RAD * MILIARCSECS_IN_A_RAD
    return FWHM_MAS


def crop_array(array, crop=25):
    PIX = array.shape[0]
    min_crop = PIX // 2 - crop // 2
    max_crop = PIX // 2 + crop // 2
    array_crop = array[min_crop:max_crop, min_crop:max_crop]
    return array_crop


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    spaxel_scale = 0.5      # mas
    RHO_APER = rho_spaxel_scale(spaxel_scale, wavelength=wave)
    check_spaxel_scale(RHO_APER, wave)
    FWHM = compute_FWHM(wave)       # mas
    crop_pix = 5 * FWHM / spaxel_scale      # How many pixels to cover enough FWHMs
    crop_pix = int(np.ceil(crop_pix))

    """ Impact of Defocus on a Round PSF - Evolution of Rings """
    # How does the defocus intensity affect the Peak and Rings of the PSF?

    x = np.linspace(-1, 1, N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho, theta = np.sqrt(xx**2 + (2*yy)**2), np.arctan2(xx, yy)
    pupil = rho <= RHO_APER
    rho, theta = rho[pupil], theta[pupil]
    zernike = zern.ZernikeNaive(mask=pupil)
    _phase = zernike(coef=np.zeros(10), rho=rho/RHO_APER, theta=theta, normalize_noll=False, mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix[:, 3:]   # remove the piston and tilts
    H_matrix = zern.invert_model_matrix(H_flat, pupil)

    # Rescale by 1/2 to have a Peak-To-Valley of 1 wave
    defocus = 0.5 * H_matrix[:, :, 1].copy()

    # Compute nominal PSF -> Pupil = Aperture * exp(2 PI wavefront)
    pupil_nom = pupil * np.exp(2*np.pi * 1j * np.zeros_like(pupil))
    image_nom = (np.abs(fftshift(fft2(pupil_nom))))**2
    PEAK = np.max(image_nom)
    image_nom /= PEAK

    focus = [0, 0.25, 0.5, 1.0]
    img = []
    mas = spaxel_scale * np.linspace(-crop_pix//2, crop_pix//2, crop_pix)
    plt.figure()
    for eps in focus:

        pupil_foc = pupil * np.exp(2*np.pi * 1j * eps * defocus)
        image_foc = (np.abs(fftshift(fft2(pupil_foc))))**2
        image_foc /= PEAK
        cropped = crop_array(image_foc, crop=crop_pix)
        img.append(cropped)

        psf_2d = cropped[crop_pix//2, :]

        plt.plot(mas, (psf_2d), label=eps)
    plt.xlabel('mas')
    plt.legend(title='Defocus Strength $f$')
    plt.title(r'PSF($\exp{(2\pi f Z_f)}$)')
    plt.show()

    # Having a defocus of PV=1 effectively suppresses the central peak

    cm = 'viridis'
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(img[0], cmap=cm)
    ax1.set_title(r'0 $\lambda$')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    # plt.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(img[1], cmap=cm)
    ax2.set_title(r'$\lambda / 4$')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    # plt.colorbar(im2, ax=ax2, orientation='horizontal')

    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(img[2], cmap=cm)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'$\lambda / 2$')

    ax4 = plt.subplot(2, 2, 4)
    im4 = ax4.imshow(img[3], cmap=cm)
    ax4.set_title(r'$\lambda$')
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    plt.show()


    #### OLD stuff

    # ================================================================================================================ #
    """ If the Slice Width is 1.0 FWHM """
    # slice_width = 11     # 11 pixels is the FWHM
    def slice_mask(array, width, i_slice=0, crop=250):
        PIX = array.shape[0]
        min_crop = PIX // 2 - crop // 2
        max_crop = PIX // 2 + crop // 2
        central_slice = array.shape[0]//2
        slice_centre = central_slice + i_slice * width

        sliced = array[slice_centre - width//2 : slice_centre + width//2+1, min_crop:max_crop]

        mask_slice = np.zeros((PIX, PIX)).astype(bool)
        mask_slice[slice_centre - width//2 : slice_centre + width//2+1, :] = True
        plt.imshow(mask_slice * array)

        # plt.figure()
        # plt.imshow(array[min_crop:max_crop, min_crop:max_crop], extent=[- crop // 2,crop // 2, - crop // 2, crop // 2])
        # plt.axhline(width/2, color='white', linestyle='--')
        # plt.axhline(-width/2, color='white', linestyle='--')
        # ratio = width / 22
        # plt.title(r'Slice Width = %d pix [FWHM_Y = %d pix]' %(width, 22))
        return sliced, mask_slice

    central, mask_slice = slice_mask(image_nom, width=101, i_slice=0, crop=50)
    plt.show()
    plt.figure()
    plt.imshow(central)
    plt.show()

    total = np.sum(crop_array(image_nom, crop=250))

    # focus = [0, 0.25, 0.5, 1.0]
    focus = np.linspace(0, 1, 15, endpoint=True)
    p_central = []
    p_upper = []
    p_upper2 = []
    p_upper3 = []
    p_tot = []
    slice_width = 11
    for eps in focus:
        print(eps)

        pupil_foc = pupil * np.exp(2*np.pi * 1j * eps * defocus)
        image_foc = (np.abs(fftshift(fft2(pupil_foc))))**2
        image_foc /= PEAK
        # total_foc = np.sum(crop_array(image_foc, crop=150))

        central = slice_mask(image_foc, slice_width, i_slice=0)
        p_central.append(np.sum(central) / total)
        upper = slice_mask(image_foc, slice_width, i_slice=1)
        p_upper.append(np.sum(upper) / total)
        upper2 = slice_mask(image_foc, slice_width, i_slice=2)
        p_upper2.append(np.sum(upper2) / total)
        upper3 = slice_mask(image_foc, slice_width, i_slice=3)
        p_upper3.append(np.sum(upper3) / total)
        s = np.sum(central + 2*upper + 2*upper2 + 2*upper3) / total
        p_tot.append(s)

    plt.figure()
    plt.plot(focus, p_central, label='Central')
    plt.plot(focus, p_upper, label='+1')
    plt.plot(focus, p_upper2, label='+2')
    plt.plot(focus, p_upper3, label='+3')
    # plt.plot(focus, p_tot, label='Total')
    plt.ylim([0, 0.5])
    plt.xlim([0, 1])
    plt.legend(title='Slice')
    plt.xlabel('PV Defocus [$\lambda$]')
    plt.ylabel('Relative power')
    plt.grid(True)
    plt.show()

    """ Fourier stuff """
    """ (1) Pupil Plane """
    pupil_f = pupil * np.exp(2*np.pi * 1j * np.zeros_like(pupil))

    """ (2) Slicer Focal Plane """
    slicer_plane = fftshift(fft2(pupil_f))
    # Mask one Slice
    masked_slicer = mask_slice * slicer_plane
    slicer_image = (np.abs(masked_slicer))**2 / PEAK

    """ (3) Pupil Plane"""
    pupil_mirror = ifft2(fftshift(masked_slicer))
    pupil_image = (np.abs(pupil_mirror))**2
    # pup_pix = int(3*N_PIX /4)
    pup_pix = int(N_PIX /4)

    # Pupil Mirror Aperture
    mask_pupil = np.zeros((N_PIX, N_PIX)).astype(bool)
    mask_pupil[(N_PIX-pup_pix)//2 : (N_PIX+pup_pix)//2, :] = True
    # masked_pupil_image = mask_pupil * pupil_image
    masked_pupil_image = mask_pupil * pupil_image

    # Real and Imaginary part of Pupil Mirror E_field
    plt.figure()
    plt.plot(np.real(pupil_mirror)[:, N_PIX//2])
    # plt.axvline((N_PIX-pup_pix)//2, color='black', linestyle='--')
    # plt.axvline((N_PIX+pup_pix)//2, color='black', linestyle='--')
    plt.title('Electric Field at Pupil Mirror')
    # plt.plot(np.real(mask_pupil * pupil_mirror)[:, N_PIX//2])
    plt.figure()
    fp = fftshift(fft((mask_pupil*pupil_mirror)[:, N_PIX//2]))
    image_fp = ((np.abs(fp))**2)[N_PIX//2 - 100: N_PIX//2 + 100]

    plt.plot(image_fp)
    plt.title('Exit Slit Image Across the Slice')
    # plt.plot(np.imag(pupil_mirror)[:, N_PIX//2])
    plt.show()

    # TOT = image_fp.sum()
    TOT_34 = image_fp.sum() / TOT
    TOT_12 = image_fp.sum() / TOT
    TOT_14 = image_fp.sum() / TOT

    """ (4) Slit Exit """
    masked_pupil_mirror = mask_pupil * pupil_mirror
    slit_plane = fftshift(fft2(masked_pupil_mirror))
    slit_image = (np.abs(slit_plane))**2

    cm = 'hot'
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4)
    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(crop_array(pupil, 500), cmap=cm)
    ax1.set_title(r'Pupil Plane')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    # plt.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(crop_array(np.log10(slicer_image), 150), cmap=cm)
    ax2.set_title(r'Slicer Plane')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    # plt.colorbar(im2, ax=ax2, orientation='horizontal')

    ax3 = plt.subplot(1, 4, 3)
    # im3 = ax3.imshow(crop_array(np.log10(masked_pupil_image), 2048), vmin=-7, cmap=cm)
    im3 = ax3.imshow(np.log10(masked_pupil_image), vmin=-7, cmap=cm)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title(r'Pupil Mirror')

    ax4 = plt.subplot(1, 4, 4)
    im4 = ax4.imshow(crop_array((slit_image), 50), cmap='viridis')
    ax4.set_title(r'Exit Slit')
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    plt.show()

    # ================================================================================================================ #

    """ Power distribution across slices - Anamorphic PSF """
    x = np.linspace(-1, 1, N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho, theta = np.sqrt((xx/2)**2 + yy**2), np.arctan2(xx, yy)
    pupil = rho <= RHO_APER
    rho, theta = rho[pupil], theta[pupil]
    zernike = zern.ZernikeNaive(mask=pupil)
    _phase = zernike(coef=np.zeros(10), rho=rho/RHO_APER, theta=theta, normalize_noll=False, mode='Jacobi', print_option='Silent')
    H_flat = zernike.model_matrix[:,3:]   # remove the piston and tilts
    H_matrix = zern.invert_model_matrix(H_flat, pupil)

    plt.figure()
    plt.imshow(H_matrix[:,:,1])
    plt.colorbar()
    plt.show()

    pupil_f = pupil * np.exp(2*np.pi * 1j * np.zeros_like(pupil))
    image = (np.abs(fftshift(fft2(pupil_f))))**2
    PEAK = np.max(image)
    image /= PEAK
    total = np.sum(crop_array(image, crop=150))

    plt.figure()
    plt.imshow(crop_array(image, crop=50))
    plt.colorbar()

    plt.figure()
    plt.imshow(crop_array(np.log10(image), crop=150))
    plt.colorbar()
    plt.show()

    slice_width = 5     # pixels
    def slice_mask(array, width, i_slice=0, crop=150):
        PIX = array.shape[0]
        min_crop = PIX // 2 - crop // 2
        max_crop = PIX // 2 + crop // 2
        central_slice = array.shape[0]//2
        slice_centre = central_slice + i_slice * width

        sliced = array[slice_centre - width//2 : slice_centre + width//2+1, min_crop:max_crop]
        return sliced

    central = slice_mask(image, slice_width, i_slice=0)
    central_pow = np.sum(central)/total
    upper = slice_mask(image, slice_width, i_slice=1)

    plt.figure()
    plt.imshow(slice_mask(image, slice_width, i_slice=1))
    plt.figure()
    plt.imshow(slice_mask(image, slice_width, i_slice=0))
    plt.show()

    ### Defocused
    focus = np.linspace(0., 0.75, 25)
    defocus = H_matrix[:, :, 1].copy()
    p_central = []
    p_upper = []
    p_upper2 = []
    p_upper3 = []
    for eps in focus:
        # eps = 0.15
        print(eps)

        pupil_foc = pupil * np.exp(2*np.pi * 1j * eps * defocus)
        image_foc = (np.abs(fftshift(fft2(pupil_foc))))**2
        image_foc /= PEAK
        total_foc = np.sum(crop_array(image_foc, crop=150))

        central = slice_mask(image_foc, slice_width, i_slice=0)
        p_central.append(np.sum(central) / total_foc)
        upper = slice_mask(image_foc, slice_width, i_slice=1)
        p_upper.append(np.sum(upper) / total_foc)
        upper2 = slice_mask(image_foc, slice_width, i_slice=2)
        p_upper2.append(np.sum(upper2) / total_foc)
        upper3 = slice_mask(image_foc, slice_width, i_slice=3)
        p_upper3.append(np.sum(upper3) / total_foc)

    plt.figure()
    plt.plot(focus, p_central)
    plt.plot(focus, p_upper)
    plt.plot(focus, p_upper2)
    plt.plot(focus, p_upper3)
    plt.ylim([0, 0.5])
    plt.show()





