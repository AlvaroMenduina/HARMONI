"""
Attempt to simulate semi-realistic Image Slicer effects with Python
Based on a 1D approach


"""

import os
import numpy as np
from numpy.fft import fft2, fftshift, ifft2
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


if __name__ == """__main__""":

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    # Sanity check on the sizes of the apertures to match a 4 m.a.s. spaxel scale at 1.5 um
    nom_wave = 1.5          # microns
    spaxels = 4             # m.a.s
    _rho = rho_spaxel_scale(spaxel_scale=spaxels, wavelength=nom_wave)
    check_spaxel_scale(rho_aper=_rho, wavelength=nom_wave)


    class SlicerModel(object):
        """
        Object that models the effect of Image Slicers in light propagation

        """

        def __init__(self, slicer_options, N_PIX, spaxel_scale, N_waves=1, wave0=1.5, waveN=1.5):

            self.N_slices = slicer_options["N_slices"]                                          # Number os Slices to use
            self.spaxels_per_slice = slicer_options["spaxels_per_slice"]                        # Size of the slice in Spaxels
            self.pupil_mirror_aperture = slicer_options["pupil_mirror_aperture"]                # Pupil Mirror Aperture
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
            self.create_pupil_mirror_apertures(self.pupil_mirror_aperture)

            wavefront = self.pupil_masks[1.5]
            nominal_pup = self.pupil_masks[1.5] * np.exp(1j * 2 * np.pi * wavefront / 1.5)
            self.nominal_PSF = (np.abs(fftshift(fft2(nominal_pup, norm='ortho')))) ** 2
            self.PEAK = np.max(self.nominal_PSF)

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

            self.pupil_masks = {}
            x0 = np.linspace(-1., 1., self.N_PIX, endpoint=True)
            xx, yy = np.meshgrid(x0, x0)
            rho = np.sqrt(xx ** 2 + yy ** 2)
            print("\nCreating Pupil Masks:")
            for i, wave in enumerate(self.waves_ratio):
                wavelength = self.wave_range[i]
                print("Wavelength: %.2f microns" % wavelength)
                pupil = (rho <= rho_aper / wave) & (rho >= rho_obsc / wave)
                self.pupil_masks[wavelength] = pupil
            return

        def create_slicer_masks(self):

            # Check the size of the arrays
            U = self.N_PIX * self.spaxel_scale          # Size in [m.a.s] of the Slicer Plane
            u0 = np.linspace(-U/2., U/2., self.N_PIX, endpoint=True)
            uu, vv = np.meshgrid(u0, u0)

            slice_width = self.slice_size_mas
            self.slicer_masks, self.slice_boundaries = [], []
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
                # plt.figure()
                # plt.imshow(mask)
                self.slicer_masks.append(mask)
            # np.sum(np.stack(self.slicer_masks), axis=0)
            return

        def create_pupil_mirror_apertures(self, aperture):

            x0 = np.linspace(-1., 1., self.N_PIX, endpoint=True)
            xx, yy = np.meshgrid(x0, x0)
            self.pupil_mirror_mask = np.abs(yy) <= aperture
            return

        def propagate_pupil_to_slicer(self, wavelength, wavefront):

            # FIXME: possibility of using other wavefronts
            print("Pupil Plane -> Image Slicer Plane")

            pupil_mask = self.pupil_masks[wavelength]
            wavefront = pupil_mask
            complex_pupil = pupil_mask * np.exp(1j * 2*np.pi * wavefront / wavelength)

            # Propagate to Slicer plane
            complex_slicer = fftshift(fft2(complex_pupil, norm='ortho'))

            return complex_slicer

        def propagate_slicer_to_pupil_mirror(self, complex_slicer):

            print("Image Slicer Plane -> Pupil Mirror Plane")
            complex_mirror = []
            for i_slice in range(self.N_slices):        # Loop over the Slices
                mask = self.slicer_masks[i_slice]
                masked_complex_slicer = mask * complex_slicer
                # Pre FFT-Shift to put it back to the format the FFT uses
                _shifted = fftshift(masked_complex_slicer)

                # propagate to Pupil Mirror plane
                complex_pupil_mirror = ifft2(_shifted, norm='ortho')
                complex_mirror.append(complex_pupil_mirror)
            return complex_mirror

        def propagate_pupil_mirror_to_exit_slit(self, complex_mirror):

            print("Pupil Mirror Plane -> Exit Slits")
            exit_slits = []
            for c_mirror in complex_mirror:
                masked_mirror = self.pupil_mirror_mask * c_mirror
                complex_slit = fftshift(fft2(masked_mirror , norm='ortho'))
                # plt.figure()
                # plt.imshow((np.abs(masked_mirror))**2)
                # plt.title()
                image_slit = (np.abs(complex_slit))**2
                exit_slits.append(image_slit)
            image = np.sum(np.stack(exit_slits), axis=0)

            return exit_slits, image

        def propagate_one_wavelength(self, wavelength, wavefront, plot=False):
            """
            Run the propagation from PUPIL Plane to EXIT SLIT Plane for a given wavelength
            :param wavelength:
            :param wavefront:
            """
            print("\nPropagating Wavelength: %.2f microns" % wavelength)
            complex_slicer = slicer.propagate_pupil_to_slicer(wavelength=wavelength, wavefront=wavefront)
            complex_mirror = slicer.propagate_slicer_to_pupil_mirror(complex_slicer)
            exit_slits, image_slit = slicer.propagate_pupil_mirror_to_exit_slit(complex_mirror)

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
                plt.imshow(image_slit / self.PEAK, extent=slicer_extents)
                self.plot_slicer_boundaries()
                plt.xlim([-zoom_size, zoom_size])
                plt.ylim([-zoom_size, zoom_size])
                plt.colorbar()
                plt.title('Exit Slit')

                residual = (image_slit - self.nominal_PSF) / self.PEAK
                m_res = min(np.min(residual), -np.max(residual))
                plt.figure()
                plt.imshow(residual, extent=slicer_extents, cmap='bwr')
                plt.xlim([-zoom_size, zoom_size])
                plt.ylim([-zoom_size, zoom_size])
                self.plot_slicer_boundaries()
                plt.colorbar()
                plt.clim(m_res, -m_res)
                plt.title('Exit Slit - No Slicer')

            return complex_slicer, complex_mirror, image_slit

        def plot_slicer_boundaries(self):

            min_alpha, max_alpha = 0.15, 0.85
            half_slices = (self.N_slices - 1)//2
            alphas = np.linspace(min_alpha, max_alpha,  half_slices)
            alphas = np.concatenate([alphas, [max_alpha], alphas[::-1]])
            for y, alpha in zip(self.slice_boundaries, alphas):
                plt.axhline(y, linestyle='--', color='white', alpha=alpha)
            return

    slicer_options = {"N_slices": 15, "spaxels_per_slice": 11, "pupil_mirror_aperture": 0.95}
    N_PIX = 2048

    slicer = SlicerModel(slicer_options=slicer_options,
                         N_PIX=N_PIX, spaxel_scale=0.5, N_waves=1, wave0=1.5, waveN=1.5)
    for wave in slicer.wave_range:
        plt.figure()
        plt.imshow(slicer.pupil_masks[wave])
        plt.title('Pupil Mask at %.2f microns' % wave)
    plt.show()

    slicer.propagate_one_wavelength(wavelength=1.5, wavefront=0, plot=True)
    plt.show()


    #___________________________________________________________________________________
    """ Impact of Pupil Mirror Aperture """
    apertures = [0.99, 0.95, 0.90, 0.80, 0.75]

    for aper in apertures:
        print("\n-------------------------------------------------------------")
        print("Pupil Mirror Aperture: ", aper)
        slicer_options = {"N_slices": 15, "spaxels_per_slice": 11, "pupil_mirror_aperture": aper}
        slicer = SlicerModel(slicer_options=slicer_options,N_PIX=N_PIX,
                             spaxel_scale=0.5, N_waves=1, wave0=1.5, waveN=1.5)
        _complex_slicer, _complex_mirror, image_slit = slicer.propagate_one_wavelength(wavelength=1.5, wavefront=0)

        slicer_size = slicer.N_PIX * slicer.spaxel_scale / 2
        slicer_extents = [-slicer_size, slicer_size, -slicer_size, slicer_size]
        zoom_size = slicer.N_slices * slicer.slice_size_mas / 2

        # Pupil Mirror plane
        central_slice = (slicer.N_slices - 1) // 2
        pupil_mirror = _complex_mirror[central_slice]
        pupil_image = (np.abs(pupil_mirror)) ** 2
        plt.figure()
        plt.imshow(np.log10(pupil_image), extent=[-1, 1, -1, 1])
        plt.clim(vmin=-10)
        plt.axhline(slicer.pupil_mirror_aperture, linestyle='--', color='white')
        plt.axhline(-slicer.pupil_mirror_aperture, linestyle='--', color='white')
        plt.title('Pupil Mirror [Central Slice] | Aperture: %.2f' % aper)

        # plt.figure()
        # plt.imshow(image_slit / slicer.PEAK, extent=slicer_extents)
        # slicer.plot_slicer_boundaries()
        # plt.xlim([-zoom_size, zoom_size])
        # plt.ylim([-zoom_size, zoom_size])
        # plt.colorbar()
        # plt.xlabel('m.a.s')
        # plt.ylabel('m.a.s')
        # plt.title('Exit Slit')

        residual = (image_slit - slicer.nominal_PSF) / slicer.PEAK
        m_res = min(np.min(residual), -np.max(residual))
        plt.figure()
        plt.imshow(residual, extent=slicer_extents, cmap='bwr')
        plt.xlim([-zoom_size, zoom_size])
        plt.ylim([-zoom_size, zoom_size])
        slicer.plot_slicer_boundaries()
        plt.colorbar()
        plt.clim(m_res, -m_res)
        plt.title('Exit Slit - No Slicer | Aperture: %.2f' % aper)
        plt.xlabel(r'm.a.s')
        plt.ylabel(r'm.a.s')
    plt.show()


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
            along the [-1, 1] line

            :param N_actuators: Number of actuators along the [-1, 1] line
            :param rho_aper: Relative size of the aperture wrt [-1, 1]
            :param rho_obsc: Relative size of the obscuration
            :param radial: if True, we add actuators at the boundaries RHO_APER, RHO_OBSC
            :return: [act (list of actuator centres), delta (actuator separation)], max_freq (max spatial frequency we sense)
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

            wave_ratios = self.slicer_model.waves_ratio
            alpha = 1 / np.sqrt(np.log(100 / h_centers))
            # alpha = alpha_centers

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

        def generate_PSF(self, N_PSF, scale, random=True):

            print("\nGenerating %d PSF images" % N_PSF)

            coef = scale * np.random.uniform(low=-1, high=1, size=(N_PSF, self.N_act))
            PSF_images = []
            for i in range(N_PSF):
                print(i)
                PSF_waves = []      # length is Number of Wavelengths
                for j, wave in enumerate(self.slicer_model.wave_range):
                    pupil = self.pupil_masks[j]
                    wavefront = pupil * np.dot(self.actuator_matrices[j], coef[i])
                    _slicer, _mirror, slit = self.slicer_model.propagate_one_wavelength(wave, wavefront, plot=False)
                    PSF_waves.append(slit)
                PSF_images.append(PSF_waves)
            return np.array(PSF_images), coef



    slicer_options = {"N_slices": 15, "spaxels_per_slice": 11, "pupil_mirror_aperture": 0.95}
    N_PIX = 2048

    slicer = SlicerModel(slicer_options=slicer_options,
                         N_PIX=N_PIX, spaxel_scale=0.5, N_waves=2, wave0=1.5, waveN=2.0)

    N_act = 8
    h_centres = 20
    PSF_generator = SlicerPSFCalculator(slicer_model=slicer, N_actuators=N_act, radial=True, h_centers=h_centres)
    images = PSF_generator.generate_PSF(N_PSF=3, scale=1.5)

    #___________________________________________________________________________________





