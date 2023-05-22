"""Image Processing Classes."""


import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.patches as _patches


# NOTE: lnls560-linux was used in benchmarking
#
#   processor       : 0
#   vendor_id       : GenuineIntel
#   cpu family      : 6
#   model           : 158
#   model name      : Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
#   stepping        : 10
#   microcode       : 0x96
#   cpu MHz         : 900.024
#   cache size      : 12288 KB
#   physical id     : 0
#   siblings        : 12
#   core id         : 0
#   cpu cores       : 6
#   apicid          : 0
#   initial apicid  : 0
#   fpu             : yes
#   fpu_exception   : yes
#   cpuid level     : 22
#   wp              : yes


class FitGaussian:
    """."""

    SATURATION_8BITS = 2**8-1
    SATURATION_12BITS = 2**12-1
    SATURATION_16BITS = 2**16-1
    _SIGMA2FWHM = 2 * _np.sqrt(2*_np.log(2))

    @staticmethod
    def gaussian(indcs, sigma, mean, amplitude, offset):
        """Gaussian curve from distribution parameters."""
        # benchmark for size=1280
        # 25.4 µs ± 378 ns per loop
        # (mean ± std. dev. of 7 runs, 10000 loops each)
        return offset + amplitude * _np.exp(-0.5*((indcs - mean)/sigma)**2)

    @classmethod
    def generate_gaussian_1d(
            cls, indcs, sigma=_np.inf, mean=0, amplitude=0,
            offset=0, rand_amplitude=0, saturation_threshold=None):
        """Generate a gaussian curve with given distribution parameters.

        Args:
            indcs (int | tuple | list | np.array) : Pixel index definition
            sigma (float) : Gaussian sigma value. Defaults to numpy.inf
            mean (float) : Gaussian mean value. Defaults to zero
            amplitude (float) : Gaussian intensity amplitude. Defaults to zero
            offset (float) : Gaussian intensity offset. Defaults to zero
            rand_amplitude (float) : Gaussian point intensity random amplitude.
                Defaults to zero
            saturation_threshold (float) : Intensity above which image is
                tagged saturated. Defaults to None, meaning no check is
                performed.

        Output:
            data (np.array) : gaussian curve
            indcs (np.array) : indice array
        """
        # benchmark for size=1280
        #   39.8 µs ± 148 ns per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)

        indcs, sigma, mean, amplitude, \
            offset, rand_amplitude, saturation_threshold = \
            cls._process_args(
                indcs, sigma, mean, amplitude,
                offset, rand_amplitude, saturation_threshold)

        data = cls.gaussian(indcs, sigma, mean, amplitude, offset)

        if rand_amplitude is not None:
            data += (_np.random.rand(*data.shape) - 0.5) * rand_amplitude
        if saturation_threshold is not None:
            data[data > saturation_threshold] = saturation_threshold
        return data, indcs

    @classmethod
    def generate_gaussian_2d(
            cls, indcs, sigma=None, mean=None, gradient=(0, 0),
            amplitude=0, offset=0,
            rand_amplitude=0, saturation_threshold=None,
            angle=0
            ):
        """Generate a bigaussian with given distribution parameters.

        Args:
            indcs (tuple(2) | list(2) | np.array(2)) :
                2-component (y and x) pixel index definition. Each component is
                a (int | tuple | list | np.array) with pixel indice definition.
            sigma (tuple(2) | list(2) | np.array(2)) :
                mode-1 and mode-2 gaussian sigma values (int | float).
                Defaults to None, corresponding to [numpy.inf]*2
            mean (tuple(2) | list(2) | np.array(2)) :
                mode-1 and mode-2 mean values. Defaults to None, corresponding
                to [0]*2
            amplitude (float) : Bigaussian intensity amplitude,
                Defaults to zero
            offset (float) : Bigaussian intensity offset. Defaults to zero
            rand_amplitude (float) : gaussian point intensity random amplitude.
                Defaults to zero
            saturation_threshold (float) :
                Intensity above which image is set to saturated. Defaults to
                None, in which case no saturation check if performed
            angle (float) : Bigaussian tilt angles [deg]

        Output:
            data (np.array) : gaussian curve
            indcsx (np.array) : x pixel index array (input ref)
            indcsy (np.array) : y pixel index array (input ref)
        """
        # benchmark for size=(1024, 1280)
        # 35.1 ms ± 833 µs per loop
        #   (mean ± std. dev. of 7 runs, 10 loops each)
        indcsx, indcsy = indcs
        sigma1, sigma2 = sigma if sigma else [None] * 2
        mean1, mean2 = mean if mean else [None] * 2

        indcsx, sigma1, mean1, \
            amplitude, offset, rand_amplitude, saturation_threshold = \
            cls._process_args(
                indcsx, sigma1, mean1,
                amplitude, offset, rand_amplitude, saturation_threshold)

        indcsy, sigma2, mean2, \
            amplitude, offset, rand_amplitude, saturation_threshold = \
            cls._process_args(
                indcsy, sigma2, mean2,
                amplitude, offset, rand_amplitude, saturation_threshold)

        y = indcsy - mean2
        x = indcsx - mean1
        mx, my = _np.meshgrid(x, y)
        angle *= _np.pi / 180  # [deg] -> [rad]
        cos_a, sin_a = _np.cos(angle), _np.sin(angle)
        m1 = cos_a * mx - sin_a * my
        m2 = sin_a * mx + cos_a * my
        data = offset + \
            gradient[0] * (mx - x[0]) + gradient[1] * (my - y[0]) + \
            amplitude * _np.exp(-0.5 * ((m1/sigma1)**2 + (m2/sigma2)**2))
        if rand_amplitude:
            data += (_np.random.rand(*data.shape) - 0.5) * rand_amplitude
        if saturation_threshold is not None:
            data[data > saturation_threshold] = saturation_threshold
        return data, indcsx, indcsy

    @staticmethod
    def fit_gaussian(proj, indcs, param0):
        """."""
        sigma0, mean0, amplitude0, offset0 = param0
        indc = indcs - mean0  # centered fitting
        proj = proj.copy() - offset0
        sel = proj > 0  # fit only positive data
        vecy, vecx = proj[sel], indc[sel]
        logy = _np.log(vecy)

        pfit = _np.polynomial.polynomial.polyfit(vecx, logy, 2)
        if pfit[2] < 0:
            sigma = _np.sqrt(-1/pfit[2]/2) if pfit[2] < 0 else 0.0
            mean = pfit[1] * sigma**2
            amplitude = _np.exp(pfit[0] + (mean/sigma)**2/2)
            mean += mean0
            offset = offset0
        else:
            sigma, mean, amplitude, offset = [_np.nan] * 4
        return sigma, mean, amplitude, offset

    @classmethod
    def calc_fit(cls, image, proj, indcs, center):
        """."""
        # get roi gaussian fit
        sigma0 = None
        mean0 = center
        amplitude0 = None
        offset0 = image.intensity_min
        param0 = (sigma0, mean0, amplitude0, offset0)
        param = cls.fit_gaussian(proj, indcs, param0)
        if param[0] > 0:
            gfit, *_ = cls.generate_gaussian_1d(indcs, *param)
            roi_gaussian_fit = gfit
            error = _np.sum((gfit - proj)**2)
            error /= _np.sum(proj**2)
            roi_gaussian_error = 100 * _np.sqrt(error)
        else:
            roi_gaussian_fit = 0 * proj
            roi_gaussian_error = _np.nan
        fit = (param, roi_gaussian_fit, roi_gaussian_error)
        return fit

    @staticmethod
    def conv_sigma2fwhm(sigma):
        """."""
        return sigma * FitGaussian._SIGMA2FWHM

    @staticmethod
    def conv_fwhm2sigma2(fwhm):
        """."""
        return fwhm / FitGaussian._SIGMA2FWHM

    @staticmethod
    def _process_args(
            indcs, sigma, mean, amplitude,
            offset, rand_amplitude, saturation_threshold):
        sigma = sigma or _np.inf
        if isinstance(indcs, (int, float)):
            indcs = _np.arange(int(indcs))
        elif isinstance(indcs, (tuple, list)):
            indcs = _np.array(indcs)
        elif isinstance(indcs, _np.ndarray):
            indcs = _np.array(indcs)
        else:
            raise ValueError('Invalid indcs!')
        if indcs.size < 2:
            raise ValueError('Invalid indcs!')
        elif indcs.size == 2:
            indcs = _np.arange(*indcs)
        res = (
            indcs, sigma, mean, amplitude,
            offset, rand_amplitude, saturation_threshold
            )
        return res


class FitGaussianScipy(FitGaussian):
    """."""

    def __init__(self, use_jacobian=True, maxfev=500):
        """."""
        try:
            from scipy.optimize import curve_fit
        except ModuleNotFoundError as err:
            raise Exception('FitGaussianScipy requires scipy!') from err
        self._use_jacobian = use_jacobian
        self._curve_fit_func = curve_fit
        self._maxfev = maxfev

    @property
    def max_func_evals(self):
        """Return maximum number of evaluations allowed in curve_fit."""
        return self._maxfev

    @property
    def use_jacobian(self):
        """."""
        return self._use_jacobian

    @use_jacobian.setter
    def use_jacobian(self, value):
        """."""
        self._use_jacobian = value

    def jac_gaussian(self, indcs, *params):
        """."""
        sigma, mean, amplitude, offset = params
        vardx = indcs - mean
        varf = - 0.5 * (vardx/sigma)**2
        expf = _np.exp(varf)
        dvarf_dsigma = varf * (-2 / sigma)
        dvarf_dmean = vardx / sigma**2

        dI_dsigma = amplitude * expf * dvarf_dsigma
        dI_dmean = amplitude * expf * dvarf_dmean
        dI_damplitude = expf
        dI_doffset = _np.ones(indcs.size)

        jac = _np.stack((
            dI_dsigma, dI_dmean, dI_damplitude, dI_doffset), axis=1)

        return jac

    def fit_gaussian(self, proj, indcs, param0):
        """."""
        # TODO: use covariance matrix to estimate parameter errors
        jac = self.jac_gaussian if self.use_jacobian else None
        param, *ret = self._curve_fit_func(
            self.gaussian, indcs, proj, param0, jac=jac, maxfev=self._maxfev)
        return param, ret

    def calc_fit(self, image, proj, indcs, center):
        """."""
        # calc param0
        sigma = max(1, image.roi_fwhm / 2.35)
        mean = image.roi_center
        amplitude = image.intensity_max - image.intensity_min
        offset = image.intensity_min
        param0 = (sigma, mean, amplitude, offset)
        param, ret = self.fit_gaussian(proj, indcs, param0)

        gfit = self.gaussian(
            indcs, param[0], param[1], param[2], param[3])
        roi_gaussian_fit = gfit
        error = _np.sum((gfit - proj)**2)
        error /= _np.sum(proj**2)
        roi_gaussian_error = 100 * _np.sqrt(error)
        fit = (param, roi_gaussian_fit, roi_gaussian_error)
        return fit


class Image1D:
    """1D-Images."""

    def __init__(self, data, saturation_threshold=None):
        """."""
        # benchmark for size=1280
        #   968 ns ± 7.72 ns per loop
        #   (mean ± std. dev. of 7 runs, 1000000 loops each)

        self._data = None
        self._saturation_threshold = saturation_threshold
        self._is_saturated = None
        self._update_image(data)

    @property
    def data(self):
        """Return image data as numpy array."""
        return self._data

    @data.setter
    def data(self, value):
        """Set image."""
        self._update_image(value)

    @property
    def saturation_threshold(self):
        """."""
        return self._saturation_threshold

    @saturation_threshold.setter
    def saturation_threshold(self, value):
        """."""
        self._saturation_threshold = value
        self._update_image(self.data)

    @property
    def shape(self):
        """Return image shape"""
        return self.data.shape

    @property
    def size(self):
        """Return number of pixels."""
        return self.data.size

    @property
    def intensity_min(self):
        """Return image min intensity value."""
        return _np.min(self.data)

    @property
    def intensity_max(self):
        """Return image max intensity value."""
        return _np.max(self.data)

    @property
    def intensity_sum(self):
        """Return image sum intensity value."""
        return _np.sum(self.data)

    @property
    def is_saturated(self):
        """Check if image is saturated."""
        return self._is_saturated

    def imshow(self, fig=None, axes=None, crop=None):
        """."""
        crop = crop or [0, self.data.size]

        if None in (fig, axes):
            fig, axes = _plt.subplots()

        data = self.data[slice(*crop)]
        axes.plot(data)
        axes.set_xlabel('pixel indices')
        axes.set_ylabel('Projection intensity')

        return fig, axes

    def generate_gaussian_1d(self, indcs=None, *args, **kwargs):
        """Generate a gaussian with given distribution parameters."""
        indcs = indcs or self.size
        return FitGaussian.generate_gaussian_1d(
            indcs=indcs, *args, **kwargs)

    def __str__(self):
        """."""
        res = ''
        res += f'size            : {self.size}'
        res += f'\nintensity_min   : {self.intensity_min}'
        res += f'\nintensity_max   : {self.intensity_max}'
        res += f'\nintensity_avg   : {self.intensity_sum/self.size}'
        res += f'\nintensity_sum   : {self.intensity_sum}'
        res += f'\nsaturation_val  : {self.saturation_threshold}'
        res += f'\nsaturated       : {self.is_saturated}'
        return res

    @staticmethod
    def update_roi(data, roi):
        """."""
        if roi is None:
            roi = [0, data.size]
        roi = [max(roi[0], 0), min(roi[1], data.size)]
        return roi

    # --- private methods ---

    def _update_image(self, data):
        """."""
        self._data = _np.asarray(data)
        if self.saturation_threshold is None:
            self._is_saturated = False
        else:
            self._is_saturated = \
                _np.any(self.data >= self.saturation_threshold)


class Image2D:
    """2D-Images."""

    SATURATION_8BITS = FitGaussian.SATURATION_8BITS
    SATURATION_12BITS = FitGaussian.SATURATION_12BITS
    SATURATION_16BITS = FitGaussian.SATURATION_16BITS

    def __init__(
            self, data,
            saturation_threshold=SATURATION_8BITS,
            intensity_threshold=0):
        """."""
        # benchmark for sizes=(1024, 1280):
        #   874 µs ± 11.9 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        self._data = None
        self._saturation_threshold = saturation_threshold
        self._intensity_threshold = intensity_threshold
        self._is_saturated = None
        self._is_with_image = None
        self._update_image(data)

    @property
    def data(self):
        """Return image data as numpy array."""
        return self._data

    @data.setter
    def data(self, value):
        """Set image."""
        self._update_image(value)

    @property
    def saturation_threshold(self):
        """."""
        return self._saturation_threshold

    @saturation_threshold.setter
    def saturation_threshold(self, value):
        """."""
        self._saturation_threshold = value
        self._update_image(self.data)

    @property
    def intensity_threshold(self):
        """."""
        return self._intensity_threshold

    @intensity_threshold.setter
    def intensity_threshold(self, value):
        """."""
        self._intensity_threshold = value
        self._update_image(self.data)

    @property
    def shape(self):
        """Return image shape"""
        return self.data.shape

    @property
    def sizey(self):
        """Return image first dimension size."""
        return self.shape[0]

    @property
    def sizex(self):
        """Return image second dimension size."""
        return self.shape[1]

    @property
    def size(self):
        """Return number of pixels."""
        return self.sizey * self.sizex

    @property
    def intensity_min(self):
        """Return image min intensity value."""
        # benchmark for sizes=(1024, 1280):
        #   383 µs ± 14.7 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        return _np.min(self.data)

    @property
    def intensity_max(self):
        """Return image max intensity value."""
        return _np.max(self.data)

    @property
    def intensity_sum(self):
        """Return image sum intensity value."""
        # benchmark for sizes=(1024, 1280):
        #   348 µs ± 3.82 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        return _np.sum(self.data)

    @property
    def is_saturated(self):
        """Check if image is saturated."""
        return self._is_saturated

    @property
    def is_with_image(self):
        """Check if image has signal."""
        return self._is_with_image

    def imshow(self, fig=None, axes=None, cropx=None, cropy=None):
        """."""
        cropx, cropy = Image2D.update_roi(self.data, cropx, cropy)

        if None in (fig, axes):
            fig, axes = _plt.subplots()

        data = self.data[slice(*cropy), slice(*cropx)]
        axes.imshow(data)

        return fig, axes

    def generate_gaussian_2d(self, indcsx=None, indcsy=None, *args, **kwargs):
        """Generate a bigaussian with distribution parameters."""
        indcsy = indcsy or self.sizey
        indcsx = indcsx or self.sizex
        indcs = [indcsx, indcsy]
        return FitGaussian.generate_gaussian_2d(indcs, *args, **kwargs)

    def __str__(self):
        """."""
        res = ''
        res += f'sizey           : {self.sizey}'
        res += f'\nsizex           : {self.sizex}'
        res += f'\nintensity_min   : {self.intensity_min}'
        res += f'\nintensity_max   : {self.intensity_max}'
        res += f'\nintensity_avg   : {self.intensity_sum/self.size}'
        res += f'\nintensity_sum   : {self.intensity_sum}'
        res += f'\nsaturation_val  : {self.saturation_threshold}'
        res += f'\nsaturated       : {self.is_saturated}'
        return res

    @staticmethod
    def update_roi(data, roix, roiy):
        """."""
        if roiy is None:
            roiy = [0, data.shape[0]]
        if roix is None:
            roix = [0, data.shape[1]]
        roiy = [max(roiy[0], 0), min(roiy[1], data.shape[0])]
        roix = [max(roix[0], 0), min(roix[1], data.shape[1])]
        return roix, roiy

    @staticmethod
    def project_image(data, axis):
        axis_ = 1 if axis == 0 else 0
        image = _np.sum(data, axis=axis_)
        return image

    # --- private methods ---

    def _update_image(self, data):
        """."""
        self._data = _np.asarray(data)
        if self.saturation_threshold is None:
            self._is_saturated = False
        else:
            self._is_saturated = \
                _np.any(self.data >= self.saturation_threshold)
        if self.intensity_threshold is None:
            self._is_with_image = True
        else:
            self._is_with_image = self.intensity_max > self.intensity_threshold


class Image1D_ROI(Image1D):
    """1D-Image ROI."""

    def __init__(self, data, roi=None, *args, **kwargs):
        """."""
        # benchmark for size=1280
        #   28.8 µs ± 194 ns per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)

        self._roi = None
        self._roi_indcs = None
        self._roi_proj = None
        self._roi_center = None
        self._roi_fwhm = None
        super().__init__(data=data, *args, **kwargs)
        self._update_image_roi(roi)

    @property
    def roi(self):
        """."""
        return self._roi

    @roi.setter
    def roi(self, value):
        """."""
        self._update_image_roi(value)

    @property
    def roi_indcs(self):
        """Image roi indices."""
        return self._roi_indcs

    @property
    def roi_proj(self):
        """Return image roi projection."""
        return self._roi_proj

    @property
    def roi_center(self):
        """Image roi center position."""
        return self._roi_center

    @property
    def roi_fwhm(self):
        """Image roi fwhm."""
        return self._roi_fwhm

    def update_roi_with_fwhm(self, fwhm_factor=2.0):
        """."""
        roi = Image1D_ROI.calc_roi_with_fwhm(self, fwhm_factor)
        self.roi = roi  # triggers recalc of center and fwhm

    def imshow(
            self, fig=None, axes=None, crop = None,
            color_ellip=None, color_roi=None):
        """Show image.

        Args:
            fig (None | matplotlib.figure) : Handle to figure.
                Defaults to None (create a new fig, axes)
            axes (None | matplotlib.axes) : Hande to axes.
                Defaults to None (create a new fig, axes)
            crop (tuple | list | numpy.array) : Two-element array with
                image pixel bounds to crop. Defaults to None and the entire
                image is ploted.
            color_ellip (str | RGB color | None): color to use for image
                ellipse plot. Defaults to None, in which case the color 'tab:red'
                is used. If it is set to string 'no' no ellipse is ploted.
            color_roi (str | RGB color | None): color to use for image
                roi rectangle plot. Defaults to None, in case the RGB color
                [0.5, 0.5, 0] is used. If it is set to string 'no' no
                roi is ploted.
        """
        color_ellip = None if color_ellip == 'no' else color_ellip or 'tab:red'
        color_roi = None if color_roi == 'no' else color_roi or [0.5, 0.5, 0]
        crop = crop or [0, self.data.size]

        if None in (fig, axes):
            fig, axes = _plt.subplots()

        # plot image
        data = Image1D_ROI._trim_image(self.data, crop)
        axes.plot(data)

        if color_ellip:
            centerx = self.roi_center - crop[0]
            # plot center
            axes.axvline(x=centerx, color=color_ellip)
            axes.axvline(x=centerx + self.roi_fwhm/2, ls='--',
                color=color_ellip)
            axes.axvline(x=centerx - self.roi_fwhm/2, ls='--',
                color=color_ellip)

        if color_roi:
            # plot roi
            roi1, roi2 = self.roi
            axes.axvline(x=roi1, color=color_roi)
            axes.axvline(x=roi2, color=color_roi)

        return fig, axes

    def create_trimmed(self):
        """Create a new image trimmed to roi."""
        # benchmark for size=1280:
        #   29.3 µs ± 390 ns per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)

        data = Image1D_ROI._trim_image(self.data, self.roi)
        return Image1D_ROI(data=data)

    def __str__(self):
        """."""
        res = super().__str__()
        res += f'\nroi             : {self.roi}'
        res += f'\nroi_center      : {self.roi_center}'
        res += f'\nroi_fwhm        : {self.roi_fwhm}'

        return res

    def _update_image_roi(self, roi):
        """."""
        # get roi, indices and slice data
        roi = Image1D_ROI.update_roi(self._data, roi)
        indcs = Image1D_ROI._calc_indcs(self._data, roi)
        proj = self.data[slice(*roi)]

        # calc center and fwhm
        dmin, dmax = self.data.min(), self.data.max()
        hmax = _np.where(proj > dmin + (dmax - dmin)/2)[0]
        fwhm = hmax[-1] - hmax[0] if len(hmax) > 1 else 0
        center = indcs[0] + _np.argmax(proj)

        self._roi, self._roi_indcs, self._roi_proj, \
            self._roi_center, self._roi_fwhm = roi, indcs, proj, center, fwhm

    @staticmethod
    def _calc_indcs(data, roi=None):
        """Return roi indices within image"""
        if roi is None:
            roi = [0, data.size]
        if roi[1] <= data.size:
            return _np.arange(data.size)[slice(*roi)]
        else:
            return None

    @staticmethod
    def _trim_image(image, roi):
        return image[slice(*roi)]

    @classmethod
    def calc_roi_with_fwhm(cls, image, fwhm_factor):
        """."""
        roi1 = int(image.roi_center - fwhm_factor * (image.roi_fwhm/2))
        roi2 = int(image.roi_center + fwhm_factor * (image.roi_fwhm/2))
        roi = [roi1, roi2]
        return cls.update_roi(image.data, roi)


class Image2D_ROI(Image2D):
    """2D-Image ROI."""

    def __init__(self, data, roix=None, roiy=None, *args, **kwargs):
        """."""
        # benchmark for sizes=(1024, 1280)
        #   1.71 ms ± 203 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        self._imagey = None
        self._imagex = None
        super().__init__(data=data, *args, **kwargs)
        self._update_image_roi(roix, roiy)

    @property
    def imagey(self):
        """."""
        return self._imagey

    @property
    def imagex(self):
        """."""
        return self._imagex

    @property
    def roiy(self):
        """."""
        return self.imagey.roi

    @roiy.setter
    def roiy(self, value):
        """."""
        self._update_image_roi(self.imagex.roi, value)

    @property
    def roix(self):
        """."""
        return self.imagex.roi

    @roix.setter
    def roix(self, value):
        """."""
        self._update_image_roi(value, self.imagey.roi)

    @property
    def roi(self):
        """."""
        return [self.imagex.roi, self.imagey.roi]

    @roi.setter
    def roi(self, value):
        """."""
        self._update_image_roi(*value)

    def update_roi_with_fwhm(self, fwhmx_factor=2, fwhmy_factor=2):
        """."""
        self.imagex.update_roi_with_fwhm(fwhm_factor=fwhmx_factor)
        self.imagey.update_roi_with_fwhm(fwhm_factor=fwhmy_factor)

    def imshow(
            self, fig=None, axes=None,
            cropx = None, cropy = None,
            color_ellip=None, color_roi=None):
        """Show image.

        Args:
            fig (None | matplotlib.figure) : Handle to figure.
                Defaults to None (create a new fig, axes)
            axes (None | matplotlib.axes) : Hande to axes.
                Defaults to None (create a new fig, axes)
            cropx (tuple | list | numpy.array) : Two-element array with
                image pixel bounds to crop in X. Defaults to None and the
                entire image is ploted.
            cropy (tuple | list | numpy.array) : Two-element array with
                image pixel bounds to crop in Y. Defaults to None and the
                entire image is ploted.
            color_ellip (str | RGB color | None): color to use for image
                ellipse plot. Defaults to None, in which case the color 'tab:red'
                is used. If it is set to string 'no' no ellipse is ploted.
            color_roi (str | RGB color | None): color to use for image
                roi rectangle plot. Defaults to None, in case the RGB color
                [0.5, 0.5, 0] is used. If it is set to string 'no' no
                roi is ploted.
        """

        return Image2D_ROI.imshow_images(
            self.data, self.imagex, self.imagey, self.roix, self.roiy,
            fig=fig, axes=axes, cropx = cropx, cropy = cropy,
            color_ellip=color_ellip, color_roi=color_roi)

    def create_trimmed(self):
        """Create a new image timmed to roi."""
        # benchmark for sizes=(1024, 1280), roi all
        #   231 µs ± 3.65 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        data = Image2D_ROI._trim_image(self.data, self.roix, self.roiy)
        return Image2D_ROI(data=data)

    def __str__(self):
        """."""
        res = super().__str__()
        res += '\n--- projx ---\n'
        res += self.imagex.__str__()
        res += '\n--- projy ---\n'
        res += self.imagey.__str__()

        return res

    def _update_image_roi(self, roix, roiy):
        """."""
        roix, roiy = Image2D.update_roi(self.data, roix, roiy)
        data = self.project_image(self._data, 0)
        self._imagey = Image1D_ROI(data=data, roi=roiy)
        data = self.project_image(self._data, 1)
        self._imagex = Image1D_ROI(data=data, roi=roix)

    @classmethod
    def imshow_images(
            cls, data, imagex, imagey, roix, roiy, angle=0,
            centerx=None, centery=None, fwhmx=None, fwhmy=None,
            fig=None, axes=None, cropx=None, cropy=None,
            color_ellip=None, color_roi=None, color_axes=None):
        """Show image.

        Args:
            data (numpy.array | list of list) : 2-index image data
            imagex (Image1D_ROI): image projection in X
            imagey (Image1D_ROI): image projection in Y
            roix (tuple | list | numpy.array) : Two-element array with
                image roi in X.
            roiy (tuple | list | numpy.array) : Two-element array with
                image roi in Y.
            angle (float) : Rotation angle of ellipse to be ploted. Defaults
                to 0. Unit: degree.
            centerx (float) : center of image in X. Defaults to None
             (in which case the center of imagex is used)
            centery (float) : center of image in Y. Defaults to None
             (in which case the center of imagey is used)
            fwhmx (float) : FWHM of image in X. Defaults to None
             (in which case the fwhm of imagex is used)
            fwhmy (float) : FWHM of image in Y. Defaults to None
             (in which case the fwhm of imagey is used)
            fig (None | matplotlib.figure) : Handle to figure.
                Defaults to None (create a new fig, axes)
            axes (None | matplotlib.axes) : Hande to axes.
                Defaults to None (create a new fig, axes)
            cropx (tuple | list | numpy.array) : Two-element array with
                image pixel bounds to crop in X. Defaults to None and the
                entire image is ploted.
            cropy (tuple | list | numpy.array) : Two-element array with
                image pixel bounds to crop in Y. Defaults to None and the
                entire image is ploted.
            color_ellip (str | RGB color | None): color to use for image
                ellipse plot. Defaults to None, in which case the color
                'tab:red' is used. If it is set to string 'no' no ellipse
                 is ploted.
            color_roi (str | RGB color | None): color to use for image
                roi rectangle plot. Defaults to None, in which case the color
                'yellow' is used. If it is set to string 'no' no
                roi is ploted.
            color_axes (str | RGB color | None): color to use for image
                principal axes. Defaults to None, in which case the color
                'blue' is used. If it is set to string 'no' no
                axes are ploted."""
        color_ellip = None if color_ellip == 'no' else color_ellip or 'tab:red'
        color_roi = None if color_roi == 'no' else color_roi or 'yellow'
        color_axes = None if color_axes == 'no' else color_axes or 'blue'
        centerx = centerx if centerx is not None else imagex.roi_center
        centery = centery if centery is not None else imagey.roi_center
        fwhmx = fwhmx if fwhmx is not None else imagex.roi_fwhm
        fwhmy = fwhmy if fwhmy is not None else imagey.roi_fwhm

        cropx, cropy = cls.update_roi(data, cropx, cropy)
        x0, y0 = cropx[0], cropy[0]
        center = centerx - x0, centery - y0

        if None in (fig, axes):
            fig, axes = _plt.subplots()

        # plot image
        data = cls._trim_image(data, cropx, cropy)
        axes.imshow(data, extent=None)

        if color_axes:
            center_ = centerx, centery
            [x1, x2], [y1, y2] = Image2D_ROI._get_normal_axes(
                center_, angle, imagex, imagey, slope_inv_flag=False)
            axes.plot([x1 - x0, x2 - x0], [y1 - y0, y2 - y0], '--', color=color_axes)
            [x1, x2], [y1, y2] = Image2D_ROI._get_normal_axes(
                center_, angle, imagex, imagey, slope_inv_flag=True)
            axes.plot([x1 - x0, x2 - x0], [y1 - y0, y2 - y0], '--', color=color_axes)

        if color_ellip:
            # plot center
            axes.plot(*center, 'o', ms=2, color=color_ellip)

            # plot intersecting ellipse at half maximum
            ellipse = _patches.Ellipse(
                xy=center, width=fwhmx, height=fwhmy,
                angle=-angle, linewidth=1,
                edgecolor=color_ellip, fill='false', facecolor='none')
            axes.add_patch(ellipse)

        if color_roi:
            # plot roi
            roix1, roix2 = roix
            roiy1, roiy2 = roiy
            width, height = _np.abs(roix2-roix1), _np.abs(roiy2-roiy1)
            rect = _patches.Rectangle(
                (roix1 - x0, roiy1 - y0),
                width, height, linewidth=1, edgecolor=color_roi,
                fill='False',
                facecolor='none')
            axes.add_patch(rect)

        return fig, axes

    @classmethod
    def _get_normal_axes(cls, center, angle, imagex, imagey, slope_inv_flag):
        # transform input angle
        if slope_inv_flag:
            angle += 90
        angle *= _np.pi / 180

        # check if angle corresponds to special cases n*pi or n*(pi/2) tilt.
        _SMALL_ANGLE_DEV = 1e-8
        sina = _np.sin(-angle)
        if abs(sina) < _SMALL_ANGLE_DEV:
            return imagex.roi, [center[1], center[1]]
        elif 1 - abs(sina) < _SMALL_ANGLE_DEV:
            return [center[0], center[0]], imagey.roi
        else:
            slope = _np.tan(-angle)

        # define auxiliary function
        def get_axis_point(x, b, slope):
            y = slope * x + b
            if y < imagey.roi[0]:
                x = (imagey.roi[0] - b) / slope
                y = imagey.roi[0] + 1
            elif y > imagey.roi[1]:
                x = (imagey.roi[1] - b) / slope
                y = imagey.roi[1] - 1
            return x, y

        # general case of angle
        b = center[1] - slope * center[0]
        x1 = imagex.roi[0] + 1
        x1, y1 = get_axis_point(x1, b, slope)
        x2 = imagex.roi[1] - 1
        x2, y2 = get_axis_point(x2, b, slope)

        return [x1, x2], [y1, y2]

    @staticmethod
    def _trim_image(image, roix, roiy):
        return image[slice(*roiy), slice(*roix)]


class Image2D_CMom(Image2D_ROI):
    """Image 2D with normalized central moments."""

    def __init__(self, *args, **kwargs):
        """."""
        # benchmark for sizes=(1024, 1280)
        #   37.7 ms ± 701 µs per loop
        #   (mean ± std. dev. of 7 runs, 10 loops each)
        # benchmark for sizes=(1024, 1280), roix=[400, 800], roiy=[400, 600]
        #   3.48 ms ± 92.8 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)

        self._roix_meshgrid = None
        self._roiy_meshgrid = None
        self._cmomx = None
        self._cmomy = None
        self._cmomyy = None
        self._cmomxy = None
        self._cmomxx = None
        self._angle = None
        self._sigma1 = None
        self._sigma2 = None
        self._sigmax = None
        self._sigmay = None
        super().__init__(*args, **kwargs)

    @property
    def roix_meshgrid(self):
        """."""
        return self._roix_meshgrid

    @property
    def roiy_meshgrid(self):
        """."""
        return self._roiy_meshgrid

    @property
    def cmomy(self):
        """."""
        return self._cmomy

    @property
    def cmomx(self):
        """."""
        return self._cmomx

    @property
    def cmomyy(self):
        """."""
        return self._cmomyy

    @property
    def cmomxy(self):
        """."""
        return self._cmomxy

    @property
    def cmomxx(self):
        """."""
        return self._cmomxx

    @property
    def angle(self):
        """Return tilt angle [deg]"""
        return self._angle

    @property
    def sigma1(self):
        """."""
        return self._sigma1

    @property
    def sigma2(self):
        """."""
        return self._sigma2

    @property
    def sigmax(self):
        """."""
        return self._sigmax

    @property
    def sigmay(self):
        """."""
        return self._sigmay

    def calc_central_moment(self, order_x, order_y):
        """."""
        # benchmark for sizes=(1024, 1280)
        # 10.7 ms ± 87.4 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)
        # benchmark for sizes=(1024, 1280), roix=[400, 800], roiy=[400, 600]
        #   223 µs ± 1.72 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        return self.calc_cmom(
            self.data,
            roix_meshgrid=self.roix_meshgrid, roiy_meshgrid=self.roiy_meshgrid,
            roix=self.imagex.roi, roiy=self.imagey.roi,
            cmomx=self._cmomx, cmomy=self._cmomy,
            order_x=order_x, order_y=order_y)

    def imshow(self, *args, **kwargs):
        """."""
        centerx = kwargs.pop('centerx', self.cmomx)
        centery = kwargs.pop('centery', self.cmomy)
        angle = kwargs.pop('angle', self.angle)
        fwhmx = kwargs.pop('fwhmx', None)
        fwhmy = kwargs.pop('fwhmy', None)
        if fwhmx is None:
            sigmax = kwargs.pop('sigmax', _np.sqrt(self.cmomxx))
            fwhmx = FitGaussian.conv_sigma2fwhm(sigmax)
        if fwhmy is None:
            sigmay = kwargs.pop('sigmay', _np.sqrt(self.cmomyy))
            fwhmy = FitGaussian.conv_sigma2fwhm(sigmay)

        fig, axes = Image2D_ROI.imshow_images(
            self.data, self.imagex, self.imagey, self.roix, self.roiy,
            *args, angle=angle, centerx=centerx, centery=centery,
            fwhmx=fwhmx, fwhmy=fwhmy, **kwargs)
        return fig, axes

    def __str__(self):
        """."""
        res = super().__str__()
        res += '\n--- cmom ---'
        res += f'\ncmomx           : {self.cmomx}'
        res += f'\ncmomy           : {self.cmomy}'
        res += f'\ncmomxx          : {self.cmomxx}'
        res += f'\ncmomyy          : {self.cmomyy}'
        res += f'\ncmomxy          : {self.cmomxy}'
        res += f'\nsigma1          : {self.sigma1}'
        res += f'\nsigma2          : {self.sigma2}'
        res += f'\nangle           : {self.angle}'
        res += f'\nsigmax          : {self.sigmax}'
        res += f'\nsigmay          : {self.sigmay}'
        return res

    def _update_image_roi(self, roix=None, roiy=None):
        """."""
        super()._update_image_roi(roix=roix, roiy=roiy)
        self._roix_meshgrid, self._roiy_meshgrid = \
            self.calc_meshgrids(self.imagex, self.imagey)
        self._cmomx, self._cmomy = self.calc_cmom1(self.imagex, self.imagey)
        self._cmomxx = self.calc_central_moment(2, 0)
        self._cmomxy = self.calc_central_moment(1, 1)
        self._cmomyy = self.calc_central_moment(0, 2)

        # calc angle and sigmas
        angle, sigma1, sigma2 = self.calc_angle_normal_sigmas(
            self.cmomxx, self.cmomyy, self.cmomxy)

        # calc sigmas in rotated (original) axes
        sigmax, sigmay = self.calc_rotated_sigma(angle, sigma1, sigma2)

        # calc angle [deg] and sigmas
        self._angle = angle
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._sigmax = sigmax
        self._sigmay = sigmay

    @staticmethod
    def calc_meshgrids(imagex : Image1D_ROI, imagey : Image1D_ROI):
        """."""
        roix_meshgrid, roiy_meshgrid = \
            _np.meshgrid(imagex.roi_indcs, imagey.roi_indcs)
        return roix_meshgrid, roiy_meshgrid

    @staticmethod
    def calc_cmom1(
            imagex : Image1D_ROI, imagey : Image1D_ROI, intensity_order=1):
        """."""
        # benchmark for sizes=(1024, 1280)
        #   18.4 µs ± 102 ns per loop
        #   (mean ± std. dev. of 7 runs, 100000 loops each)
        # benchmark for sizes=(1024, 1280)
        #   15.5 µs ± 131 ns per loop
        #   (mean ± std. dev. of 7 runs, 100000 loops each)

        datax, datay = imagex.roi_proj, imagey.roi_proj
        if intensity_order > 1:
            datax_n, datay_n = datax.copy(), datay.copy()
            for _ in range(intensity_order-1):
                datax_n = datax_n * datax
                datay_n = datay_n * datay
            cmom0x = _np.sum(datax_n)
            cmom0y = _np.sum(datay_n)
        else:
            datax_n, datay_n = datax, datay
            cmom0x = cmom0y = _np.sum(datax_n)
        cmomx = _np.sum(datax_n * imagex.roi_indcs) / cmom0x
        cmomy = _np.sum(datay_n * imagey.roi_indcs) / cmom0y
        return cmomx, cmomy

    @staticmethod
    def calc_cmom(
            data, roix_meshgrid, roiy_meshgrid, roix, roiy,
            cmomx, cmomy, order_x, order_y, intensity_threshold=0.01):
        """."""
        # benchmark for sizes=(1024, 1280)
        #   10.6 ms ± 49.3 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)
        # benchmark for sizes=(1024, 1280), roix=[400, 800], roiy=[400, 600]
        #   223 µs ± 1.66 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)
        data = data[slice(*roiy), slice(*roix)]
        # sel = data > intensity_threshold * data.max()
        mgx, mgy = roix_meshgrid - cmomx, roiy_meshgrid - cmomy
        # data = data[sel]
        # mgx, mgy = mgx[sel], mgy[sel]
        # data = data / _np.sum(data)
        mompq = _np.sum(mgx**order_x * mgy**order_y * data)
        mompq /= _np.sum(data)
        return mompq

    @staticmethod
    def calc_angle_normal_sigmas(cmomxx, cmomyy, cmomxy):
        """Return angle [deg] and normal mode sigmas from second moments."""
        # benchmark
        #   92.3 µs ± 704 ns per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)

        # SVD decomposition of second moment matrix
        # print('cmomxx', cmomxx)
        # print('cmomyy', cmomyy)
        # print('cmomxy', cmomxy)
        sigma = _np.array([[cmomxx, cmomxy], [cmomxy, cmomyy]])
        u, s, vt = _np.linalg.svd(sigma, hermitian=True)
        sigma1, sigma2 = _np.sqrt(s)  # sigma1 is largest
        # print('s', s)
        # print('u', u)
        # print('vt', vt)
        axis1, axis2 = vt.T
        axis1, axis2 = u.T
        if axis1[0] < 0:
            axis1 *= -1
        # print('axis1', axis1)
        angle = - _np.arctan2(axis1[1], axis1[0]) * 180 / _np.pi

        return angle, sigma1, sigma2

    @staticmethod
    def calc_rotated_sigma(angle, sigma1, sigma2):
        """Convert normal sigmas to sigmas in rotated axes give angle [deg]."""
        angle *= _np.pi / 180
        cosa2, sina2 = _np.cos(angle)**2, _np.sin(angle)**2
        sigma1_2, sigma2_2 = sigma1**2, sigma2**2
        sigmax = _np.sqrt(cosa2 * sigma1_2 + sina2 * sigma2_2)
        sigmay = _np.sqrt(sina2 * sigma1_2 + cosa2 * sigma2_2)
        return sigmax, sigmay


class Image1D_Fit(Image1D_ROI):
    """1D Image Fit."""

    def __init__(self, *args, fitgauss=None, **kwargs):
        """."""
        # benchmark for size=1280
        #   586 µs ± 1.56 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        self._roi_mean = None
        self._roi_sigma = None
        self._roi_amp = None
        self._roi_fit = None
        self._roi_fit_error = None
        self._fitgauss = fitgauss or FitGaussianScipy()
        super().__init__(*args, **kwargs)
        self._update_image_roi(*args, **kwargs)

    @property
    def roi_sigma(self):
        """Image roiy fitted gaussian sigma."""
        return self._roi_sigma

    @property
    def roi_mean(self):
        """Image roiy fitted gaussian mean."""
        return self._roi_mean

    @property
    def roi_amplitude(self):
        """Image roiy fitted gaussian amplitude."""
        return self._roi_amp

    @property
    def roi_fit_error(self):
        """."""
        return self._roi_fit_error

    @property
    def roi_fit(self):
        """."""
        return self._roi_fit, self.roi_indcs

    @property
    def invalid_fit(self):
        """."""
        is_nan = _np.any(_np.isnan([
            self.roi_amplitude,
            self.roi_mean,
            self.roi_sigma,
            self.roi_fit_error,
            ]))
        is_inf = _np.any(_np.isinf([
            self.roi_amplitude,
            self.roi_mean,
            self.roi_sigma,
            self.roi_fit_error,
            ]))
        return is_nan or is_inf

    def set_saturation_flag(self, value):
        """."""
        self._is_saturated = value is True

    def plot_projection(
            self, fig=None, axes=None):
        """."""
        if None in (fig, axes):
            fig, axes = _plt.subplots()

        color = [0, 0.7, 0]

        axes.plot(
            self.roi_indcs, self.roi_proj,
            color=color, alpha=1.0,
            lw=5, label='roi_proj')
        vecy, vecx = self.roi_fit
        if vecy is not None:
            axes.plot(
                vecx, vecy, color=[0.5, 1, 0.5], alpha=1.0,
                lw=2, label='roix_fit')

        axes.legend()
        axes.grid()
        axes.set_ylabel('ROI pixel index')
        axes.set_ylabel('Projection Intensity')

    def __str__(self):
        """."""
        res = super().__str__()
        res += f'\nroi_amplitude   : {self.roi_amplitude}'
        res += f'\nroi_mean        : {self.roi_mean}'
        res += f'\nroi_sigma       : {self.roi_sigma}'
        res += f'\nroi_fit_err     : {self.roi_fit_error} %'

        return res

    def _update_image_roi(self, roi=None, *args, **kwargs):
        """."""
        super()._update_image_roi(roi=roi)

        # fit roi
        param, roi_fit, roi_error = \
            self._fitgauss.calc_fit(
                self, self.roi_proj, self.roi_indcs, self.roi_center)
        self._roi_sigma, self._roi_mean, self._roi_amp, _ = param
        self._roi_fit, self._roi_fit_error = roi_fit, roi_error


class Image2D_Fit(Image2D):
    """2D Image Fit."""

    def __init__(
            self, roix=None, roiy=None, fitgauss=None,
            use_svd4theta=False, *args, **kwargs):
        """."""
        # benchmark for sizes=(1024, 1280)
        #   21.2 ms ± 1.78 ms per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)
        # benchmark for sizes=(1024, 1280), roix=[350, 849], roiy=[399, 600]
        #   5.49 ms ± 48.8 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)

        self._use_svd4theta = use_svd4theta
        self._fitx = None
        self._fity = None
        self._angle = None
        self._sigma1 = None
        self._sigma2 = None
        self._fitgauss = fitgauss or FitGaussianScipy()
        super().__init__(*args, **kwargs)
        self._update_image_fit(roix=roix, roiy=roiy)

    @property
    def use_svd4theta(self):
        """Return whether SVD of 2nd-moment matrix is used for theta calc."""
        return self._use_svd4theta

    @use_svd4theta.setter
    def use_svd4theta(self, value):
        """Set whether SVD of 2nd-moment matrix is used for theta calc."""
        self._use_svd4theta = bool(value)

    @property
    def fity(self):
        """."""
        return self._fity

    @property
    def fitx(self):
        """."""
        return self._fitx

    @property
    def roiy(self):
        """."""
        return self.fity.roi

    @roiy.setter
    def roiy(self, value):
        """."""
        self._update_image_fit(roix=self.fitx.roi, roiy=value)

    @property
    def roix(self):
        """."""
        return self.fitx.roi

    @roix.setter
    def roix(self, value):
        """."""
        self._update_image_fit(roix=value, roiy=self.fity.roi)

    @property
    def roi(self):
        """."""
        return self.fitx.roi, self.fity.roi

    @roi.setter
    def roi(self, value):
        """."""
        self._update_image_fit(*value)

    @property
    def angle(self):
        """Return tilt angle [deg]"""
        return self._angle

    @property
    def sigma1(self):
        """."""
        return self._sigma1

    @property
    def sigma2(self):
        """."""
        return self._sigma2

    def update_roi_with_fwhm(self, fwhmx_factor=2, fwhmy_factor=2):
        """."""
        self.fitx.update_roi_with_fwhm(fwhm_factor=fwhmx_factor)
        self.fity.update_roi_with_fwhm(fwhm_factor=fwhmy_factor)

    def calc_angle_normal_sigmas(self):
        """Calculate image tilt angle [deg] and normal axes sigmas."""
        # benchmark for sizes=(1024, 1280)
        #   12.3 ms ± 120 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)
        # benchmark for sizes=(1024, 1280), roix=[350, 849], roiy=[399, 600]
        #   823 µs ± 16.7 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        roix_meshgrid, roiy_meshgrid = \
            Image2D_CMom.calc_meshgrids(self.fitx, self.fity)
        # cmomx, cmomy = Image2D_CMom.calc_cmom1(self.fitx, self.fity, intensity_order=4)
        cmomx, cmomy = self.fitx.roi_mean, self.fity.roi_mean

        # calc central moments
        args = (
            self.data,
            roix_meshgrid, roiy_meshgrid,
            self.fitx.roi, self.fity.roi,
            cmomx, cmomy)
        cmomxy = Image2D_CMom.calc_cmom(*args, order_x=1, order_y=1)
        # cmomxx = Image2D_CMom.calc_cmom(*args, order_x=2, order_y=0)
        # cmomyy = Image2D_CMom.calc_cmom(*args, order_x=0, order_y=2)
        cmomxx = self.fitx.roi_sigma ** 2  # from fit instead of from cmom!
        cmomyy = self.fity.roi_sigma ** 2  # from fit instead of from cmom!

        # print('sigmax', self.fitx.roi_sigma)
        # print('sigmay', self.fity.roi_sigma)
        # print('roix', self.fitx.roi, 'cmomx', cmomx)
        # print('roiy', self.fity.roi, 'cmomy', cmomy)
        # print('roix_indcs', self.fitx.roi_indcs)
        # print('roiy_indcs', self.fity.roi_indcs)
        # print('cmomxx', cmomxx)
        # print('cmomyy', cmomyy)
        # print('cmomxy', cmomxy)

        # calc angle and sigmas
        angle, sigma1, sigma2 = Image2D_CMom.calc_angle_normal_sigmas(
            cmomxx, cmomyy, cmomxy)

        return angle, sigma1, sigma2

    def calc_angle_with_roi(self):
        """Calculate image tilt angle within ROI.
        A linear y = ax + b fit is performed over the image and the tilt angle
        is taken from the arctan of the angular coefficient. Each image point
        is weighted by the fourth power of the image intensity.
        """
        roix, roiy = self.fitx.roi, self.fity.roi
        indcsx, indcsy = self.fitx.roi_indcs, self.fity.roi_indcs
        mx, my = _np.meshgrid(indcsx, indcsy)
        data = self.data[slice(*roiy), slice(*roix)]
        data = data * data
        data *= data
        mxd = mx * data
        a11 = _np.sum(data)
        a12 = _np.sum(mxd)
        a22 = _np.sum(mx * mxd)
        b1 = _np.sum(my * data)
        b2 = _np.sum(my * mxd)
        a = _np.array([[a11, a12], [a12, a22]])
        b = _np.array([b1, b2])
        v = _np.linalg.solve(a, b)
        angle = _np.arctan(v[1]) * 180 / _np.pi
        angle *= -1  # sign change due to the dir of vertical pixel increase
        return angle

    def calc_mode_sigmas(self):
        """."""
        # method:
        #
        # [x, y] = R(-angle) [u1, u2]
        #
        # R(-angle) = [[C, S], [-S, C]]
        #
        # sigmax² = C² sigma1² + S² sigma2²
        # sigmay² = S² sigma1² + C² sigma2²
        angle = -self.angle * _np.pi / 180
        func, funs = _np.cos(angle), _np.sin(angle)
        det = func**2 - funs**2
        if abs(det) < 1e-6:
            return _np.nan, _np.nan

        sigmax = self.fitx.roi_sigma
        sigmay = self.fity.roi_sigma
        sigma1sqr = 1/det * (func**2 * sigmax**2 - funs**2 * sigmay**2)
        sigma2sqr = 1/det * (-funs**2 * sigmax**2 + func**2 * sigmay**2)
        sigma1 = _np.sqrt(sigma1sqr) if sigma1sqr > 0 else _np.nan
        sigma2 = _np.sqrt(sigma2sqr) if sigma2sqr > 0 else _np.nan
        return sigma1, sigma2

    def imshow(
            self, fig=None, axes=None,
            cropx = None, cropy = None,
            color_ellip=None, color_roi=None, color_axes=None):
        """."""
        return Image2D_ROI.imshow_images(
            self.data, self.fitx, self.fity, self.fitx.roi, self.fity.roi,
            angle=self.angle, fig=fig, axes=axes, cropx = cropx, cropy = cropy,
            color_ellip=color_ellip, color_roi=color_roi,
            color_axes=color_axes)

    def plot_projections(
            self, fig=None, axes=None):
        """."""
        if None in (fig, axes):
            fig, axes = _plt.subplots()

        colorx, colory = [0, 0, 0.7], [0.7, 0, 0]

        axes.plot(
            self.fitx.roi_indcs, self.fitx.roi_proj,
            color=colorx, alpha=1.0,
            lw=5, label='roix_proj')
        vecy, vecx = self.fitx.roi_fit
        if vecy is not None:
            axes.plot(
                vecx, vecy, color=[0.5, 0.5, 1], alpha=1.0,
                lw=2, label='roix_fit')

        axes.plot(
            self.fity.roi_indcs, self.fity.roi_proj,
            color=colory, alpha=1.0, lw=5,
            label='roiy_proj')
        vecy, vecx = self.fity.roi_fit
        if vecy is not None:
            axes.plot(
                vecx, vecy, color=[1, 0.5, 0.5], alpha=1.0,
                lw=2, label='roiy_fit')

        axes.legend()
        axes.grid()
        axes.set_ylabel('ROI pixel indices')
        axes.set_ylabel('Projection Intensity')

        return fig, axes

    def __str__(self):
        """."""
        res = super().__str__()
        res += '\n--- fitx ---\n'
        res += self.fitx.__str__()
        res += f'\nroi_amplitude   : {self.fitx.roi_amplitude}'
        res += f'\nroi_mean        : {self.fitx.roi_mean}'
        res += f'\nroi_sigma       : {self.fitx.roi_sigma}'
        res += f'\nroi_fit_err     : {self.fitx.roi_fit_error} %'
        res += '\n--- fity ---\n'
        res += self.fity.__str__()
        res += f'\nroi_amplitude   : {self.fity.roi_amplitude}'
        res += f'\nroi_mean        : {self.fity.roi_mean}'
        res += f'\nroi_sigma       : {self.fity.roi_sigma}'
        res += f'\nroi_fit_err     : {self.fity.roi_fit_error} %'
        res += '\n--- bigauss ---'
        res += f'\nangle           : {self.angle} deg.'

        return res

    @staticmethod
    def calc_roi_with_fwhm(image, fwhmx_factor, fwhmy_factor):
        """."""
        roix = Image1D_Fit.calc_roi_with_fwhm(
            image.fitx, fwhm_factor=fwhmx_factor)
        roiy = Image1D_Fit.calc_roi_with_fwhm(
            image.fity, fwhm_factor=fwhmy_factor)
        return roix, roiy

    def _update_image_fit(self, roix=None, roiy=None):
        """."""
        # fit projections
        roix, roiy = Image2D.update_roi(self.data, roix, roiy)
        data = self.project_image(self._data, 0)

        self._fity = Image1D_Fit(
            data=data, roi=roiy, fitgauss=self._fitgauss)
        self._fity.set_saturation_flag(self.is_saturated)
        data = self.project_image(self._data, 1)
        self._fitx = Image1D_Fit(
            data=data, roi=roix, fitgauss=self._fitgauss)
        self._fitx.set_saturation_flag(self.is_saturated)

        # fit angle
        angle, self._sigma1, self._sigma2 = self.calc_angle_normal_sigmas()
        if self.use_svd4theta:
            self._angle = angle
        else:
            self._angle = self.calc_angle_with_roi()
