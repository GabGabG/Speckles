import imageio as imio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.ndimage import gaussian_filter, median_filter
from typing import Tuple, Callable
from scipy.signal import convolve2d
import scipy.stats as stats


class SpeckleImageReader:
    """
    Class used to read image files. The file can have multiple frames. Will most likely be moved somewhere else.
    """

    def __init__(self, filepath: str):
        """
        Initializer of the class.
        :param filepath: str. Path leading to the file.
        """
        self._filepath = filepath

    def read(self):
        """
        Method used to read / load the images in memory. Without calling this method, nothing is loaded.
        :return: A NumPy array of the images as a stack with shape (N,M,S) where N is the height of the images, M is
        the width and S is the number of images. If only one image / frame is present, the shape is (N,M,1).
        """
        im = imio.mimread(self._filepath)
        if len(im) == 1:
            return im[0]
        return np.dstack(im)


class SpeckleImageManipulations:
    """
    Class used to manipulate a single speckle image.
    """

    def __init__(self, image_path: str = None, image_from_array: np.ndarray = None, image_name: str = None):
        """
        Initializer of the class.
        :param image_path: str. Path leading to the image. Cannot be used with `image_from_array`.
        :param image_from_array: np.ndarray. NumPy array containing the image's pixels. Cannot be used with
        `image_path`.
        :param image_name: str. Name of the image. Is `None` (no name) by default.
        """
        c1 = image_path is None and image_from_array is None
        c2 = image_path is not None and image_from_array is not None
        if c1 or c2:
            raise ValueError("Please give either the image path or the image (as an array), not both.")
        self._original_image = None
        self._image_path = image_path
        self._image_name = image_path
        if image_name is not None:
            self._image_name = image_name
        if image_path is not None:
            self._original_image = SpeckleImageReader(image_path).read()
        if image_from_array is not None:
            self._original_image = image_from_array.copy()
        self._modified_image = self._original_image.copy()
        self._autocorrelation_obj = None

    @property
    def original_image(self):
        """
        Getter of the original unmodified image (copy). It returns a copy, because otherwise any modification by the
        user would translate to modification inside the object. This is an arbitrary choice, it may change.
        :return: A copy of the original unmodified image.
        """
        return self._original_image.copy()

    @property
    def modified_image(self):
        """
        Getter of the modified image (copy). It returns a copy, because otherwise any modification by the
        user would translate to modification inside the object. This is an arbitrary choice, it may change.
        :return: A copy of the modified image.
        """
        return self._modified_image.copy()

    @property
    def autocorrelation(self):
        """
        Getter of the autocorrelation of the image as a copy. We really don't want this to be modified.
        :return: A copy of the autocorrelation of the image.
        """
        if self._autocorrelation_obj is None:
            return None
        return self._autocorrelation_obj.autocorrelation

    @property
    def image_path(self):
        """
        Path leading to the original image.
        :return: The path leading to the image file. Can be `None`.
        """
        return self._image_path

    @property
    def image_name(self):
        """
        Getter of the name of the image (if provided).
        :return: The name of the image. Can be `None`.
        """
        return self._image_name

    @image_name.setter
    def image_name(self, new_name: str):
        """
        Setter of the name of the image.
        :param new_name: New name for the image.
        :return: Nothing.
        """
        self._image_name = new_name

    def restore_original_image(self):
        """
        Method used to restore the original image as the current one we want to work on. Replaces the internally
        modified image with the original.
        :return: Nothing.
        """
        self._modified_image = self._original_image.copy()

    def crop(self, x_coords: Tuple[int, int] = (None, None), y_coords: Tuple[int, int] = (None, None)):
        """
        Method used to crop the current modified image (image worked on). Cropping can be horizontal and vertical. For
        example:
        ```python
        image.crop((10, -10), (10, -10))
        ```
        will remove the first and last pixels horizontally and vertically (we keep from 10 to width - 10 and
        height - 10).
        :param x_coords: Tuple of integers. Bounds of indices of pixels to keep horizontally (first is 0). Upper bound
        is excluded.
        :param y_coords: Tuple of integers. Bounds of indices of pixels to keep vertically (first is 0). Upper bound is
        excluded.
        :return: Nothing.
        """
        self._modified_image = self._modified_image[slice(*x_coords), slice(*y_coords)]

    def centered_crop(self, width: int, height: int):
        """
        Method used to crop around the center of the image.
        :param width: int. New width of the image (second dimension).
        :param height: int. New height of the image (first dimension).
        :return: Nothing.
        """
        half_width = width / 2
        half_height = height / 2
        shape = self._modified_image.shape
        x_start = shape[1] // 2 - np.ceil(half_width)
        x_end = shape[1] // 2 + np.floor(half_width)
        y_start = shape[0] // 2 - np.ceil(half_height)
        y_end = shape[0] // 2 + np.floor(half_height)
        self.crop((x_start, x_end), (y_start, y_end))

    def remove_background(self, background_image_path: str = None, background_image_as_array: np.ndarray = None,
                          clip: bool = False):
        """
        Method used to remove the background of the image. We simply subtract pixel by pixel the currently modified
        image by the background image. The shape must then be the same.
        :param background_image_path: str. Path leading to the image. Cannot be used with `background_image_as_array`.
        :param background_image_as_array: np.ndarray. NumPy array containing the pixel values. Cannot be used with
        `background_image_path`.
        :param clip: bool. Boolean specifying if we clip the values between [0, max(image)] if there are any negative
        values. Default is `False`, which translate the image by the minimum (if negative), meaning there is no
        clipping.
        :return: Nothing.
        """
        c1 = background_image_path is None and background_image_as_array is None
        c2 = background_image_path is not None and background_image_as_array is not None
        if c1 or c2:
            raise ValueError("Please give either the image path or the image (as an array), not both.")
        b_image = None
        if background_image_path is not None:
            b_image = SpeckleImageReader(background_image_path).read()
        if background_image_as_array is not None:
            b_image = background_image_as_array
        if b_image.shape != self._modified_image.shape:
            raise ValueError("The shape of the background image must match the current modified image's shape.")
        removed_bkg = self._modified_image - b_image
        if clip:
            self._modified_image = np.clip(removed_bkg, 0, None)
        else:
            minimum = np.min(removed_bkg)
            self._modified_image = removed_bkg
            if minimum < 0:
                self._modified_image -= minimum

    def apply_gaussian_normalization(self, filter_std_dev: float = 0):
        """
        Method used to normalize the image when the illumination is not uniform and follows a gaussian profile. This is
        perhaps not the best way to normalize as it can affect the contrast and the speckle size, but it can be good
        when we want to uniformize the intensity. A better way would be to normalize with a uniform image taken without
        speckles.
        :param filter_std_dev: float. Standard deviation of the gaussian filter. Should be positive. Default is 0, which
        means no normalization.
        :return: Nothing.
        """
        if filter_std_dev < 0:
            msg = "The gaussian filter's standard deviation must be positive (or 0, which means no normalization)."
            raise ValueError(msg)
        if filter_std_dev > 0:
            filtered_image = gaussian_filter(self._modified_image, filter_std_dev)
            self._modified_image = np.clip(self._modified_image / filtered_image - np.mean(self._modified_image), 0,
                                           None)

    def apply_median_filter(self, filter_size: int = 3):
        """
        Method to apply a median filter to remove small salt and pepper noise. The user should note that this filter
        affects the speckle size if the filter size is relatively large. Use with caution.
        :param filter_size: int. Size of the (square) filter. Should be positive and at least 2. However, the size can
        be 0, in which case no filtering is done.
        :return: Nothing.
        """
        if filter_size == 0:
            return
        if filter_size < 2:
            raise ValueError("The size of the median filter must be at least 2 (or 0, which means no filtering).")
        self._modified_image = median_filter(self._modified_image, filter_size)

    def apply_on_modified_image(self, func: Callable, *fargs, **fkwargs):
        """
        Method used to apply a certain function or filter on the images. It should be a function accepting a 2D array.
        Note that the shape of the image is (N,M), where N is the height of the images, M is the width. The function
        can do whatever it needs, but should return the image in the same shape.
        :param func: callable. Function to use with the image. Should accept a 2D array.
        :param fargs: args. Arguments to give to the function.
        :param fkwargs: keyword args. Keyword arguments to give to the function.
        :return: Nothing.
        """
        self._modified_image = func(self._modified_image, *fargs, **fkwargs)

    def compute_global_contrast(self):
        """
        Method used to compute the global contrast of the image. It is defined as the standard deviation of the image
        divided by the average of the image. For fully developed speckles, this quantity should be around 1.
        :return: The global contrast.
        """
        return np.std(self._modified_image) / np.mean(self._modified_image)

    def compute_michelson_contrast(self):
        """
        Method used to compute the Michelson contrast of the image. It is defined as (max - min) / (max + min).
        :return: The global Michelson contrast.
        """
        min_, max_ = np.min(self._modified_image), np.max(self._modified_image)
        top = max_ - min_
        bottom = max_ + min_
        return top / bottom

    def compute_local_constrast(self, kernel_size: int = 7):
        """
        Method used to compute the local contrast of the speckle image. The contrast is computed by rolling windows
        computing the contrast in 3 step: compute the average, then the average of the square of the image, then
        compute the standard deviation and divide it by the average (definition of contrast).
        :param kernel_size: int. Size of the (square) kernel. Should be at least 2. The default is 7 (a 7x7 window),
        which is often use in the literature (see David A. Boas, Andrew K. Dunn, "Laser speckle contrast imaging in
        biomedical optics," J. Biomed. Opt. 15(1) 011109 (1 January 2010) https://doi.org/10.1117/1.3285504)
        :return: A NumPy array (2D) of shape (N',M') where N' is the height of the images after passing in the kernel
        and M' is the width of the images after passing in the kernel.
        """
        if kernel_size < 2:
            raise ValueError("The size of the local contrast kernel must be at least 2.")
        kernel = np.ones((kernel_size, kernel_size))
        n = kernel.size
        temp_image = self._modified_image.astype(float)
        windowed_avg = convolve2d(temp_image, kernel, "valid") / n
        squared_image_filter = convolve2d(temp_image ** 2, kernel, "valid")
        std_image_windowed = ((squared_image_filter - n * windowed_avg ** 2) / (n - 1)) ** 0.5
        return std_image_windowed / windowed_avg

    def do_autocorrelation(self):
        """
        Method used to do the autocorrelation of the image (i.e. the cross correlation of the image with itself).
        :return: Nothing.
        """
        self._autocorrelation_obj = AutocorrelationUtils(self._modified_image)
        self._autocorrelation_obj.autocorrelate()

    def access_autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
        """
        Method used to access autocorrelation slices (i.e. a 1D view of the 2D autocorrelation).
        :param slices_pos: tuple of integers. The first integer (must be positive) is the vertical position (i.e. the
        line number of the array, which starts at 0). It is `None` by default which gives the central slice (i.e. the
        slice at shape[0] // 2). The second integer (must also be positive) is the horizontal position (i.e. the column
        number of the array, which also starts at 0). It is also `None` by default, which gives the central slice (i.e.
        the slice at shape[1] // 2).
        :return: A tuple of 1D arrays. The first array is the horizontal slice (i.e. the slice at the position of the
        first element of `slice_pos`), while the second array is the vertical slice (i.e. the slice at the position of
        the second element of `slice_pos`).
        """
        if self._autocorrelation_obj is None:
            raise ValueError("Please do the autocorrelation before accessing its slices.")
        return self._autocorrelation_obj.autocorrelation_slices(slices_pos)

    def get_speckle_sizes(self):
        """
        Method used to get the speckle sizes (the horizontal and vertical sizes). See `PeakMeasurementsUtils.find_FWHMs`
        for more info about what is done.
        :return: A tuple of 2 floats. The first element is the horizontal speckle size while the second is the vertial
        speckle size.
        """
        if self._autocorrelation_obj is None:
            raise ValueError("Please do the autocorrelation before finding the (average) speckle sizes.")
        h_slice, v_slice = self.access_autocorrelation_slices()
        data_x_h = np.arange(len(h_slice))
        h_speckle_size = PeakMeasurementUtils(data_x_h, h_slice).find_FWHM()
        data_x_v = np.arange(len(v_slice))
        v_speckle_size = PeakMeasurementUtils(data_x_v, v_slice).find_FWHM()
        return h_speckle_size, v_speckle_size

    # def speckle_developedness(self, n_bins_histogram: int = 256, show_fits: bool = True):
    #     data = np.ravel(self._modified_image)
    #     exponential_fit_args = stats.expon.fit(data, floc=0)
    #     gamma_fit_args = stats.gamma.fit(data, floc=0)
    #     n_data, bins, _ = plt.hist(data, n_bins_histogram, None, True, label="Speckle intensity")
    #     x_data = (bins[:-1] + bins[1:]) / 2
    #     res_kstest_exponential = stats.ks_1samp(data, stats.expon.cdf, args=exponential_fit_args)
    #     res_kstest_gamma = stats.ks_1samp(data, stats.gamma.cdf, args=gamma_fit_args)
    #     plt.plot(x_data, stats.expon.pdf(x_data, *exponential_fit_args), color="green", linestyle="--",
    #              label=f"Exponential fit\nParameters : {exponential_fit_args}")
    #     plt.plot(x_data, stats.gamma.pdf(x_data, *gamma_fit_args), color="red", linestyle=":",
    #              label=f"Gamma fit\nParameters : {gamma_fit_args}")
    #     plt.legend()
    #     plt.xlabel("Intensity value [-]")
    #     plt.ylabel("Frequency [-]")
    #     if show_fits:
    #         plt.show()
    #     else:
    #         plt.clf()
    #     return {"Exponential fit": dict(zip(["Test value", "P-value"], res_kstest_exponential)),
    #             "Gamma fit": dict(zip(["Test value", "P-value"], res_kstest_gamma))}


class AutocorrelationUtils:
    """
    Class used to do autocorrelations and other related work.
    """

    def __init__(self, speckle_image: np.ndarray):
        """
        Initializer of the class.
        :param speckle_image: np.ndarray. NumPy array containing the image. Note that the shape must be (N,M) where N
        is the height of the images and M is the width. If the shape is different, the autocorrelation algorithm won't
        work properly.
        """
        self._speckle_image = speckle_image
        self._autocorrelation = None

    @property
    def autocorrelation(self):
        """
        Getter of the autocorrelation. It returns a copy of the array, because we don't want any unwanted
        modifications.
        :return: The autocorrelation (as a copy).
        """
        if self._autocorrelation is None:
            return None
        return self._autocorrelation.copy()

    def autocorrelate(self):
        """
        Method used to do the autocorrelation (cross correlation of the image with itself). The algorithm uses the
        Wiener–Khinchin theorem (https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem) which states that the
        autocorrelation of a process (stationary) can be computed with the power spectrum of the process. This allows
        for fast computations using Fourier transforms.
        :return: Nothing.
        """
        fft = np.fft.fft2(self._speckle_image)
        ifft = np.fft.ifftshift(np.fft.ifft2(np.abs(fft) ** 2)).real
        ifft /= np.size(ifft)
        self._autocorrelation = (ifft - np.mean(self._speckle_image) ** 2) / np.var(self._speckle_image)

    def autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
        """
        Method used to access autocorrelation slices (i.e. a 1D view of the 3D autocorrelations).
        :param slices_pos: tuple of integers. The first integer (must be positive) is the vertical position (i.e. the
        line number of the array, which starts at 0). It is `None` by default which gives the central slice (i.e. the
        slice at shape[0] // 2). The second integer (must also be positive) is the horizontal position (i.e. the column
        number of the array, which also starts at 0). It is also `None` by default, which gives the central slice (i.e.
        the slice at shape[1] // 2).
        :return: A tuple of 1D arrays. The first array is the horizontal slice (i.e. the slice at the position of the
        first element of `slice_pos`), while the second array is the vertical slice (i.e. the slice at the position of
        the second element of `slice_pos`).
        """
        if self._autocorrelation is None:
            raise ValueError("Please do the autocorrelation before accessing its slices.")
        v_pos, h_pos = slices_pos
        if v_pos is None:
            v_pos = self._autocorrelation.shape[0] // 2
        if h_pos is None:
            h_pos = self._autocorrelation.shape[1] // 2
        h_slice = self._autocorrelation[v_pos, :]
        v_slice = self._autocorrelation[:, h_pos]
        return h_slice, v_slice


class PeakMeasurementUtils:
    """
    Class used to compute FWHM of a peaks in a 2D signal (i.e. 1D signal stacked).
    """

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):
        """
        Initializer of the class.
        :param data_x: np.ndarray. NumPy 1D array of x values of the signal. Should be ordered.
        :param data_y: np.ndarray. NumPy 1D array of y values of the signal. Should be ordered with the right x
        values and should have the same shape.
        """
        if data_x.ndim != 1:
            raise ValueError("`data_x` must be a 1-dimensional array.")
        if data_y.ndim != 1:
            raise ValueError("`data_y` must be a 1-dimensional array.")
        if data_y.shape != data_x.shape:
            raise ValueError("`data_x` must have the same shape as `data_y`.")
        self._data_x = data_x.copy()
        self._data_y = data_y.copy()
        self._interpolated_data = interp1d(self._data_x, self._data_y, "cubic", assume_sorted=True)

    def find_FWHM(self, maximum: float = None):
        """
        Method used to compute the FWHM of the peak in the internal data (specified in the initializer). The data is
        interpolated, then the roots where the signal equals the half maximum are computed and subtracted to find the
        full width at half maximum.
        :param maximum: float Supposed maximums of the peak. The default is `None`, which means that the maximum is
        inferred from the data.
        :return: The distance between the right root and the left root.
        """
        min_x = np.min(self._data_x)
        max_x = np.max(self._data_x)
        other_bound = self._data_x[np.argmax(self._data_y)]
        maximum = np.max(self._data_y) if maximum is None else maximum
        half_max = maximum / 2

        def func(inner_func_x):
            return self._interpolated_data(inner_func_x) - half_max

        left_root = root_scalar(func, bracket=(min_x, other_bound)).root
        right_root = root_scalar(func, bracket=(other_bound, max_x)).root
        return right_root - left_root


if __name__ == '__main__':
    plt.rcParams.update({"font.size": 36})
    from scipy.stats import ks_1samp, anderson, expon, gamma

    path = r"../test_speckle_size_ellipsoid.tiff"
    manip = SpeckleImageManipulations(path)
    im = manip.modified_image
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(im, cmap="gray")
    axes[0, 0].axis("off")
    manip.do_autocorrelation()
    autocorr = manip.autocorrelation
    axes[0, 1].imshow(autocorr)
    axes[0, 1].axis("off")
    slices = manip.access_autocorrelation_slices()
    axes[1, 0].plot(slices[0], linewidth=10)
    axes[1, 0].set_xlabel("Width [px]")
    axes[1, 0].set_ylabel("Autocorrelation")
    axes[1, 1].plot(slices[1], linewidth=10)
    axes[1, 1].set_xlabel("Height [px]")
    axes[1, 1].set_ylabel("Autocorrelation")
    plt.show()
    print(manip.get_speckle_sizes())
    exit()
    path = r"C:\Users\goubi\Desktop\NormalizedSpecklePattern.tiff"
    path2 = r"C:\Users\goubi\Desktop\SpeckleTest (1).tiff"
    image = imio.imread(path)  # / 255
    image2 = imio.imread(path2) / 255
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(image.ravel(), 256, None, True)
    ax2.hist(image2.ravel(), 256, None, True)
    plt.show()
    print(image.shape)

    vals, x_bins, _ = plt.hist(image.ravel(), 256, None, True, label="Histogram")
    plt.show()
    fit_args_expon = expon.fit(image.ravel(), floc=0)
    loc = np.argmax(vals)
    keep_vals = image.ravel()
    keep_vals = keep_vals[keep_vals >= x_bins[loc]]
    plt.hist(keep_vals, 256, None, True)
    plt.show()
    fit_args_expon_pas_force = expon.fit(image.ravel())
    fit_args_expon_remove = expon.fit(keep_vals)
    # fit_args_gamma = gamma.fit(image.ravel())

    print(fit_args_expon_remove)
    print(fit_args_expon_pas_force)
    # print(fit_args_gamma)
    vals, x_data, _ = plt.hist(image.ravel(), 256, None, True, label="Histogram")
    x = (x_data[:-1] + x_data[1:]) / 2
    plt.plot(x, expon.pdf(x, *fit_args_expon_remove), linestyle=":",
             label="Exponential distribution (remove before peak)")
    plt.plot(x, expon.pdf(x, *fit_args_expon_pas_force), linestyle="--",
             label="Exponential distribution (force loc = None)")
    # plt.plot(x, gamma.pdf(x, *fit_args_gamma), label="Gamma distribution (force loc = 0)")
    plt.legend()
    plt.show()
    exit()


    # from scipy.special import gamma

    # path = r"../SpeckleSimulations/test.tif"
    # sp = SpeckleImageManipulations(path)
    # sp.do_autocorrelation()
    # image = sp.modified_image
    # plt.imshow(image, cmap="gray")
    # plt.show()
    # autocrr = sp.autocorrelation
    # plt.imshow(autocrr)
    # plt.show()
    # print(sp.get_speckle_sizes())  # Should be around 10 and 4

    def rebin(arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return (arr.reshape(shape).mean(-1).mean(1))  # / n_bins


    def speckling(mask: np.ndarray):
        phases = np.random.uniform(-np.pi, np.pi, mask.shape)
        phasors = np.exp(1j * phases)
        speckles = (np.abs(np.fft.fftshift(np.fft.fft2(phasors * mask))) ** 2).real
        speckles /= np.max(speckles)
        return speckles


    def p_I_integration(intensite_integree, moyenne_intensite_pas_integree, n_bins):
        return gamma.pdf(intensite_integree, a=1 - 1 / n_bins, scale=moyenne_intensite_pas_integree, loc=0)


    shape = 1000
    shape_mat = (shape, shape)
    desired_size = 10
    r = shape / (2 * desired_size)
    Y, X = np.indices(shape_mat)
    Y -= shape // 2
    X -= shape // 2
    mask = (X ** 2 + Y ** 2 - r ** 2) <= 0
    speckles = speckling(mask)
    taille_speckles = shape / (2 * r)
    speckles_1d = speckles.ravel()
    sp = SpeckleImageManipulations(None, speckles)
    sp.do_autocorrelation()
    print(sp.get_speckle_sizes())
    for nbins in [1000, 500, 250, 100, 50, 25, 10, 4, 1]:
        print(f"=========={nbins}==========")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(speckles, cmap="gray")
        nb_bins = nbins
        binned_speckles = rebin(speckles, (nb_bins, nb_bins))
        sp = SpeckleImageManipulations(None, binned_speckles)
        sp.do_autocorrelation()

        ax2.imshow(binned_speckles, cmap="gray")
        try:
            print(sp.get_speckle_sizes())
        except Exception:
            print("Speckle size cannot be computed")
        intensity_base = speckles.ravel()
        intensity_binned = binned_speckles.ravel()
        ax3.hist(intensity_base, int(len(intensity_base) ** 0.5), density=True,
                 label="Sans binning", histtype="step", linestyle="--")
        data, x_bins, _ = ax3.hist(intensity_binned, int(len(intensity_binned) ** 0.5), density=True, histtype="step",
                                   linestyle=":", label="Avec binning")
        x = (x_bins[:-1] + x_bins[1:]) / 2


        def exponential_fit(data):
            return expon.fit(data, floc=0)


        def gamma_fit(data):
            return gamma.fit(data, floc=0)


        def gamma_cdf(x, n, loc, theta):
            return gamma.cdf(x, n, scale=theta, loc=loc)


        def exponential_cdf(x, loc, scale):
            return expon.cdf(x, scale=scale, loc=loc)


        gamma_fit_arg = gamma_fit(intensity_binned)
        m = np.mean(intensity_binned) ** 2 / np.var(intensity_binned)
        print(m)
        print(np.mean(intensity_binned) / m)
        print(gamma_fit_arg)
        if not any(np.isnan(gamma_fit_arg)):
            ax3.plot(x, gamma.pdf(x, *gamma_fit_arg), color="green", label="Gamma fit on binned data", alpha=0.75)

        plt.legend()
        ax1.set_title("Original image 1000 x 1000")
        ax2.set_title(f"Binned image {nb_bins} x {nb_bins}")
        ax3.set_title("Intensity histograms")
        ax3.set_xlabel("Intensité [-]")
        ax3.set_ylabel(r"$P_I(I)$ [-]")
        plt.show()
    exit()
    gamma_args_base = gamma_fit(intensity_base)
    expon_args_base = exponential_fit(intensity_base)
    gamma_args_binned = gamma_fit(intensity_binned)
    expon_args_binned = exponential_fit(intensity_binned)
    k_s_gamma_base = ks_1samp(intensity_base, gamma_cdf, gamma_args_base)
    k_s_gamma_binned = ks_1samp(intensity_binned, gamma_cdf, gamma_args_binned)
    k_s_expon_base = ks_1samp(intensity_base, exponential_cdf, expon_args_base)
    k_s_expon_binned = ks_1samp(intensity_binned, exponential_cdf, expon_args_binned)
    a_d_expon_base = anderson(intensity_base, "expon")
    a_d_expon_binned = anderson(intensity_binned, "expon")
    print(k_s_gamma_base)
    print(k_s_gamma_binned)
    print(k_s_expon_base)
    print(k_s_expon_binned)
    print(a_d_expon_base)
    print(a_d_expon_binned)
