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
        return self._original_image.copy()

    @property
    def modified_image(self):
        return self._modified_image.copy()

    @property
    def autocorrelation(self):
        if self._autocorrelation_obj is None:
            return None
        return self._autocorrelation_obj.autocorrelation

    @property
    def image_path(self):
        return self._image_path

    @property
    def image_name(self):
        return self._image_name

    @image_name.setter
    def image_name(self, new_name: str):
        self._image_name = new_name

    def restore_original_image(self):
        self._modified_image = self._original_image.copy()

    def crop(self, x_coords: Tuple[int, int] = (None, None), y_coords: Tuple[int, int] = (None, None)):
        """
        Applied on modified image
        :param x_coords:
        :param y_coords:
        :return:
        """
        self._modified_image = self._modified_image[slice(*x_coords), slice(*y_coords)]

    def centered_crop(self, width: int, height: int):
        half_width = width / 2
        half_height = height / 2
        shape = self._modified_image.shape
        x_start = shape[1] // 2 - np.ceil(half_width)
        x_end = shape[1] // 2 + np.floor(half_width)
        y_start = shape[0] // 2 - np.ceil(half_height)
        y_end = shape[0] // 2 + np.floor(half_height)
        self.crop((x_start, x_end), (y_start, y_end))

    def remove_background(self, background_image_path: str = None, background_image_as_array: np.ndarray = None):
        """
        Applied on modified image
        :param background_image_path:
        :param background_image_as_array:
        :return:
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
        self._modified_image = np.clip(self._modified_image - b_image, 0, None)

    def apply_gaussian_normalization(self, filter_std_dev: float = 0):
        """
        Applied on modified image
        :param filter_std_dev:
        :return:
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
        Applied on modified image
        :param filter_size:
        :return:
        """
        if filter_size == 0:
            return
        if filter_size < 2:
            raise ValueError("The size of the median filter must be at least 2 (or 0, which means no filtering).")
        self._modified_image = median_filter(self._modified_image, filter_size)

    def apply_on_modified_image(self, func: Callable, *fargs, **fkwargs):
        """
        Applied on modified image
        :param func:
        :param fargs:
        :param fkwargs:
        :return:
        """
        self._modified_image = func(self._modified_image, *fargs, **fkwargs)

    def compute_global_contrast(self):
        return np.std(self._modified_image) / np.mean(self._modified_image)

    def compute_michelson_contrast(self):
        min_, max_ = np.min(self._modified_image), np.max(self._modified_image)
        top = max_ - min_
        bottom = max_ + min_
        return top / bottom

    def compute_local_constrast(self, kernel_size: int = 7):
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
        self._autocorrelation_obj = AutocorrelationUtils(self._modified_image)
        self._autocorrelation_obj.autocorrelate()

    def access_autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
        if self._autocorrelation_obj is None:
            raise ValueError("Please do the autocorrelation before accessing its slices.")
        return self._autocorrelation_obj.autocorrelation_slices(slices_pos)

    def get_speckle_sizes(self):
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

    def __init__(self, speckle_image: np.ndarray):
        self._speckle_image = speckle_image
        self._autocorrelation = None

    @property
    def autocorrelation(self):
        if self._autocorrelation is None:
            return None
        return self._autocorrelation.copy()

    def autocorrelate(self):
        fft = np.fft.fft2(self._speckle_image)
        ifft = np.fft.ifftshift(np.fft.ifft2(np.abs(fft) ** 2)).real
        ifft /= np.size(ifft)
        self._autocorrelation = (ifft - np.mean(self._speckle_image) ** 2) / np.var(self._speckle_image)

    def autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
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

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):
        if data_x.ndim != 1:
            raise ValueError("`data_x` must be a 1-dimensional array.")
        if data_y.ndim != 1:
            raise ValueError("`data_y` must be a 1-dimensional array.")
        self._data_x = data_x.copy()
        self._data_y = data_y.copy()
        self._interpolated_data = interp1d(self._data_x, self._data_y, "cubic", assume_sorted=True)

    def find_FWHM(self, maximum: float = None):
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