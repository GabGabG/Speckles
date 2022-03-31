import imageio as imio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.ndimage import gaussian_filter, median_filter
from typing import Tuple, Callable


class SpeckleImageReader:

    def __init__(self, filepath: str):
        self._filepath = filepath

    def read(self):
        im = imio.mimread(self._filepath)
        if len(im) == 1:
            return im[0]
        return im


class SpeckleImageManipulations:

    def __init__(self, image_path: str = None, image_from_array: np.ndarray = None, image_name: str = None):
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

    @property
    def original_image(self):
        return self._original_image.copy()

    @property
    def modified_image(self):
        return self._modified_image.copy()

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

    def remove_background(self, background_image_path: str = None, background_image_as_array: np.ndarray = None):
        """
        Applied on modified image
        :param background_image_path:
        :param background_image_as_array:
        :return:
        """
        pass

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
            self._modified_image = self._modified_image / filtered_image - np.mean(self._modified_image)

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

    def do_autocorrelation(self):
        pass


class AutocorrelationUtils:

    def __init__(self, speckle_image: np.ndarray):
        self._speckle_image = speckle_image
        self._autocorrelation = None

    def autocorrelate(self):
        fft = np.fft.fft2(self._speckle_image)
        ifft = np.fft.ifftshift(np.fft.ifft2(np.abs(fft) ** 2)).real
        ifft /= np.size(ifft)
        self._autocorrelation = (ifft - np.mean(self._speckle_image) ** 2) / np.var(self._speckle_image)

    def autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
        pass

    def show_autocorrelation(self, with_color_bar: bool = True):
        pass

    def show_autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None), show_horizontal: bool = True,
                                    show_vertical: bool = True):
        pass


class PeakMeasurementUtils:

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):
        if data_x.ndim != 1:
            raise ValueError("`data_x` must be a 1-dimensional array.")
        if data_y.ndim != 1:
            raise ValueError("`data_y` must be a 1-dimensional array.")
        self._data_x = data_x.copy()
        self._data_y = data_y.copy()
        self._interpolated_data = None

    def interpolate_data(self):
        self._interpolated_data = interp1d(self._data_x, self._data_y, 'cubic', assume_sorted=True)

    def show_data(self, with_interpolation: bool = True):
        if with_interpolation and self._interpolated_data is None:
            raise ValueError("Please interpolate the data before showing the interpolation.")
        plt.plot(self._data_x, self._data_y, label="Data points")
        if with_interpolation:
            plt.plot(self._data_x, self._interpolated_data(self._data_x), label="Interpolation", linestyle="--")
        plt.legend()
        plt.show()

    def find_FWHM(self, assume_maximum_is_1: bool = True):
        if self._interpolated_data is None:
            raise ValueError("Please interpolate the data before finding FWHM")
        min_x = np.min(self._data_x)
        max_x = np.max(self._data_x)
        maximum = 1 if assume_maximum_is_1 else np.max(self._data_y)
        half_max = maximum / 2

        def func(x):
            return self._interpolated_data(x) - half_max

        left_root = root_scalar(func, bracket=(min_x, 0)).root
        right_root = root_scalar(func, bracket=(0, max_x)).root
        return right_root - left_root


if __name__ == '__main__':
    from scipy.special import sinc

    x = np.linspace(-10, 10, 1000)
    y = sinc(x)
    plt.plot(x, y, label="Data")

    interp = interp1d(x, y, "cubic", assume_sorted=True)
    plt.plot(x, interp(x), label="Interp", linestyle="--")
    f = lambda x: interp(x) - 0.5
    roots = root_scalar(f, bracket=(min(x), 0)), root_scalar(f, bracket=(0, max(x)))
    sols = roots[0].root, roots[1].root
    plt.scatter(sols, (0.5, 0.5))
    plt.plot([min(x), max(x)], [0.5, 0.5])
    plt.show()
    print(sols)
    print(f"FWHM = {sols[1] - sols[0]}")
