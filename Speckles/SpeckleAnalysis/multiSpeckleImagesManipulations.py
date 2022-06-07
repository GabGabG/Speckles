from speckleImageManipulations import SpeckleImageReader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

#TODO: Rework everything!

class SpeckleMovieReader:
    """
    Class used to read movie files. This should be somewhere else.
    """

    def __init__(self, filepath: str):
        """
        Initializer of the class.
        :param filepath: File path to the movie we want to read / load.
        """
        self._filepath = filepath

    def read(self):
        """
        Method used to read the specified file. Nothing is read otherwise.
        :return: A stack of frames. The shape is (N,M,S), where N is the height of the images, M is the width and S is
        the number of images.
        """
        video = cv2.VideoCapture(self._filepath)
        stack = []
        while True:
            ret, frame = video.read()
            if ret:
                stack.append(frame)
            else:
                break
        stack = np.dstack(stack)
        return stack


class MultiSpeckleImagesManipulations:
    """
    Class used to manipulate multiple speckle images at the same time. Maybe this should be merged with
    `SpeckleImageManipulations`? (the single image version)
    """

    def __init__(self, movie_path: str = None, images_path: str = None, images_from_array: np.ndarray = None,
                 images_name: str = None, stack_order: str = "dhw"):
        """
        Initializer of the class.
        :param movie_path: str. Path leading to the movie we want to read. Cannot be used with `images_path` and
        `images_from_array`.
        :param images_path: str. Path leading to the images file (like a Tiff with multiple pages). Cannot be used with
        `movie_path` and `images_from_array`.
        :param images_from_array: np.ndarray. NumPy array containing the images (pixels are the values). Cannot be used
        with `movie_path` and `images_path`.
        :param images_name: str. Name of the stack of images. TODO: should be a list of str?
        :param stack_order: str. Order of the images stack. Can be:
         - 'dhw' for depth, height, width (depth being the first dimension, the number of images) (default)
         - 'whd' for width, height, depth.
         - 'hwd' for height, width, depth. This is the format used internally. Everything else is transposed.
        Any other string raises an error.
        """
        c1 = movie_path is None and images_path is None and images_from_array is None
        c2 = movie_path is not None and images_path is not None
        c3 = movie_path is not None and images_from_array is not None
        c4 = images_path is not None and images_from_array is not None
        if c1 or c2 or c3 or c4:
            msg = "Please given either a movie path, an 'image' file containing the images or an array containing " \
                  "the images, not multiple combinations."
            raise ValueError(msg)
        self._original_images = None
        self._images_path = images_path if images_path is not None else movie_path
        self._images_name = images_path if images_path is not None else movie_path
        if images_name is not None:
            self._images_name = images_name
        if images_path is not None:
            self._original_images = SpeckleImageReader(images_path).read()
        if movie_path is not None:
            self._original_images = SpeckleMovieReader(movie_path).read()
        if images_from_array is not None:
            self._original_images = images_from_array
            if stack_order == "dhw":
                self._original_images = self._original_images.transpose((1, 2, 0))
            elif stack_order == "whd":
                self._original_images = self._original_images.transpose((1, 0, 2))
            elif stack_order != "hwd":
                raise ValueError(f"Stack order `{stack_order}` not recognized.")
        self._modified_images = self._original_images.copy()
        self._autocorrelations_obj = None

    @property
    def original_images(self):
        """
        Getter of the original unmodified images (copy). It returns a copy, because otherwise any modification by the
        user would translate to modification inside the object. This is an arbitrary choice, it may change.
        :return: A copy of the original unmodified images.
        """
        return self._original_images.copy()

    @property
    def modified_images(self):
        """
        Getter of the modified images (copy). It returns a copy, because otherwise any modification by the
        user would translate to modification inside the object. This is an arbitrary choice, it may change.
        :return: A copy of the modified images.
        """
        return self._modified_images.copy()

    @property
    def autocorrelations(self):
        """
        Getter of the autocorrelation stack of the images as a copy. We really don't want this to be modified.
        :return: A copy of the autocorrelation stack of the images.
        """
        if self._autocorrelations_obj is None:
            return None
        return self._autocorrelations_obj.autocorrelations

    @property
    def images_path(self):
        """
        Path leading to the original movie or images if one of them is not `None`. Otherwise, it returns `None`.
        :return: The path leading to the images (movie or images file). Can be `None`.
        """
        return self._images_path

    @property
    def images_name(self):
        """
        Getter of the name of the images (if provided).
        :return: The name of the images. Can be `None`.
        """
        return self._images_name

    @images_name.setter
    def images_name(self, new_name: str):
        """
        Setter of the name of the images.
        :param new_name: New name for the images.
        :return: Nothing.
        """
        self._images_name = new_name

    def restore_original_images(self):
        """
        Method used to restore the original images as the current ones we want to work on. Replaces any internally
        modified image with the original.
        :return: Nothing.
        """
        self._modified_images = self._original_images.copy()

    def crop(self, x_coords: Tuple[int, int] = (None, None), y_coords: Tuple[int, int] = (None, None),
             depth_coords: Tuple[int, int] = (None, None)):
        """
        Method used to crop the current modified images (images worked on). Cropping can be horizontal, vertical and
        in depth (removing images from the stack). For example:
        ```python
        images.crop((10, -10), (10, -10), (0, 5))
        ```
        will remove the first and last 10 pixels horizontally and vertically (we keep from 10 to width - 10 and
        height - 10), as well as only keeping the first 5 frames.
        :param x_coords: Tuple of integers. Bounds of indices of pixels to keep horizontally (first is 0). Upper bound
        is excluded.
        :param y_coords: Tuple of integers. Bounds of indices of pixels to keep vertically (first is 0). Upper bound is
        excluded.
        :param depth_coords: Tuple of integers. Bounds of indices of images to keep (first is 0). Upper bound is
        excluded.
        :return: Nothing.
        """
        self._modified_images = self._modified_images[slice(*y_coords), slice(*x_coords), slice(*depth_coords)]

    def centered_crop(self, width: int, height: int, depth: int):
        """
        Method used to crop around the center of the stack.
        :param width: int. New width of the stack (second dimension).
        :param height: int. New height of the stack (first dimension).
        :param depth: int. New depth of the stack (last dimension).
        :return: Nothing.
        """
        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2
        shape = self._modified_images.shape
        x_start = shape[1] // 2 - np.ceil(half_width)
        x_end = shape[1] // 2 + np.floor(half_width)
        y_start = shape[0] // 2 - np.ceil(half_height)
        y_end = shape[0] // 2 + np.floor(half_height)
        z_start = shape[-1] // 2 - np.ceil(half_depth)
        z_end = shape[-1] // 2 + np.floor(half_depth)
        self.crop((x_start, x_end), (y_start, y_end), (z_start, z_end))

    def remove_background(self):
        # TODO
        pass

    def apply_gaussian_normalization(self, filter_std_dev: float = 0):
        if filter_std_dev < 0:
            msg = "The gaussian filter's standard deviation must be positive (or 0, which means no normalization)."
            raise ValueError(msg)
        if filter_std_dev > 0:
            n_images = self._modified_images.shape[-1]
            filters = np.full_like(self._modified_images, np.nan)
            for i in range(n_images):
                filtered_image_i = gaussian_filter(self._modified_images[:, :, i], filter_std_dev)
                filters[:, :, i] = filtered_image_i
            self._modified_images = self._modified_images / filters - np.mean(self._modified_images)

    def apply_median_filter(self, filter_size: int = 0):
        if filter_size == 0:
            return
        if filter_size < 2:
            raise ValueError("The size of the median filter must be at least 2 (or 0, which means no filtering).")
        n_images = self._modified_images.shape[-1]
        median_filtered = np.full_like(self._modified_images, np.nan)
        for i in range(n_images):
            filtered_image_i = median_filter(self._modified_images[:, :, i], filter_size)
            median_filtered[:, :, i] = filtered_image_i
        self._modified_images = median_filtered

    def apply_on_modified_image(self, func: Callable, *fargs, force_on_slices: bool = False, **fkwargs):
        if force_on_slices:
            n_images = self._modified_images.shape[-1]
            modified_image = np.full_like(self._modified_images, np.nan)
            for i in range(n_images):
                modified_image_i = func(self._modified_images[:, :, i], *fargs, **fkwargs)
                modified_image[:, :, i] = modified_image_i
            self._modified_images = modified_image
        else:
            self._modified_images = func(self._modified_images, *fargs, **fkwargs)

    def compute_local_contrast(self, kernel_size: int = 7):
        if kernel_size < 2:
            raise ValueError("The size of the local contrast kernel must be at least 2.")
        n_images = self._modified_images.shape[-1]
        kernel = np.ones((kernel_size, kernel_size))
        n = kernel.size
        shape_y = np.abs(self._modified_images.shape[0] - kernel.shape[0]) + 1
        shape_x = np.abs(self._modified_images.shape[1] - kernel.shape[1]) + 1
        windowed_avgs = np.full_like((shape_y, shape_x, n_images), np.nan)
        squared_images_filtered = windowed_avgs.copy()
        temp_images = self._modified_images.astype(float)
        for i in range(n_images):
            windowed_avg = convolve2d(temp_images[:, :, i], kernel, "valid") / n
            squared_image_filtered = convolve2d(temp_images[:, :, i] ** 2, kernel, "valid")
            windowed_avgs[:, :, i] = windowed_avg
            squared_images_filtered[:, :, i] = squared_image_filtered
        std_image_windowed = ((squared_images_filtered - n * windowed_avgs ** 2) / (n - 1)) ** 0.5
        return std_image_windowed / windowed_avgs

    def do_autocorrelations(self):
        self._autocorrelations_obj = AutocorrelationsUtils(self._modified_images)
        self._autocorrelations_obj.autocorrelate()

    def access_autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
        if self._autocorrelations_obj is None:
            raise ValueError("Please do the autocorrelation before accessing its slices.")
        return self._autocorrelations_obj.autocorrelation_slices(slices_pos)

    def get_speckle_sizes(self, average: bool = True):
        if self._autocorrelations_obj is None:
            raise ValueError("Please do the autocorrelation before finding the (average) speckle sizes.")
        h_slices, v_slices = self.access_autocorrelation_slices()
        data_x_h = np.vstack([np.arange(len(h_slices[i])) for i in range(len(h_slices))])
        h_speckle_sizes = PeakMeasurementsUtils(data_x_h, h_slices).find_FWHMs()
        data_x_v = np.vstack([np.arange(len(v_slices[i])) for i in range(len(v_slices))])
        v_speckle_sizes = PeakMeasurementsUtils(data_x_v, v_slices).find_FWHMs()
        if average:
            return np.mean(h_speckle_sizes), np.mean(v_speckle_sizes)
        return h_speckle_sizes, v_speckle_sizes


class AutocorrelationsUtils:

    def __init__(self, speckle_images: np.ndarray):
        self._speckle_images = speckle_images
        self._autocorrelations = None

    @property
    def autocorrelations(self):
        if self._autocorrelations is None:
            return None
        return self._autocorrelations.copy()

    def autocorrelate(self):
        ffts = np.fft.fft2(self._speckle_images, axes=(0, 1))
        iffts = np.fft.ifftshift(np.fft.ifft2(np.abs(ffts) ** 2, axes=(0, 1)), axes=(0, 1)).real
        iffts /= np.size(iffts[:, :, 0])
        self._autocorrelations = (iffts - np.mean(self._speckle_images, axis=(0, 1)) ** 2) / np.var(
            self._speckle_images, axis=(0, 1))

    def autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
        if self._autocorrelations is None:
            raise ValueError("Please do the autocorrelation before accessing its slices.")
        v_pos, h_pos = slices_pos
        if v_pos is None:
            v_pos = self._autocorrelations.shape[0] // 2
        if h_pos is None:
            h_pos = self._autocorrelations.shape[1] // 2
        h_slices = self._autocorrelations[v_pos, :, :].T
        v_slices = self._autocorrelations[:, h_pos, :].T
        return h_slices, v_slices


class PeakMeasurementsUtils:

    def __init__(self, data_x_s: np.ndarray, data_y_s: np.ndarray):
        if data_x_s.ndim != 2:
            raise ValueError("`data_x_s` must be a 2-dimensional array.")
        if data_y_s.ndim != 2:
            raise ValueError("`data_y_s` must be a 2-dimensional array.")
        self._data_x_s = data_x_s.copy()
        self._data_y_s = data_y_s.copy()
        self._interpolations = [interp1d(data_x_s[0], data_y_s[0], "cubic", assume_sorted=True)]

    def find_FWHMs(self, maximums: Tuple[float] = None):
        mins_x = np.min(self._data_x_s, axis=1)
        maxs_x = np.max(self._data_x_s, axis=1)
        other_bounds = (maxs_x - mins_x) / 2
        maximums = np.max(self._data_y_s, axis=1) if maximums is None else np.array(maximums)
        half_maxs = maximums / 2

        def func(inner_func_x, i):
            return self._interpolations[i](inner_func_x) - half_maxs[i]

        left_roots = [root_scalar(func, (i,), bracket=(mins_x[i], other_bounds[i])).root for i in
                      range(len(self._interpolations))]
        right_roots = [root_scalar(func, (i,), bracket=(other_bounds[i], maxs_x[i])).root for i in
                       range(len(self._interpolations))]
        return np.subtract(right_roots, left_roots)
