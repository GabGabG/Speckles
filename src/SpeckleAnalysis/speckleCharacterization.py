from speckleImageManipulations import SpeckleImageManipulations
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple


class SpeckleCharacterization:

    def __init__(self, image_path: str = None, image_from_array: np.ndarray = None, image_name: str = None,
                 gaussian_filter_normalization_std: float = 0, median_filter_size: int = 0,
                 crop_image: Tuple[Tuple[int, int], Tuple[int, int]] = None, crop_around_center: Tuple[int, int] = None,
                 crop_order: int = -1, func_to_apply: Callable = None, func_to_apply_order: int = -1, *fargs,
                 **fkwargs):
        self._speckle_im = SpeckleImageManipulations(image_path, image_from_array, image_name)
        if crop_order == 0 and crop_image is not None:
            self._speckle_im.crop(*crop_image)
        if crop_order == 0 and crop_around_center is not None:
            self._speckle_im.centered_crop(*crop_around_center)
        if func_to_apply_order == 0 and func_to_apply is not None:
            self._speckle_im.apply_on_modified_image(func_to_apply, *fargs, **fkwargs)
        self._speckle_im.apply_gaussian_normalization(gaussian_filter_normalization_std)
        if crop_order == 1 and crop_image is not None:
            self._speckle_im.crop(*crop_image)
        if crop_order == 1 and crop_around_center is not None:
            self._speckle_im.centered_crop(*crop_around_center)
        if func_to_apply_order == 1 and func_to_apply is not None:
            self._speckle_im.apply_on_modified_image(func_to_apply, *fargs, **fkwargs)
        self._speckle_im.apply_median_filter(median_filter_size)
        if crop_order not in [0, 1] and crop_image is not None:
            self._speckle_im.crop(*crop_image)
        if crop_order not in [0, 1] and crop_around_center is not None:
            self._speckle_im.centered_crop(*crop_around_center)
        if func_to_apply_order not in [0, 1] and func_to_apply is not None:
            self._speckle_im.apply_on_modified_image(func_to_apply, *fargs, **fkwargs)
        self._speckle_im.do_autocorrelation()
        self._horizontal_speckle_size, self._vertical_speckle_size = self._speckle_im.get_speckle_sizes()

    def show_original_speckle_image(self, with_colorbar: bool = True, cmap: str = "gray"):
        plt.imshow(self._speckle_im.original_image, cmap)
        if with_colorbar:
            plt.colorbar()
        plt.show()

    def show_modified_speckle_image(self, with_colorbar: bool = True, cmap: str = "gray"):
        plt.imshow(self._speckle_im.modified_image, cmap)
        if with_colorbar:
            plt.colorbar()
        plt.show()

    def show_autocorrelation(self, with_colorbar:bool=True, cmap:str="gray"):
        plt.imshow(self._speckle_im.autocorrelation, cmap)
        if with_colorbar:
            plt.colorbar()
        plt.show()

    def show_autocorrelation_slices(self, slices_pos: Tuple[int, int] = (None, None)):
        fig, (ax1, ax2) = plt.subplots(2)
        h_slice, v_slice = self._speckle_im.access_autocorrelation_slices(slices_pos)
        ax1.plot(range(len(h_slice)), h_slice)
        ax1.set_title("Horizontal slice")
        ax2.plot(range(len(v_slice)), v_slice)
        ax2.set_title("Vertical slice")
        plt.show()
