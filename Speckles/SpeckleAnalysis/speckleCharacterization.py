from speckleImageManipulations import SpeckleImageManipulations
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple


class SpeckleCharacterization:

    def __init__(self, image_path: str = None, image_from_array: np.ndarray = None, image_name: str = None,
                 background_image_to_remove_from_path: str = None,
                 background_image_to_remove_from_array: np.ndarray = None,
                 gaussian_filter_normalization_std: float = 0, median_filter_size: int = 0,
                 local_contrast_kernel_size: int = 7, crop_image: Tuple[Tuple[int, int], Tuple[int, int]] = None,
                 crop_around_center: Tuple[int, int] = None, crop_order: int = -1, func_to_apply: Callable = None,
                 func_to_apply_order: int = -1, *fargs, **fkwargs):
        self._speckle_im = SpeckleImageManipulations(image_path, image_from_array, image_name)
        if background_image_to_remove_from_path is not None:
            self._speckle_im.remove_background(background_image_to_remove_from_path)
        if background_image_to_remove_from_array is not None:
            self._speckle_im.remove_background(None, background_image_to_remove_from_array)
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
        self._local_contrast = self._speckle_im.compute_local_constrast(local_contrast_kernel_size)
        self._speckle_im.do_autocorrelation()
        self._horizontal_speckle_size, self._vertical_speckle_size = self._speckle_im.get_speckle_sizes()

    @property
    def speckle_sizes(self):
        return self._horizontal_speckle_size, self._vertical_speckle_size

    @property
    def speckle_image_name(self):
        return self._speckle_im.image_name

    @speckle_image_name.setter
    def speckle_image_name(self, image_name: str):
        self._speckle_im.image_name = image_name

    @property
    def original_speckle_image(self):
        return self._speckle_im.original_image

    @property
    def modified_speckle_image(self):
        return self._speckle_im.modified_image

    @property
    def autocorrelation(self):
        return self._speckle_im.autocorrelation

    @property
    def local_contrast(self):
        return self._local_contrast.copy()

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

    def show_autocorrelation(self, with_colorbar: bool = True, cmap: str = "gray"):
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

    def show_local_contrast(self, with_colorbar: bool = True):
        plt.imshow(self._local_contrast)
        if with_colorbar:
            plt.colorbar()
        plt.show()

    def show_intensity_histogram(self, n_bins: int = 256, normalized: bool = False):
        plt.hist(self._speckle_im.modified_image.ravel(), bins=n_bins, density=normalized)
        plt.xlabel("Intensity value [-]")
        if normalized:
            plt.ylabel("Probability [-]")
        else:
            plt.ylabel("Count [-]")
        plt.xlim(left=0)
        plt.show()

    def show_local_contrast_histogram(self, n_bins: int = 256, normalized: bool = False):
        plt.hist(self._local_contrast.ravel(), bins=n_bins, density=normalized)
        plt.xlabel("Contrast value [-]")
        if normalized:
            plt.ylabel("Probability [-]")
        else:
            plt.ylabel("Count [-]")
        plt.xlim(left=0)
        plt.show()


if __name__ == '__main__':
    import cv2

    path = r"C:\Users\goubi\Downloads\PremièreMesures.avi"
    video = cv2.VideoCapture(path)
    stack = []
    while True:
        ret, frame = video.read()
        if ret:
            stack.append(frame)
        else:
            break
    stack = np.dstack(stack)
    plt.imshow(stack[:, :, 0])
    plt.show()
    char = SpeckleCharacterization(None, stack[:, :, 0])
    print(char.speckle_sizes)
    plt.hist(stack[:, :, 0].ravel(), 256)
    plt.show()
    exit()
    path = r"../SpeckleSimulations/test.tif"
    sc = SpeckleCharacterization(path)
    sc.show_modified_speckle_image()
    sc.show_autocorrelation()
    sc.show_autocorrelation_slices()
    sc.show_local_contrast()
    sc.show_intensity_histogram()
    sc.show_local_contrast_histogram(normalized=True)
    print(sc.speckle_sizes)
