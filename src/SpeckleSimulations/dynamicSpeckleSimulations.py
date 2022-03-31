import numpy as np
import imageio as imio
import matplotlib.pyplot as plt
import abc
from typing import Union, Tuple, List
from matplotlib.animation import ArtistAnimation, FuncAnimation
import warnings


# TODO: Shift before * masks

class DynamicSpeckleSimulations(abc.ABC):

    def __init__(self, sim_shape: int, n_time_steps: int = 500):
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        if n_time_steps <= 1:
            raise ValueError("There must be at least 2 time steps for a dynamic speckle pattern.")
        self._sim_shape = (sim_shape, sim_shape)
        self._n_time_steps = n_time_steps
        self._previous_simulations = None

    @property
    def previous_simulations(self):
        return self._previous_simulations

    @property
    def time_steps(self):
        return self._n_time_steps

    @time_steps.setter
    @abc.abstractmethod
    def time_steps(self, n_time_steps: int):
        if n_time_steps <= 1:
            raise ValueError("There must be at least 2 time steps for a dynamic speckle pattern.")
        self._n_time_steps = n_time_steps

    @property
    def sim_shape(self):
        return self._sim_shape

    @sim_shape.setter
    def sim_shape(self, sim_shape: int):
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        self._sim_shape = (sim_shape, sim_shape)

    def save_previous_simulations(self, filepath: str,
                                  indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        if self._previous_simulations is None:
            raise ValueError("No simulation to save.")
        if indices == "all":
            sims_to_save = self._previous_simulations
        else:
            sims = self._get_specific_indices(indices)
            sims_to_save = self._previous_simulations[sims, :, :]
        if "." not in filepath:
            filepath += ".tiff"
        imio.mimwrite(filepath, sims_to_save)

    def show_previous_simulations(self, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        if self._previous_simulations is None:
            raise ValueError("No simulation to show.")
        sims = self._get_specific_indices(indices)
        for sim in sims:
            plt.imshow(self._previous_simulations[sim, :, :], cmap="gray")
            plt.show()

    def animate_previous_simulations(self, savename: str = None, ffmpeg_encoder_path: str = None):
        fig, ax = plt.subplots()
        ims = [[ax.imshow(self._previous_simulations[i, :, :], cmap="gray")] for i in range(self._n_time_steps)]
        ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
        plt.show()
        if savename is not None:
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)

    def intensity_histogram(self, n_bins: int = 256, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        sims = self._get_specific_indices(indices)
        all_values = []
        all_bin_edges = []
        for sim in sims:
            values, bin_edges, _ = plt.hist(self._previous_simulations[sim, :, :].ravel(), n_bins)
            all_values.append(values)
            all_bin_edges.append(bin_edges)
            plt.show()
        return all_values, all_bin_edges

    def animate_previous_simulations_histogram(self, n_bins: int = 256, savename: str = None,
                                               ffmpeg_encoder_path: str = None):
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        dists = [self._previous_simulations[i, :, :].ravel() for i in range(self._n_time_steps)]
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(dists[0], 256)
        global previous_max_y, previous_max_x, previous_min_x
        previous_max_y = max(n)
        previous_max_x = max(bins)
        previous_min_x = min(bins)
        ax.set_ylim(top=previous_max_y)
        ax.set_xlim(previous_min_x - 0.5 * np.abs(previous_min_x), previous_max_x + 0.5 * np.abs(previous_min_x))
        ax.set_title(f"Histogram of time step {0}")

        def update_hist(i):
            ax.clear()
            n, bins, patches = ax.hist(dists[i], 256)

            current_max_y = max(n)
            current_max_x = max(bins)
            current_min_x = min(bins)
            global previous_max_y, previous_max_x, previous_min_x
            if current_max_y > previous_max_y:
                previous_max_y = current_max_y
            if current_max_x > previous_max_x:
                previous_max_x = current_max_x
            if current_min_x < previous_min_x:
                previous_min_x = current_min_x
            ax.set_ylim(top=previous_max_y)
            ax.set_xlim(previous_min_x - 0.5 * np.abs(previous_min_x), previous_max_x + 0.5 * np.abs(previous_min_x))
            ax.set_title(f"Histogram of time step {i}")
            return patches  # .patches

        ani = FuncAnimation(fig, update_hist, frames=self._n_time_steps, blit=False, repeat=False)
        plt.show()
        if savename is not None:
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)

    def _get_specific_indices(self, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        if indices == "all":
            sims = range(self._n_time_steps)
        elif isinstance(indices, int):
            sims = [indices]
        elif isinstance(indices, slice):
            sims = range(indices.start, indices.stop, indices.step)
        elif isinstance(indices, (Tuple, List, np.ndarray)):
            sims = indices
        else:
            raise ValueError(f"Parameter `{indices}` is not recognized.")
        return sims

    def _generate_phases(self, lower_bound: float = -np.pi, upper_bound: float = np.pi):
        phases = np.random.uniform(lower_bound, upper_bound, self._sim_shape)
        return np.exp(1j * phases)

    @abc.abstractmethod
    def simulate(self):
        pass


class DynamicSpeckleSimulationsFromCircularSource(DynamicSpeckleSimulations, abc.ABC):
    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float):
        super(DynamicSpeckleSimulationsFromCircularSource, self).__init__(sim_shape, n_time_steps)
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        self._circle_diameter = circle_diameter
        self._circle_radius = circle_diameter / 2

    def _generate_circular_mask(self):
        Y, X = np.indices(self._sim_shape)
        Y -= self._sim_shape[0] // 2
        X -= self._sim_shape[1] // 2
        mask = (X ** 2 + Y ** 2 - self._circle_radius ** 2) <= 0
        return mask


class DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation(DynamicSpeckleSimulationsFromCircularSource):

    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float, r_min: float = 0, r_max: float = 1):
        super(DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation, self).__init__(sim_shape, n_time_steps,
                                                                                                circle_diameter)
        if not (0 <= r_min < r_max):
            raise ValueError("`r_min` must be between 0 and `r_max` (`r_max` excluded).")
        if not (r_min < r_max <= 1):
            raise ValueError("`r_max` must be between `r_min` and 1 (`r_min` excluded).")
        self._r_s = np.linspace(r_min, r_max, self._n_time_steps)

    @property
    def time_steps(self):
        return super(DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation, self).time_steps

    @time_steps.setter
    def time_steps(self, n_time_steps: int):
        super(DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation, self).time_steps = n_time_steps
        self._r_s = np.linspace(self._r_s[0], self._r_s[-1], self._n_time_steps)

    @property
    def r_s(self):
        return self._r_s.copy()

    @r_s.setter
    def r_s(self, rmin_rmax: Tuple[float, float]):
        r_min, r_max = rmin_rmax
        if not (0 <= r_min < r_max):
            raise ValueError("`r_min` must be between 0 and `r_max` (`r_max` excluded).")
        if not (r_min < r_max <= 1):
            raise ValueError("`r_max` must be between `r_min` and 1 (`r_min` excluded).")
        self._r_s = np.linspace(r_min, r_max, self._n_time_steps)

    def simulate(self):
        M_1 = self._generate_phases()
        M_2 = self._generate_phases()
        W = np.multiply.outer(M_1, self._r_s) + np.multiply.outer(M_2, np.sqrt(1 - self._r_s ** 2))
        mask = self._generate_circular_mask()
        masks = np.broadcast_to(mask, (self._n_time_steps, *mask.shape)).transpose((1, 2, 0))
        sims = (np.abs(np.fft.ifft2(np.fft.fft2(W, axes=(0, 1)) * masks, axes=(0, 1))) ** 2).real
        sims /= np.max(sims, (0, 1))
        self._previous_simulations = sims.transpose((2, 0, 1))


class DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion(DynamicSpeckleSimulationsFromCircularSource):

    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float, tau_min: float, tau_max: float,
                 tau_c: float):
        super(DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion, self).__init__(sim_shape, n_time_steps,
                                                                                            circle_diameter)
        if not (0 <= tau_min < tau_max):
            raise ValueError("`tau_min` must be between 0 and `tau_max` (`tau_max` excluded).")
        if not (tau_min < tau_max <= 1):
            raise ValueError("`tau_max` must be greater than `tau_min`(`tau_min` excluded).")
        if tau_c <= 0:
            raise ValueError("`tau_c` must be strictly positive.")
        self._tau_s = np.linspace(tau_min, tau_max, self._n_time_steps)
        self._tau_c = tau_c
        self._r_s = np.sqrt(np.exp(-self._tau_s / self._tau_c))

    @property
    def time_steps(self):
        return super(DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion, self).time_steps

    @time_steps.setter
    def time_steps(self, n_time_steps: int):
        super(DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion, self).time_steps = n_time_steps
        self._tau_s = np.linspace(self._tau_s[0], self._tau_s[-1], self._n_time_steps)
        self._r_s = np.sqrt(np.exp(-self._tau_s / self._tau_c))

    @property
    def r_s(self):
        return self._r_s.copy()

    @property
    def tau_s(self):
        return self._tau_s.copy()

    @property
    def tau_c(self):
        return self._tau_c

    @tau_s.setter
    def tau_s(self, taumin_taumax: Tuple[float, float]):
        tau_min, tau_max = taumin_taumax
        if not (0 <= tau_min < tau_max):
            raise ValueError("`tau_min` must be between 0 and `tau_max` (`tau_max` excluded).")
        if not (tau_min < tau_max <= 1):
            raise ValueError("`tau_max` must be greater than `tau_min`(`tau_min` excluded).")
        self._tau_s = np.linspace(tau_min, tau_max, self._n_time_steps)
        self._r_s = np.sqrt(np.exp(-self._tau_s / self._tau_c))

    @tau_c.setter
    def tau_c(self, tau_c: float):
        if tau_c <= 0:
            raise ValueError("`tau_c` must be strictly positive.")
        self._tau_c = tau_c
        self._r_s = np.sqrt(np.exp(-self._tau_s / self._tau_c))

    def simulate(self):
        M_1 = self._generate_phases()
        M_2 = self._generate_phases()
        W = np.multiply.outer(M_1, self._r_s) + np.multiply.outer(M_2, np.sqrt(1 - self._r_s ** 2))
        mask = self._generate_circular_mask()
        masks = np.broadcast_to(mask, (self._n_time_steps, *mask.shape)).transpose((1, 2, 0))
        sims = (np.abs(np.fft.ifft2(np.fft.fft2(W, axes=(0, 1)) * masks, axes=(0, 1))) ** 2).real
        sims /= np.max(sims, (0, 1))
        self._previous_simulations = sims.transpose((2, 0, 1))


class DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion(DynamicSpeckleSimulationsFromCircularSource):

    def __init__(self, sim_shape: int, n_time_step: int, circle_diameter: float,
                 initial_pupil_position: Tuple[float, float], final_pupil_position: Tuple[float, float]):
        super(DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion, self).__init__(sim_shape, n_time_step,
                                                                                         circle_diameter)
        self._initial_pupil_position = initial_pupil_position
        self._final_pupil_position = final_pupil_position
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    @property
    def initial_pupil_position(self):
        return self._initial_pupil_position

    @property
    def final_pupil_position(self):
        return self._final_pupil_position

    @property
    def time_steps(self):
        return super(DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion, self).time_steps

    @time_steps.setter
    def time_steps(self, n_time_steps: int):
        super(DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion, self).time_steps = n_time_steps
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    @initial_pupil_position.setter
    def initial_pupil_position(self, initial_pupil_position: Tuple[float, float]):
        self._initial_pupil_position = initial_pupil_position
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    @final_pupil_position.setter
    def final_pupil_position(self, final_pupil_position: Tuple[float, float]):
        self._final_pupil_position = final_pupil_position
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    def _generate_circular_mask(self):
        warnings.warn("In the future, use `_generate_circular_masks`", FutureWarning)
        return self._generate_circular_masks()

    def _generate_circular_masks(self):
        Y, X = np.indices(self._sim_shape)
        Y -= self._sim_shape[0] // 2
        X -= self._sim_shape[1] // 2
        masks = np.full((*self._sim_shape, self._n_time_steps), np.nan)
        for i in range(self._n_time_steps):
            center_x = self._positions_x[i]
            center_y = self._positions_y[i]
            mask = ((X - center_x) ** 2 + (Y - center_y) ** 2 - self._circle_radius ** 2) <= 0
            masks[:, :, i] = mask
        return masks

    def simulate(self):
        masks = self._generate_circular_masks()
        W = self._generate_phases(-np.pi, np.pi)
        W = np.broadcast_to(W, (self._n_time_steps, *W.shape)).transpose((1, 2, 0))
        sims = (np.abs(np.fft.ifft2(np.fft.fft2(W, axes=(0, 1)) * masks, axes=(0, 1))) ** 2).real
        sims /= np.max(sims, (0, 1))
        self._previous_simulations = sims.transpose((2, 0, 1))


if __name__ == '__main__':
    # speckles = DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation(600, 25, 200)
    # speckles.simulate()
    # print(speckles.previous_simulations.shape)
    # speckles.animate_previous_simulations("test.mp4", r'C:\Users\goubi\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe')
    # speckles.animate_previous_simulations_histogram()
    speckles_2 = DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion(1000, 50, 100, (0, 0), (200, 200))
    speckles_2.simulate()
    speckles_2.animate_previous_simulations()
