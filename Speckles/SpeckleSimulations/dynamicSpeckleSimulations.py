import numpy as np
import imageio as imio
import matplotlib.pyplot as plt
import abc
from typing import Union, Tuple, List
from matplotlib.animation import ArtistAnimation, FuncAnimation
import warnings


class DynamicSpeckleSimulations(abc.ABC):

    def __init__(self, sim_shape: int, n_time_steps: int = 500):
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        if n_time_steps <= 1:
            raise ValueError("There must be at least 2 time steps for a dynamic speckle pattern.")
        self._sim_shape = (sim_shape, sim_shape)
        self._n_time_steps = n_time_steps
        self._previous_simulations: np.ndarray = None

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

    def animate_previous_simulations(self, savename: str = None, ffmpeg_encoder_path: str = None, show: bool = True):
        fig, ax = plt.subplots()
        ims = [[ax.imshow(self._previous_simulations[i, :, :], cmap="gray")] for i in range(self._n_time_steps)]
        ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
        if show:
            plt.show()
        if savename is not None:
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)
        return ani

    def intensity_histogram(self, n_bins: int = 256, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all",
                            density: bool = True):
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        sims = self._get_specific_indices(indices)
        all_values = []
        all_bin_edges = []
        for sim in sims:
            values, bin_edges, _ = plt.hist(self._previous_simulations[sim, :, :].ravel(), n_bins, density=density)
            all_values.append(values)
            all_bin_edges.append(bin_edges)
            plt.show()
        return all_values, all_bin_edges

    def animate_previous_simulations_histogram(self, n_bins: int = 256, savename: str = None,
                                               ffmpeg_encoder_path: str = None, density: bool = True,
                                               show: bool = True):
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        dists = [self._previous_simulations[i, :, :].ravel() for i in range(self._n_time_steps)]
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(dists[0], n_bins, density=density)
        global previous_max_y, previous_max_x, previous_min_x
        previous_max_y = max(n)
        previous_max_x = max(bins)
        previous_min_x = min(bins)
        ax.set_ylim(top=previous_max_y)
        ax.set_xlim(previous_min_x - 0.5 * np.abs(previous_min_x), previous_max_x + 0.5 * np.abs(previous_min_x))
        ax.set_title(f"Histogram of time step {0}")

        def update_hist(i):
            ax.clear()
            n, bins, patches = ax.hist(dists[i], n_bins, density=density)

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
        if show:
            plt.show()
        if savename is not None:
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)
        return ani

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
        sims = (np.abs(
            np.fft.ifft2(
                np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(W, axes=(0, 1)), axes=(0, 1)) * masks, axes=(0, 1)),
                axes=(0, 1))) ** 2).real
        sims /= np.max(sims, (0, 1))
        self._previous_simulations = sims.transpose((2, 0, 1))
        return W


class DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion(DynamicSpeckleSimulationsFromCircularSource):

    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float, tau_min: float, tau_max: float,
                 tau_c: float):
        super(DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion, self).__init__(sim_shape, n_time_steps,
                                                                                            circle_diameter)
        if not (0 <= tau_min < tau_max):
            raise ValueError("`tau_min` must be between 0 and `tau_max` (`tau_max` excluded).")
        if not (tau_min < tau_max):
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
        sims = (np.abs(
            np.fft.ifft2(
                np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(W, axes=(0, 1)), axes=(0, 1)) * masks, axes=(0, 1)),
                axes=(0, 1))) ** 2).real
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
        sims = (np.abs(
            np.fft.ifft2(
                np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(W, axes=(0, 1)), axes=(0, 1)) * masks, axes=(0, 1)),
                axes=(0, 1))) ** 2).real
        sims = np.clip(sims, 0, None)
        sims /= np.max(sims, (0, 1))
        self._previous_simulations = sims.transpose((2, 0, 1))


class DynamicSpeckleSimulationsPartiallyDeveloped:

    def __init__(self, base_simulations: DynamicSpeckleSimulations):
        self._base_simulations = base_simulations
        self._previous_simulations = None

    @property
    def previous_simulations(self):
        if self._previous_simulations is None:
            return None
        return self._previous_simulations.copy()

    def simulate(self, n_simulations_per_summation: int = 3, do_average: bool = False):
        if n_simulations_per_summation < 2:
            raise ValueError("The number of simulations per summation must be at least 2.")
        if self._base_simulations.previous_simulations is None:
            self._base_simulations.simulate()
        n = n_simulations_per_summation
        if do_average:
            simulations = [np.mean(self._base_simulations.previous_simulations[i:i + n, :, :], axis=0) for i in
                           range(0, self._base_simulations.time_steps, n)]
        else:
            simulations = [np.sum(self._base_simulations.previous_simulations[i:i + n, :, :], axis=0) for i in
                           range(0, self._base_simulations.time_steps, n)]
        self._previous_simulations = np.stack(simulations, 0)

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
        # TODO: save with other than pyplot (this saves with the borders and the ticks!)
        fig, ax = plt.subplots()
        ims = [[ax.imshow(self._previous_simulations[i, :, :], cmap="gray")] for i in
               range(self._previous_simulations.shape[0])]
        ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
        plt.show()
        if savename is not None:
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)

    def intensity_histogram(self, n_bins: int = 256, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all",
                            density: bool = True):
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        sims = self._get_specific_indices(indices)
        all_values = []
        all_bin_edges = []
        for sim in sims:
            values, bin_edges, _ = plt.hist(self._previous_simulations[sim, :, :].ravel(), n_bins, density=density)
            all_values.append(values)
            all_bin_edges.append(bin_edges)
            plt.show()
        return all_values, all_bin_edges

    def animate_previous_simulations_histogram(self, n_bins: int = 256, savename: str = None,
                                               ffmpeg_encoder_path: str = None, density: bool = True):
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        dists = [self._previous_simulations[i, :, :].ravel() for i in range(self._previous_simulations.shape[0])]
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(dists[0], n_bins, density=density)
        global previous_max_y, previous_max_x, previous_min_x
        previous_max_y = max(n)
        previous_max_x = max(bins)
        previous_min_x = min(bins)
        ax.set_ylim(top=previous_max_y)
        ax.set_xlim(previous_min_x - 0.5 * np.abs(previous_min_x), previous_max_x + 0.5 * np.abs(previous_min_x))
        ax.set_title(f"Histogram of time step {0}")

        def update_hist(i):
            ax.clear()
            n, bins, patches = ax.hist(dists[i], n_bins, density=density)

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

        ani = FuncAnimation(fig, update_hist, frames=self._previous_simulations.shape[0], blit=False, repeat=False)
        plt.show()
        if savename is not None:
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)

    def _get_specific_indices(self, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        if indices == "all":
            sims = range(self._previous_simulations.shape[0])
        elif isinstance(indices, int):
            sims = [indices]
        elif isinstance(indices, slice):
            sims = range(indices.start, indices.stop, indices.step)
        elif isinstance(indices, (Tuple, List, np.ndarray)):
            sims = indices
        else:
            raise ValueError(f"Parameter `{indices}` is not recognized.")
        return sims


if __name__ == '__main__':
    ffmpeg_path = r"C:\Users\goubi\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe"
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path


    def corr(X, Y):
        X = np.ravel(X)
        Y = np.ravel(Y)
        return np.mean((X - np.mean(X)) * (Y - np.mean(Y))) / (np.std(X) * np.std(Y))


    t_steps = 100
    # speckles = DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion(500, t_steps, 25, (0, 0), (0, 25))
    speckles = DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation(500, t_steps, 50)

    W = speckles.simulate()
    specks = speckles.previous_simulations
    correlations = [corr(speckles.previous_simulations[0], speckles.previous_simulations[i]) for i in range(t_steps)]
    W2 = speckles.simulate()
    W = np.append(W, W2, -1)
    speckles = np.append(specks, speckles.previous_simulations, 0)

    def mu(W_n: np.ndarray, W_m: np.ndarray):
        num = np.mean(W_n * W_m.conj())
        denom = np.sqrt(np.mean(np.abs(W_n) ** 2) * np.mean(np.abs(W_m) ** 2))
        return num / denom


    #speckles = speckles.previous_simulations
    indices = [0, 50, 100, 149]
    # speckles = [speckles[i] for i in indices]
    size = [len(indices)] * 2

    intensity = 0
    matrix4 = np.zeros(size, complex)
    matrix5 = matrix4.copy()
    for ii, i in enumerate(indices):
        intensity += speckles[i]
        for jj, j in enumerate(indices):
            matrix4[ii, jj] = np.sqrt(np.mean(speckles[i]) * np.mean(speckles[j])) * mu(W[:, :, i], W[:, :, j])
            matrix5[ii, jj] = np.abs(np.sqrt(np.mean(speckles[i]) * np.mean(speckles[j])) * mu(W[:, :, i], W[:, :, j]))
    print(np.linalg.eigvalsh(matrix4))
    print(np.linalg.eigvalsh(matrix5))
    print(matrix4)
    print(matrix5)


    # matrix_2_r_1 = [np.mean(speckle_1), np.sqrt(np.mean(speckle_1) * np.mean(speckle_2)) * wanted_correlation ** .5]
    # matrix_2_r_2 = [np.sqrt(np.mean(speckle_1) * np.mean(speckle_2)) * wanted_correlation ** .5, np.mean(speckle_2)]
    # matrix_2 = np.array([matrix_2_r_1, matrix_2_r_2])

    def prob_density_2x2(x, W_corr_matrix):
        eigenvals = np.linalg.eigvalsh(W_corr_matrix)
        print(eigenvals)
        part1 = np.exp(-x / eigenvals[0]) / (eigenvals[0] - eigenvals[1])
        part2 = np.exp(-x / eigenvals[1]) / (eigenvals[0] - eigenvals[1])
        return part1 - part2


    def prob_density_3x3(x, W_corr_matrix):
        l_1, l_2, l_3 = np.linalg.eigvalsh(W_corr_matrix)
        print(l_1)
        print(l_2)
        print(l_3)
        if abs(l_1) <= 1e-7:
            part_1 = 0
        else:
            part_1 = l_1 * np.exp(-x / l_1) / ((l_1 - l_2) * (l_1 - l_3))
        if abs(l_2) <= 1e-10:
            l_2 = 0
        if abs(l_3) <= 1e-10:
            l_3 = 0

        part_2 = l_2 * np.exp(-x / l_2) / ((l_1 - l_2) * (l_2 - l_3))
        part_3 = l_3 * np.exp(-x / l_3) / ((l_1 - l_3) * (l_2 - l_3))
        return part_1 - part_2 + part_3


    def prob_density_NxN(x, W_corr_matrix):
        eigenvals = np.abs(np.linalg.eigvalsh(W_corr_matrix))
        density = 0
        for i, eig in enumerate(eigenvals):
            if abs(eig) <= 1e-15:
                continue
            num = eig ** (len(eigenvals) - 2) * np.exp(-x / eig)
            denom = 1
            for j, eig2 in enumerate(eigenvals):
                if i != j:
                    denom *= (eig - eig2)
            density += num / denom
        return density


    vals, x_bins, _ = plt.hist(intensity.ravel(), 256, None, True, label="Intensity histogram")
    x = (x_bins[:-1] + x_bins[1:]) / 2
    #x = np.insert(x, 0, 0)
    plt.plot(x, prob_density_NxN(x, matrix4), linestyle="--", color="red", label="Field correlation")
    plt.plot(x, prob_density_NxN(x, matrix5), linestyle=":", color="black", label="Intensity correlation")
    # plt.plot(x, prob_density_2x2(x, matrix_2), linestyle=":", color="green", label="With intensity correlation")
    plt.xlabel(r"Intensity $I$ [-]")
    plt.ylabel(r"$P_I(I)$ [-]")
    plt.legend()
    plt.show()
