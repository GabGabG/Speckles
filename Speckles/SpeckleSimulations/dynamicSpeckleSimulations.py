import numpy as np
import imageio as imio
import matplotlib.pyplot as plt
import abc
from typing import Union, Tuple, List
from matplotlib.animation import ArtistAnimation, FuncAnimation
import warnings
import seaborn as sns


# TODO: Save animation with something else than pyplot
class DynamicSpeckleSimulations(abc.ABC):
    """
    Base abstract class to generate dynamic speckle simulations. No object can be instantiated from it.
    """

    def __init__(self, sim_shape: int, n_time_steps: int = 500):
        """
        Initializer of the class.
        :param sim_shape:  int. Shape of the simulation, which will create simulations
        `sim_shape`x`sim_shape`x`time steps`. Must be strictly positive.
        :param n_time_steps: int. Number of simulations. Must be greater than one in order to have dynamic speckles.
        """
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        if n_time_steps <= 1:
            raise ValueError("There must be at least 2 time steps for a dynamic speckle pattern.")
        self._sim_shape = (sim_shape, sim_shape)
        self._n_time_steps = n_time_steps
        self._previous_simulations: np.ndarray = None

    @property
    def previous_simulations(self):
        """
        Getter of the previous simulations (last ones to have been created).
        :return: An array containing the previous simulations.
        """
        return self._previous_simulations

    @property
    def time_steps(self):
        """
        Getter of the number of time steps (i.e. number of simulations).
        :return: The number of time steps, an integer.
        """
        return self._n_time_steps

    @time_steps.setter
    @abc.abstractmethod
    def time_steps(self, n_time_steps: int):
        """
        Setter of the number of time steps.
        :param n_time_steps: int. Number of time steps. Must be greater than one.
        :return: Nothing.
        """
        if n_time_steps <= 1:
            raise ValueError("There must be at least 2 time steps for a dynamic speckle pattern.")
        self._n_time_steps = n_time_steps

    @property
    def sim_shape(self):
        """
        Getter of the simulation shape.
        :return: A tuple containing the shape of the simulation.
        """
        return self._sim_shape

    @sim_shape.setter
    def sim_shape(self, sim_shape: int):
        """
        Setter of the shape of the simulation.
        :param sim_shape: int. New shape for the simulation. Must be strictly positive. The future simulations will have
        a shape `sim_shape`x`sim_shape`x`time steps`.
        :return: Nothing.
        """
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        self._sim_shape = (sim_shape, sim_shape)

    def save_previous_simulations(self, filepath: str,
                                  indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        """
        Method used to save the previous simulations (the last ones done).
        :param filepath: str. Path / name under which we save the previous simulations. If no extension is provided, we
        save under a TIFF.
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation indices to
        save under `filepath`. It can be different object. By default, it is a string 'all' which means we save all the
        frames of the simulation (it is the only accepted string). When it is an integer, it means we only keep one
        frame. When a slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the
        ones contained inside.
        :return: Nothing.
        """
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
        """
        Method used to display and show the previous simulations (the last ones done).
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation indices to
        show. It can be different object. By default, it is a string 'all' which means we save all the frames of the
        simulation (it is the only accepted string). When it is an integer, it means we only keep one frame. When a
        slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the ones contained
        inside.
        :return: Nothing.
        """
        if self._previous_simulations is None:
            raise ValueError("No simulation to show.")
        sims = self._get_specific_indices(indices)
        for sim in sims:
            plt.imshow(self._previous_simulations[sim, :, :], cmap="gray")
            plt.show()

    def animate_previous_simulations(self, savename: str = None, ffmpeg_encoder_path: str = None, show: bool = True):
        """
        Method used to animate the previous simulations. Each frame is displayed after a certain interval and does not
        repeat.
        :param savename: str. Path / name under which we save the animation. If no extension is provided, we
        save under a MP4 file. It is `None` by default, which means that it doesn't save anything.
        :param ffmpeg_encoder_path: str. Path leading to the ffmpeg executable file used to save under various formats,
        like MP4. Used by Matplotlib to save videos. Can be required if the path is not known to Matplotlib. It is
        `None` by default, which means that Matplotlib uses its default value. Is not required when the `savename` is
        `None`.
        :param show: bool. Boolean specifying if we want to show the animation when it is ready. Can be useful to be
        `False` when we only want to save the animation. Is `True` by default.
        :return: The animation, a Matplotlib `ArtistAnimation` instance.
        """
        fig, ax = plt.subplots()
        ims = [[ax.imshow(self._previous_simulations[i, :, :], cmap="gray")] for i in range(self._n_time_steps)]
        ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
        if show:
            plt.show()
        if savename is not None:
            if "." not in savename:
                savename += ".mp4"
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)
        return ani

    def intensity_histogram(self, n_bins: int = 256, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all",
                            density: bool = True):
        """
        Method used to display and show the intensity histogram of previous simulations.
        :param n_bins: int. Number of bins for the histogram. Must be strictly positive and is 256 by default.
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation indices to
        show the histogram. It can be different object. By default, it is a string 'all' which means we save all the
        frames of the simulation (it is the only accepted string). When it is an integer, it means we only keep one
        frame. When a slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the
        ones contained inside.
        :param density: bool. Boolean specifying if the histogram should be a density (integral of the histogram is 1,
        like a probability density).
        :return: A tuple of lists. The first one contains the vertical values of the histogram, for each frame shown,
        while the second contains the bin edges of each frame shown.
        """
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
        """
        Method used to animate the simulation histograms.
        :param n_bins: int. Number of bins for the histogram. Must be strictly positive and is 256 by default.
        :param savename: str. Path / name under which we save the animation. If no extension is provided, we
        save under a MP4 file. It is `None` by default, which means that it doesn't save anything.
        :param ffmpeg_encoder_path: str. Path leading to the ffmpeg executable file used to save under various formats,
        like MP4. Used by Matplotlib to save videos. Can be required if the path is not known to Matplotlib. It is
        `None` by default, which means that Matplotlib uses its default value. Is not required when the `savename` is
        `None`.
        :param density: bool. Boolean specifying if the histogram should be a density (integral of the histogram is 1,
        like a probability density).
        :param show: bool. Boolean specifying if we want to show the animation when it is ready. Can be useful to be
        `False` when we only want to save the animation. Is `True` by default.
        :return: The animation, a Matplotlib `FuncAnimation` instance.
        """
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        dists = [self._previous_simulations[i, :, :].ravel() for i in range(self._n_time_steps)]
        fig, ax = plt.subplots()
        ax.hist(dists[0], n_bins, density=density)
        ax.set_title(f"Histogram of time step {0}")

        def update_hist(i):
            ax.clear()
            n, bins, patches = ax.hist(dists[i], n_bins, density=density)
            ax.autoscale()
            ax.set_title(f"Histogram of time step {i}")
            return patches  # .patches

        ani = FuncAnimation(fig, update_hist, frames=self._n_time_steps, blit=False, repeat=False)
        if show:
            plt.show()
        if savename is not None:
            if "." not in savename:
                savename += ".mp4"
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)
        return ani

    def _get_specific_indices(self, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        """
        Protected method used to access specific frames depending on specified indices.
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation to keep.
        It can be different object. By default, it is a string 'all' which means we save all the frames of the
        simulation (it is the only accepted string). When it is an integer, it means we only keep one frame. When a
        slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the ones contained
        inside.
        :return: The wanted simulations indices, a list.
        """
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
        """
        Protected method to generate phases and the phase factors exp(jθ) where  are the phases. For now,
        this method only potentially creates fully developed speckles where the phases are drawn from a uniform
        distribution. When the distribution interval is 2π, we have fully developed speckles.
        :param lower_bound: float. Lower bound of the uniform interval. By default, this is -π.
        :param upper_bound: float. Upper bound (not included) of the uniform interval. By default, this is π.
        :return: The phase factors exp(jθ), a complex array.
        """
        phases = np.random.uniform(lower_bound, upper_bound, self._sim_shape)
        return np.exp(1j * phases)

    @abc.abstractmethod
    def simulate(self):
        """
        Abstract method to be reimplemented in derived class. This is the method to simulate speckles (i.e. the
        simulation algorithm).
        :return: Nothing.
        """
        pass


class DynamicSpeckleSimulationsFromCircularSource(DynamicSpeckleSimulations, abc.ABC):
    """
    Abstract class used to generate circular dynamic speckles. Derived from `DynamicSpeckleSimulations`, but is still
    an abstract class (no object can be instantiated from it).
    """

    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations
        `sim_shape`x`sim_shape`x`time steps`. Must be strictly positive.
        :param n_time_steps: int. Number of simulations. Must be greater than one in order to have dynamic speckles.
        :param circle_diameter: float. Diameter of the circle, related to the speckle size. Should be strictly positive.
        """
        super(DynamicSpeckleSimulationsFromCircularSource, self).__init__(sim_shape, n_time_steps)
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        self._circle_diameter = circle_diameter
        self._circle_radius = circle_diameter / 2

    def _generate_circular_mask(self):
        """
        Protected method used to generate a circular mask (i.e. a mask used to dictate the shape of the speckles).
        :return: The mask (a 2D NumPy array).
        """
        Y, X = np.indices(self._sim_shape)
        Y -= self._sim_shape[0] // 2
        X -= self._sim_shape[1] // 2
        mask = (X ** 2 + Y ** 2 - self._circle_radius ** 2) <= 0
        return mask


class DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation(DynamicSpeckleSimulationsFromCircularSource):
    """
    Class used to generate circular dynamic speckles with a uniform electric field decorrelation (the intensity
    decorrelation is quadratic). Derived from `DynamicSpeckleSimulationsFromCircularSource`.
    """

    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float, r_min: float = 0, r_max: float = 1):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations
        `sim_shape`x`sim_shape`x`time steps`. Must be strictly positive.
        :param n_time_steps: int. Number of simulations. Must be greater than one in order to have dynamic speckles.
        :param circle_diameter: float. Diameter of the circle, related to the speckle size. Should be strictly positive.
        :param r_min: float. The minimum accepted correlation value. Should be less than 1 and more than 0. Should also
        be less than the maximum correlation value. Default is 0.
        :param r_max: float. The maximum accepted correlation value. Should be less than 1 and more than 0. Should also
        be more than the minimum correlation value. Default is 1.
        """
        super(DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation, self).__init__(sim_shape, n_time_steps,
                                                                                                circle_diameter)
        if not (0 <= r_min < r_max):
            raise ValueError("`r_min` must be between 0 and `r_max` (`r_max` excluded).")
        if not (r_min < r_max <= 1):
            raise ValueError("`r_max` must be between `r_min` and 1 (`r_min` excluded).")
        self._r_s = np.linspace(r_min, r_max, self._n_time_steps)

    @property
    def time_steps(self):
        """
        Getter of the number of time steps (i.e. number of simulations).
        :return: The number of time steps, an integer.
        """
        return super(DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation, self).time_steps

    @time_steps.setter
    def time_steps(self, n_time_steps: int):
        """
        Setter of the number of time steps.
        :param n_time_steps: int. Number of time steps. Must be greater than one.
        :return: Nothing.
        """
        super(DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation, self).time_steps = n_time_steps
        self._r_s = np.linspace(self._r_s[0], self._r_s[-1], self._n_time_steps)

    @property
    def r_s(self):
        """
        Getter of the phasor field linear uniform decorrelation. It is a copy (arbitrary choice).
        :return: The phasor field decorrelation, an array. It is a copy of the object.
        """
        return self._r_s.copy()

    @r_s.setter
    def r_s(self, rmin_rmax: Tuple[float, float]):
        """
        Setter of the phasor field linear uniform decorrelation.
        :param rmin_rmax: tuple. Tuple of two elements. The first one is the minimum correlation accepted and the second
        is the maximum correlation accepted. The minimum must be greater (or equal) than 0 and less than 1. The maximum
        must be greater than 0 and less (or equal) than 1. The maximum must be strictly greater than the minimum. Both
        can be `None`. In the case where one is `None`, the value is taken from the previous correlation minimum or
        maximum, depending on which element is `None`.
        :return: Nothing.
        """
        r_min, r_max = rmin_rmax
        if r_min is None:
            r_min = self._r_s[0]
        if r_max is None:
            r_max = self._r_s[-1]
        if not (0 <= r_min < r_max):
            raise ValueError("`r_min` must be between 0 and `r_max` (`r_max` excluded).")
        if not (r_min < r_max <= 1):
            raise ValueError("`r_max` must be between `r_min` and 1 (`r_min` excluded).")
        self._r_s = np.linspace(r_min, r_max, self._n_time_steps)

    def simulate(self):
        """
        Method used to do the simulations. Redefinition of the abstract method. This method creates fully developed
        speckles (for now) and uses an "imaging" algorithm (i.e. simulate imaging speckles). It simulates a certain
        number of speckle patterns.
        :return: An array containing the phasor fields.
        """
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
    """
    Class used to simulate circular speckles following a Brownian motion decorrelation (negative exponential).
    """

    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float, tau_min: float, tau_max: float,
                 tau_c: float):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations
        `sim_shape`x`sim_shape`x`time steps`. Must be strictly positive.
        :param n_time_steps: int. Number of simulations. Must be greater than one in order to have dynamic speckles.
        :param circle_diameter: float. Diameter of the circle, related to the speckle size. Should be strictly positive.
        :param tau_min: float. Minimum time step. Should be greater (or equal) than 0, but less than the maximum time
        step.
        :param tau_max: float. Maximum time step. Should be greater (or equal) than 0, but greater than the minimum time
        step.
        :param tau_c: float. Characteristic decorrelation time. Should be greater than 0.
        """
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
        """
        Getter of the number of time steps (i.e. number of simulations).
        :return: The number of time steps, an integer.
        """
        return super(DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion, self).time_steps

    @time_steps.setter
    def time_steps(self, n_time_steps: int):
        """
        Setter of the number of time steps.
        :param n_time_steps: int. Number of time steps. Must be greater than one.
        :return: Nothing.
        """
        super(DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion, self).time_steps = n_time_steps
        self._tau_s = np.linspace(self._tau_s[0], self._tau_s[-1], self._n_time_steps)
        self._r_s = np.sqrt(np.exp(-self._tau_s / self._tau_c))

    @property
    def r_s(self):
        """
        Getter of the phasor field decorrelation function. Should be exponential decreasing. It is a copy (arbitrary
        choice).
        :return: The phasor field decorrelation, an array. It is a copy of the object.
        """
        return self._r_s.copy()

    # TODO: Do something about the r_s setter. Should not be accessed.

    @property
    def tau_s(self):
        """
        Getter of the decorrelation time steps (not to be confused with the number of time steps).
        :return: An array containing the decorrelation time steps. It is a copy of the object.
        """
        return self._tau_s.copy()

    @property
    def tau_c(self):
        """
        Getter of the characteristic correlation time.
        :return: The characteristic correlation time, a float.
        """
        return self._tau_c

    @tau_s.setter
    def tau_s(self, taumin_taumax: Tuple[float, float]):
        """
        Setter of the minimum and maximum time steps for the decorrelation.
        :param taumin_taumax: tuple. Tuple of 2 elements. The first element is the minimum time step and the second is
        the maximum time step. Both should be greater (or equal) than 0, but the minimum must be strictly less than the
        maximum.
        :return: Nothing.
        """
        tau_min, tau_max = taumin_taumax
        if tau_min is None:
            tau_min = self._tau_s[0]
        if tau_max is None:
            tau_max = self._tau_s[-1]
        if not (0 <= tau_min < tau_max):
            raise ValueError("`tau_min` must be between 0 and `tau_max` (`tau_max` excluded).")
        if not (tau_min < tau_max <= 1):
            raise ValueError("`tau_max` must be greater than `tau_min`(`tau_min` excluded).")
        self._tau_s = np.linspace(tau_min, tau_max, self._n_time_steps)
        self._r_s = np.sqrt(np.exp(-self._tau_s / self._tau_c))

    @tau_c.setter
    def tau_c(self, tau_c: float):
        """
        Setter of the characteristic correlation time.
        :param tau_c: float. Characteristic correlation time. Should be strictly positive.
        :return: Nothing.
        """
        if tau_c <= 0:
            raise ValueError("`tau_c` must be strictly positive.")
        self._tau_c = tau_c
        self._r_s = np.sqrt(np.exp(-self._tau_s / self._tau_c))

    def simulate(self):
        """
        Method used to do the simulations. Redefinition of the abstract method. This method creates fully developed
        speckles (for now) and uses an "imaging" algorithm (i.e. simulate imaging speckles). It simulates a certain
        number of speckle patterns.
        :return: An array containing the phasor fields.
        """
        M_1 = self._generate_phases()
        M_2 = self._generate_phases()
        W = np.multiply.outer(M_1, self._r_s) + np.multiply.outer(M_2, np.sqrt(1 - self._r_s ** 2))
        mask = self._generate_circular_mask()
        masks = np.broadcast_to(mask, (self._n_time_steps, *mask.shape)).transpose((1, 2, 0))
        sims = (np.abs(
            np.fft.ifft2(
                np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(W, axes=(0, 1)), axes=(0, 1)) * masks, axes=(0, 1)),
                axes=(0, 1))) ** 2).real
        # sims /= np.max(sims, (0, 1))
        self._previous_simulations = sims.transpose((2, 0, 1))
        return W


class DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion(DynamicSpeckleSimulationsFromCircularSource):
    """
    Class used to simulate circular dynamic speckles where the speckles are generated from a motion of a pupil.
    This is analogous to moving the sample.
    """

    def __init__(self, sim_shape: int, n_time_steps: int, circle_diameter: float,
                 initial_pupil_position: Tuple[float, float] = (0, 0),
                 final_pupil_position: Tuple[float, float] = (None, None)):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations
        `sim_shape`x`sim_shape`x`time steps`. Must be strictly positive.
        :param n_time_steps: int. Number of simulations. Must be greater than one in order to have dynamic speckles.
        :param circle_diameter: float. Diameter of the circle, related to the speckle size. Should be strictly positive.
        :param initial_pupil_position: Tuple. Tuple of 2 floats. The first one is the horizontal initial position of the
        pupil (or center of the sample). The second one is the vertical initial position of the pupil (or center of the
        sample). The default is `(0,0)` which means that the initial position is the center.
        :param final_pupil_position: Tuple. Tuple of 2 floats. The first one is the horizontal final position of the
        pupil (or center of the sample). The second one is the vertical final position of the pupil (or center of the
        sample). The default is `(None, None)`, which means that the final position is at
        `(circle_diameter, circle_diameter)`.
        """
        super(DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion, self).__init__(sim_shape, n_time_steps,
                                                                                         circle_diameter)
        self._initial_pupil_position = initial_pupil_position
        f_x, f_y = final_pupil_position
        if f_x is None:
            f_x = circle_diameter
        if f_y is None:
            f_y = circle_diameter
        self._final_pupil_position = (f_x, f_y)
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    @property
    def initial_pupil_position(self):
        """
        Getter of the initial pupil (or sample) position.
        :return: The initial position, a tuple of two elements. The first one is the horizontal position and the second
        is the vertical position.
        """
        return self._initial_pupil_position

    @property
    def final_pupil_position(self):
        """
        Getter of the final pupil (or sample) position.
        :return: The final position, a tuple of two elements. The first one is the horizontal position and the second
        is the vertical position.
        """
        return self._final_pupil_position

    @property
    def time_steps(self):
        """
        Getter of the number of time steps (i.e. number of simulations).
        :return: The number of time steps, an integer.
        """
        return super(DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion, self).time_steps

    @time_steps.setter
    def time_steps(self, n_time_steps: int):
        """
        Setter of the number of time steps.
        :param n_time_steps: int. Number of time steps. Must be greater than one.
        :return: Nothing.
        """
        super(DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion, self).time_steps = n_time_steps
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    @initial_pupil_position.setter
    def initial_pupil_position(self, initial_pupil_position: Tuple[float, float]):
        """
        Setter of the initial pupil (or sample) position.
        :param initial_pupil_position: tuple. Tuple of two elements. The first one is the horizontal position, while the
        second is the vertical position.
        :return: Nothing.
        """
        self._initial_pupil_position = initial_pupil_position
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    @final_pupil_position.setter
    def final_pupil_position(self, final_pupil_position: Tuple[float, float]):
        """
        Setter of the final pupil (or sample) position.
        :param final_pupil_position: tuple. Tuple of two elements. The first one is the horizontal position, while the
        second is the vertical position. Both can be `None`, in which case each takes the value of `circle_diameter`.
        :return: Nothing.
        """
        f_x, f_y = final_pupil_position
        if f_x is None:
            f_x = self._circle_diameter
        if f_y is None:
            f_y = self._circle_diameter
        self._final_pupil_position = (f_x, f_y)
        self._positions_x = np.linspace(self._initial_pupil_position[0], self._final_pupil_position[0],
                                        self._n_time_steps)
        self._positions_y = np.linspace(self._initial_pupil_position[1], self._final_pupil_position[1],
                                        self._n_time_steps)

    def _generate_circular_mask(self, show_warning: bool = True):
        """
        (Deprecated) Protected method used to generate the circular masks for the pupil simulation. This method exists
        for compatibility purposes, as well as coherency. Relies on `_generate_circular_masks`.
        :param show_warning: bool. Boolean indicating if we must show the deprecation / future warning. `True` by
        default.
        :return: The circular masks, a 3D array.
        """
        if show_warning:
            warnings.warn("In the future, use `_generate_circular_masks`", FutureWarning)
        return self._generate_circular_masks()

    def _generate_circular_masks(self):
        """
        Protected method used to generate circular masks simulating the pupil motion (or sample motion).
        :return: The circular masks, a 3D array. The shape of the array is (N,M,S) where N is the height of the masks,
        M is the width and S is the number of masks. Each mask is moved in a certain direction depending on where is the
        final position.
        """
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
        """
        Method used to do the simulations. Redefinition of the abstract method. This method creates fully developed
        speckles (for now) and uses an "imaging" algorithm (i.e. simulate imaging speckles). It simulates a certain
        number of speckle patterns.
        :return: Nothing.
        """
        # TODO: Return W for coherency?
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
    """
    Class used to do simulations of dynamic partially developed (here, it is the sum of a certain number of patterns)
    speckles, based on specific decorrelation dynamics.
    """

    def __init__(self, base_simulations: DynamicSpeckleSimulations):
        # TODO: Coherent with static simulations? Which one take: class + args or object already done?
        """
        Initializer of the class.
        :param base_simulations: DynamicSpeckleSimulations. Instance of `DynamicSpeckleSimulations` derived class
        (because `DynamicSpeckleSimulations` itself cannot have instances). Simulations will be based on that base
        object.
        """
        self._base_simulations = base_simulations
        self._previous_simulations = None

    @property
    def previous_simulations(self):
        """
        Getter of the previous simulations (last ones to have been created).
        :return: An array containing the previous simulations.
        """
        if self._previous_simulations is None:
            return None
        return self._previous_simulations.copy()

    def simulate(self, n_simulations_per_summation: int = 3, do_average: bool = False):
        """
        Method used to do the simulations. This method creates partially developed speckles as a sum of fully developed
        speckles.
        :param n_simulations_per_summation: int. Number of patterns to sum in order to obtain the partially developed
        pattern. Must be greater than 1 in order to obtain something partially developed.
        :param do_average: bool. Boolean indicating if we must normalize the sum. This affects the statistics a little,
        but is possible and can be interesting. It is `False` by default.
        :return: Nothing.
        """
        # TODO: Return something?
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
        """
        Method used to save the previous simulations (the last ones done).
        :param filepath: str. Path / name under which we save the previous simulations. If no extension is provided, we
        save under a TIFF.
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation indices to
        save under `filepath`. It can be different object. By default, it is a string 'all' which means we save all the
        frames of the simulation (it is the only accepted string). When it is an integer, it means we only keep one
        frame. When a slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the
        ones contained inside.
        :return: Nothing.
        """
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
        """
        Method used to display and show the previous simulations (the last ones done).
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation indices to
        show. It can be different object. By default, it is a string 'all' which means we save all the frames of the
        simulation (it is the only accepted string). When it is an integer, it means we only keep one frame. When a
        slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the ones contained
        inside.
        :return: Nothing.
        """
        if self._previous_simulations is None:
            raise ValueError("No simulation to show.")
        sims = self._get_specific_indices(indices)
        for sim in sims:
            plt.imshow(self._previous_simulations[sim, :, :], cmap="gray")
            plt.show()

    def animate_previous_simulations(self, savename: str = None, ffmpeg_encoder_path: str = None, show: bool = True):
        """
        Method used to animate the previous simulations. Each frame is displayed after a certain interval and does not
        repeat.
        :param savename: str. Path / name under which we save the animation. If no extension is provided, we
        save under a MP4 file. It is `None` by default, which means that it doesn't save anything.
        :param ffmpeg_encoder_path: str. Path leading to the ffmpeg executable file used to save under various formats,
        like MP4. Used by Matplotlib to save videos. Can be required if the path is not known to Matplotlib. It is
        `None` by default, which means that Matplotlib uses its default value. Is not required when the `savename` is
        `None`.
        :param show: bool. Boolean specifying if we want to show the animation when it is ready. Can be useful to be
        `False` when we only want to save the animation. Is `True` by default.
        :return: The animation, a Matplotlib `ArtistAnimation` instance.
        """
        fig, ax = plt.subplots()
        ims = [[ax.imshow(self._previous_simulations[i, :, :], cmap="gray")] for i in
               range(self._previous_simulations.shape[0])]
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
        """
        Method used to display and show the intensity histogram of previous simulations.
        :param n_bins: int. Number of bins for the histogram. Must be strictly positive and is 256 by default.
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation indices to
        show the histogram. It can be different object. By default, it is a string 'all' which means we save all the
        frames of the simulation (it is the only accepted string). When it is an integer, it means we only keep one
        frame. When a slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the
        ones contained inside.
        :param density: bool. Boolean specifying if the histogram should be a density (integral of the histogram is 1,
        like a probability density).
        :return: A tuple of lists. The first one contains the vertical values of the histogram, for each frame shown,
        while the second contains the bin edges of each frame shown.
        """
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
        """
        Method used to animate the simulation histograms.
        :param n_bins: int. Number of bins for the histogram. Must be strictly positive and is 256 by default.
        :param savename: str. Path / name under which we save the animation. If no extension is provided, we
        save under a MP4 file. It is `None` by default, which means that it doesn't save anything.
        :param ffmpeg_encoder_path: str. Path leading to the ffmpeg executable file used to save under various formats,
        like MP4. Used by Matplotlib to save videos. Can be required if the path is not known to Matplotlib. It is
        `None` by default, which means that Matplotlib uses its default value. Is not required when the `savename` is
        `None`.
        :param density: bool. Boolean specifying if the histogram should be a density (integral of the histogram is 1,
        like a probability density).
        :param show: bool. Boolean specifying if we want to show the animation when it is ready. Can be useful to be
        `False` when we only want to save the animation. Is `True` by default.
        :return: The animation, a Matplotlib `FuncAnimation` instance.
        """
        if self._previous_simulations is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        dists = [self._previous_simulations[i, :, :].ravel() for i in range(self._previous_simulations.shape[0])]
        fig, ax = plt.subplots()
        ax.hist(dists[0], n_bins, density=density)
        ax.set_title(f"Histogram of time step {0}")

        def update_hist(i):
            ax.clear()
            n, bins, patches = ax.hist(dists[i], n_bins, density=density)
            ax.autoscale()
            ax.set_title(f"Histogram of time step {i}")
            return patches  # .patches

        ani = FuncAnimation(fig, update_hist, frames=self._previous_simulations.shape[0], blit=False, repeat=False)
        if show:
            plt.show()
        if savename is not None:
            if ffmpeg_encoder_path is not None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_encoder_path
            ani.save(savename)

    def _get_specific_indices(self, indices: Union[str, int, slice, Tuple, List, np.ndarray] = "all"):
        """
        Protected method used to access specific frames depending on specified indices.
        :param indices: str, int, slice Tuple, List or np.ndarray. Parameter used to specify which simulation to keep.
        It can be different object. By default, it is a string 'all' which means we save all the frames of the
        simulation (it is the only accepted string). When it is an integer, it means we only keep one frame. When a
        slice, we keep the frames included in it. When an iterable (list, tuple, array), we keep the ones contained
        inside.
        :return: The wanted simulations indices, a list.
        """
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
    plt.rcParams.update({"font.size": 36})


    def corr(X, Y):
        X = np.ravel(X)
        Y = np.ravel(Y)
        return np.mean((X - np.mean(X)) * (Y - np.mean(Y))) / (np.std(X) * np.std(Y))


    t_steps = 100
    # speckles = DynamicSpeckleSimulationsFromCircularSourceWithPupilMotion(500, t_steps, 25, (0, 0), (0, 25))
    speckles = DynamicSpeckleSimulationsFromCircularSourceWithUniformCorrelation(500, t_steps, 50)
    speckles = DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion(500, t_steps, 100, 0, 1, 0.3)
    W = speckles.simulate()
    imio.imwrite("dynamic_1.png", speckles.previous_simulations[0])
    imio.imwrite("dynamic_2.png", speckles.previous_simulations[1])
    imio.imwrite("dynamic_last.png", speckles.previous_simulations[-1])
    correlations = []
    for i in range(t_steps):
        correlations.append(corr(speckles.previous_simulations[0], speckles.previous_simulations[i]))
    fig, axe = plt.subplots(1,1, figsize=(5,8))
    axe.plot(np.arange(t_steps), correlations, linewidth=10)
    axe.set_xlabel("Time step [a.u.]")
    axe.set_ylabel("Correlation")
    plt.show()
    exit()
    specks = speckles.previous_simulations
    W2 = speckles.simulate()
    W = np.append(W, W2, -1)
    speckles = np.append(specks, speckles.previous_simulations, 0)

    speckles = (speckles.transpose(1, 2, 0) * np.arange(1, t_steps * 2 + 1)).transpose(2, 0, 1)


    def mu(W_n: np.ndarray, W_m: np.ndarray):
        num = np.mean(W_n * W_m.conj())
        denom = np.sqrt(np.mean(np.abs(W_n) ** 2) * np.mean(np.abs(W_m) ** 2))
        return num / denom


    # speckles = speckles.previous_simulations
    indices = np.arange(0, 200, 10)
    # indices = [0] * 10
    # speckles = [speckles[i] for i in indices]
    size = [len(indices), len(indices)]

    intensity = 0
    angles = []
    matrix4 = np.zeros(size, complex)
    corr_matrix = np.zeros(size, float)
    for ii, i in enumerate(indices):
        intensity += speckles[i]
        # print(np.mean(speckles[i]))
        for jj, j in enumerate(indices):
            current_angle = np.random.normal(0, 2 * np.pi)

            if ii == jj:
                current_angle = 0
            elif ii > jj:
                current_angle *= -1
            current_angle = np.angle(mu(W[:, :, i], W[:, :, j]))
            current_mu_0 = mu(W[:, :, i], W[:, :, j])
            current_mu_1 = np.abs(current_mu_0) * np.exp(1j * current_angle)
            angle = np.angle(current_mu_1)
            angles.append(angle)
            matrix4[ii, jj] = np.sqrt(np.mean(speckles[i]) * np.mean(speckles[j])) * current_mu_1
            corr_matrix[ii, jj] = corr(speckles[i], speckles[j])
    matrix5 = np.abs(matrix4)
    print(angles)
    print("====")
    print(np.linalg.eigvalsh(matrix4))
    print(np.linalg.eigvalsh(matrix5))
    print("====")
    print(matrix4)
    print(matrix5.astype(float))
    print(corr_matrix)
    print("====")
    print(np.linalg.norm(matrix4 - matrix5))
    print(np.linalg.norm(matrix4 - matrix5) / np.linalg.norm(matrix4))
    print("====")
    print(np.max(np.abs(matrix5 - np.abs(matrix4.real))))
    print(np.linalg.svd(matrix4)[1])
    print(np.linalg.svd(matrix5)[1])


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
    # x = np.insert(x, 0, 0)
    plt.plot(x, prob_density_NxN(x, matrix4), linestyle="--", color="red", label="Field correlation")
    plt.plot(x, prob_density_NxN(x, matrix5), linestyle=":", color="black", label="Intensity correlation")
    # plt.plot(x, prob_density_NxN(x, matrix6), color="green", label="Field correlation v2")
    plt.xlabel(r"Intensity $I$ [-]")
    plt.ylabel(r"$P_I(I)$ [-]")
    plt.legend()
    # plt.savefig(f"cas_intermediaire_{case}.png")
    plt.show()
