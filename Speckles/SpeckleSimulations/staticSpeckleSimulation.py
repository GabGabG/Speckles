import numpy as np
import imageio as imio
import matplotlib.pyplot as plt
import abc
from typing import Type, Tuple, Dict

plt.rcParams.update({"font.size": 22})


class StaticSpeckleSimulation(abc.ABC):

    def __init__(self, sim_shape: int):
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        self._sim_shape = (sim_shape, sim_shape)
        self._previous_simulation = None

    @property
    def previous_simulation(self):
        return self._previous_simulation

    @property
    def sim_shape(self):
        return self._sim_shape

    @sim_shape.setter
    def sim_shape(self, sim_shape: int):
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        self._sim_shape = (sim_shape, sim_shape)

    def save_previous_simulation(self, filepath: str):
        if self._previous_simulation is None:
            raise ValueError("No simulation to save.")
        if "." not in filepath:
            filepath += ".tiff"
        imio.imwrite(filepath, self._previous_simulation)

    def show_previous_simulation(self):
        if self._previous_simulation is None:
            raise ValueError("No simulation to show.")
        plt.imshow(self._previous_simulation, "gray")
        plt.colorbar()
        plt.show()

    def _generate_phases(self, lower_bound: float = -np.pi, upper_bound: float = np.pi, uniform: bool = True):
        phases = np.random.uniform(lower_bound, upper_bound, self._sim_shape)
        if not uniform:
            phases = np.random.normal(0, np.pi / 2, self._sim_shape)
        return np.exp(1j * phases)

    @abc.abstractmethod
    def simulate(self):
        pass

    def intensity_histogram(self, n_bins: int = 256, density: bool = True):
        if self._previous_simulation is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        values, bin_edges, _ = plt.hist(self._previous_simulation.ravel(), n_bins, density=density)
        plt.show()
        return values, bin_edges


class SpeckleSimulationFromEllipsoidSource(StaticSpeckleSimulation):

    def __init__(self, sim_shape: int, vertical_diameter: float, horizontal_diameter: float):
        super(SpeckleSimulationFromEllipsoidSource, self).__init__(sim_shape)
        if vertical_diameter <= 0:
            raise ValueError("The vertical diameter must be strictly positive.")
        if horizontal_diameter <= 0:
            raise ValueError("The horizontal diameter must be strictly positive.")
        self._2b = vertical_diameter
        self._b = vertical_diameter / 2
        self._2a = horizontal_diameter
        self._a = horizontal_diameter / 2

    @property
    def vertical_diameter(self):
        return self._2b

    @vertical_diameter.setter
    def vertical_diameter(self, vertical_diameter: float):
        if vertical_diameter <= 0:
            raise ValueError("The vertical diameter must be strictly positive.")
        self._2b = vertical_diameter
        self._b = vertical_diameter / 2

    @property
    def horizontal_diameter(self):
        return self._2a

    @horizontal_diameter.setter
    def horizontal_diameter(self, horizontal_diameter: float):
        if horizontal_diameter <= 0:
            raise ValueError("The vertical diameter must be strictly positive.")
        self._2a = horizontal_diameter
        self._a = horizontal_diameter / 2

    def simulate(self, fully: bool = True):
        mask = self._generate_ellipsoid_mask()
        sim_before_fft = self._generate_phases(-np.pi, np.pi, uniform=fully)
        sim = (np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(sim_before_fft)) * mask))) ** 2).real
        sim /= np.max(sim)
        self._previous_simulation = sim

    def _generate_ellipsoid_mask(self):
        Y, X = np.indices(self._sim_shape)
        Y -= self._sim_shape[0] // 2
        X -= self._sim_shape[1] // 2
        mask = (X ** 2 / self._a ** 2 + Y ** 2 / self._b ** 2) <= 1
        return mask


class SpeckleSimulationFromCircularSource(SpeckleSimulationFromEllipsoidSource):

    def __init__(self, sim_shape: int, circle_diameter: float):
        super(SpeckleSimulationFromCircularSource, self).__init__(sim_shape, circle_diameter, circle_diameter)
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        self._circle_diameter = circle_diameter
        self._circle_radius = circle_diameter / 2

    @property
    def circle_diameter(self):
        return self._circle_diameter

    @circle_diameter.setter
    def circle_diameter(self, circle_diameter: float):
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        self._circle_diameter = circle_diameter
        self._circle_radius = circle_diameter / 2


class SpeckleSimulationFromEllipsoidSourceWithPolarization(SpeckleSimulationFromEllipsoidSource):

    def __init__(self, sim_shape: int, vertical_diameter: float, horizontal_diameter: float,
                 polarization_degree: float):
        super(SpeckleSimulationFromEllipsoidSourceWithPolarization, self).__init__(sim_shape, vertical_diameter,
                                                                                   horizontal_diameter)
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    @property
    def polarization_degree(self):
        return self._polarization_degree

    @polarization_degree.setter
    def polarization_degree(self, polarization_degree: float):
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    def simulate(self):
        super(SpeckleSimulationFromEllipsoidSourceWithPolarization, self).simulate()
        sim_1 = self.previous_simulation.copy()
        super(SpeckleSimulationFromEllipsoidSourceWithPolarization, self).simulate()
        sim_2 = self.previous_simulation.copy()
        first_part = 0.5 * (1 + self._polarization_degree) * sim_1
        second_part = 0.5 * (1 - self._polarization_degree) * sim_2
        self._previous_simulation = first_part + second_part


class SpeckleSimulationFromCircularSourceWithPolarization(SpeckleSimulationFromCircularSource):

    def __init__(self, sim_shape: int, circle_diameter: float, polarization_degree: float):
        super(SpeckleSimulationFromCircularSourceWithPolarization, self).__init__(sim_shape, circle_diameter)
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    @property
    def polarization_degree(self):
        return self._polarization_degree

    @polarization_degree.setter
    def polarization_degree(self, polarization_degree: float):
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    def simulate(self):
        super(SpeckleSimulationFromCircularSourceWithPolarization, self).simulate()
        sim_1 = self.previous_simulation.copy()
        super(SpeckleSimulationFromCircularSourceWithPolarization, self).simulate()
        sim_2 = self.previous_simulation.copy()
        first_part = 0.5 * (1 + self._polarization_degree) * sim_1
        second_part = 0.5 * (1 - self._polarization_degree) * sim_2
        self._previous_simulation = first_part + second_part


class PartiallyDevelopedSpeckleSimulation:

    def __init__(self, base_speckle_sim_class: Type[StaticSpeckleSimulation], *class_args, **class_kwargs):
        if not issubclass(base_speckle_sim_class, StaticSpeckleSimulation):
            raise TypeError("`base_speckle_class` must be a type derived from `StaticSpeckleSimulation`.")
        self._base_speckle_sim_class = base_speckle_sim_class(*class_args, **class_kwargs)
        self._previous_simulation = None
        self._means = None
        self._cargs = class_args
        self._ckwargs = class_kwargs

    @property
    def base_speckle_sim_class_and_args(self):
        return self._base_speckle_sim_class, self._cargs, self._ckwargs

    @base_speckle_sim_class_and_args.setter
    def base_speckle_sim_class_and_args(self, class_args_kwargs: Tuple[Type[StaticSpeckleSimulation], Tuple, Dict]):
        """
        Arg format: (<class>, <args> (empty tuple if no args), <kwargs> (empty dict if no kwargs))
        :param class_args_kwargs:
        :return:
        """
        if len(class_args_kwargs) != 3:
            msg = "There must be 3 elements in `class_args_kwargs`. The first one it the base class, the second is" \
                  " a tuple of arguments for the creation of the base class (empty tuple if no args) and the last one" \
                  " is a dictionary of keyword arguments (empty dictionary if no kwargs)."
            raise ValueError(msg)
        base_class = class_args_kwargs[0]
        class_args = class_args_kwargs[1]
        class_kwargs = class_args_kwargs[2]
        if not issubclass(base_class, StaticSpeckleSimulation):
            raise TypeError("The first argument must be a type derived from `StaticSpeckleSimulation`.")
        if not isinstance(class_args, Tuple):
            raise TypeError("The second argument must be a tuple of arguments (empty if no arguments).")
        if not isinstance(class_kwargs, Dict):
            raise TypeError("The third argument must be a dictionary of keyword arguments (empty if no kwarguments).")
        self._base_speckle_sim_class = base_class
        self._cargs = class_args
        self._ckwargs = class_kwargs

    @property
    def previous_simulation(self):
        if self._previous_simulation is None:
            return None
        return self._previous_simulation.copy()

    @property
    def previous_simulation_means(self):
        if self._means is None:
            return None
        return self._means.copy()

    def simulate(self, n_simulations_per_summation: int = 3, do_average: bool = False):
        n = n_simulations_per_summation
        sim_shape = self._base_speckle_sim_class.sim_shape
        speckles = np.full((*sim_shape, n), np.nan)
        means = np.full(n, np.nan)
        for i in range(n_simulations_per_summation):
            self._base_speckle_sim_class.simulate()
            current_speckle = self._base_speckle_sim_class.previous_simulation
            speckles[:, :, i] = current_speckle
            means[i] = np.mean(current_speckle)
        if do_average:
            ret_speckles = np.mean(speckles, axis=-1)
        else:
            ret_speckles = np.sum(speckles, axis=-1)
        self._previous_simulation = ret_speckles
        self._means = means

    def intensity_histogram(self, n_bins: int = 256, density: bool = True):
        if self._previous_simulation is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        values, bin_edges, _ = plt.hist(self._previous_simulation.ravel(), n_bins, density=density)
        plt.show()
        return values, bin_edges

    def show_previous_simulation(self):
        if self._previous_simulation is None:
            raise ValueError("No simulation to show.")
        plt.imshow(self._previous_simulation, "gray")
        plt.show()
