import numpy as np
import imageio as imio
import matplotlib.pyplot as plt
import abc
from typing import Type, Tuple, Dict


class StaticSpeckleSimulation(abc.ABC):
    """
    Base abstract class to generate static speckle simulations. No object can be instantiated from it.
    """

    def __init__(self, sim_shape: int):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations `sim_shape`x`sim_shape`. Must be
        strictly positive.
        """
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        self._sim_shape = (sim_shape, sim_shape)
        self._previous_simulation = None

    @property
    def previous_simulation(self):
        """
        Getter of the previous simulation (last one to have been created).
        :return: An array containing the previous simulation.
        """
        return self._previous_simulation

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
        a shape `sim_shape`x`sim_shape`.
        :return: Nothing.
        """
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        self._sim_shape = (sim_shape, sim_shape)

    def save_previous_simulation(self, filepath: str):
        """
        Method used to save the previous simulation (the last one done).
        :param filepath: str. Path / name under which we save the previous simulation. If no extension is provided, we
        save under a TIFF.
        :return: Nothing.
        """
        if self._previous_simulation is None:
            raise ValueError("No simulation to save.")
        if "." not in filepath:
            filepath += ".tiff"
        imio.imwrite(filepath, self._previous_simulation)

    def show_previous_simulation(self):
        """
        Method used to display and show the previous simulation (the last one done).
        :return: Nothing.
        """
        if self._previous_simulation is None:
            raise ValueError("No simulation to show.")
        plt.imshow(self._previous_simulation, "gray")
        plt.colorbar()
        plt.show()

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

    def intensity_histogram(self, n_bins: int = 256, density: bool = True):
        """
        Method used to display and show the intensity histogram of the current simulation.
        :param n_bins: int. Number of bins to use for the histogram. Should be strictly positive. It is 256 by default.
        :param density: bool. Boolean indicating if the histogram should be normalized (i.e. the integral over the
        domain is 1). Default is `True`.
        :return: A tuple of 2 elements. The first is the array of "y" values (i.e. the probability value or counts), the
        second is the array of bin edges.
        """
        if self._previous_simulation is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        values, bin_edges, _ = plt.hist(self._previous_simulation.ravel(), n_bins, density=density)
        plt.show()
        return values, bin_edges


class SpeckleSimulationFromEllipsoidSource(StaticSpeckleSimulation):
    """
    Class used to simulate static speckles with ellipsoid shape. Derived from `StaticSpeckleSimulation`, but is not
    abstract.
    """

    def __init__(self, sim_shape: int, vertical_diameter: float, horizontal_diameter: float):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations `sim_shape`x`sim_shape`. Must be
        strictly positive.
        :param vertical_diameter: float. Vertical diameter of the ellipse, related to the vertical diameter of the
        speckles. Should be strictly positive.
        :param horizontal_diameter: float. Horizontal diameter of the ellipse, related to the horizontal diameter of the
        speckles. Should be strictly positive.
        """
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
        """
        Getter of the vertical diameter.
        :return: The vertical diameter (a float).
        """
        return self._2b

    @vertical_diameter.setter
    def vertical_diameter(self, vertical_diameter: float):
        """
        Setter of the vertical diameter.
        :param vertical_diameter: float. New vertical diameter. Must be strictly positive.
        :return: Nothing.
        """
        if vertical_diameter <= 0:
            raise ValueError("The vertical diameter must be strictly positive.")
        self._2b = vertical_diameter
        self._b = vertical_diameter / 2

    @property
    def horizontal_diameter(self):
        """
        Getter of the horizontal diameter.
        :return: The horizontal diameter (a float).
        """
        return self._2a

    @horizontal_diameter.setter
    def horizontal_diameter(self, horizontal_diameter: float):
        """
        Setter of the horizontal diameter
        :param horizontal_diameter: float. New horizontal diameter. Must be strictly positive.
        :return: Nothing.
        """
        if horizontal_diameter <= 0:
            raise ValueError("The vertical diameter must be strictly positive.")
        self._2a = horizontal_diameter
        self._a = horizontal_diameter / 2

    def simulate(self):
        """
        Method used to do the simulations. Redefinition of the abstract method. This method creates fully developed
        speckles (for now) and uses an "imaging" algorithm (i.e. simulate imaging speckles).
        :return: Nothing.
        """
        mask = self._generate_ellipsoid_mask()
        sim_before_fft = self._generate_phases(-np.pi, np.pi)
        sim = (np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(sim_before_fft)) * mask))) ** 2).real
        sim /= np.max(sim)
        self._previous_simulation = sim

    def _generate_ellipsoid_mask(self):
        """
        Method used to generate an ellipsoid mask (i.e. a mask used to dictate the shape of the speckles).
        :return: The mask (a 2D NumPy array).
        """
        Y, X = np.indices(self._sim_shape)
        Y -= self._sim_shape[0] // 2
        X -= self._sim_shape[1] // 2
        mask = (X ** 2 / self._a ** 2 + Y ** 2 / self._b ** 2) <= 1
        return mask


class SpeckleSimulationFromCircularSource(SpeckleSimulationFromEllipsoidSource):
    # TODO: Override get / set of vertical and horizontal diameters to prevent or do change both at same time.
    """
    Class used to simulate circular speckles. Derived from `SpeckleSimulationFromEllipsoidSource`.
    """

    def __init__(self, sim_shape: int, circle_diameter: float):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations `sim_shape`x`sim_shape`. Must be
        strictly positive.
        :param circle_diameter: float. Diameter of the circle, related to the speckle size. Should be strictly positive.
        """
        super(SpeckleSimulationFromCircularSource, self).__init__(sim_shape, circle_diameter, circle_diameter)
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        self._circle_diameter = circle_diameter
        self._circle_radius = circle_diameter / 2

    @property
    def circle_diameter(self):
        """
        Getter of the circle diameter.
        :return: The circle diameter (a float).
        """
        return self._circle_diameter

    @circle_diameter.setter
    def circle_diameter(self, circle_diameter: float):
        """
        Setter of the circle diameter
        :param circle_diameter: float. New circle diameter (related to the speckle diameter). Must be strictly positive.
        :return: Nothing.
        """
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        self._circle_diameter = circle_diameter
        self._circle_radius = circle_diameter / 2


class SpeckleSimulationFromEllipsoidSourceWithPolarization(SpeckleSimulationFromEllipsoidSource):
    """
    Class used to simulate static partially polarized speckles with ellipsoid shape.
    """

    def __init__(self, sim_shape: int, vertical_diameter: float, horizontal_diameter: float,
                 polarization_degree: float):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations `sim_shape`x`sim_shape`. Must be
        strictly positive.
        :param vertical_diameter:  float. Vertical diameter of the ellipse, related to the vertical diameter of the
        speckles. Should be strictly positive.
        :param horizontal_diameter: float. Horizontal diameter of the ellipse, related to the horizontal diameter of the
        speckles. Should be strictly positive.
        :param polarization_degree: float. Degree of polarization of the laser. Must be between 0 and 1, both included
        (0 is totally unpolarized and 1 is perfectly polarized).
        """
        super(SpeckleSimulationFromEllipsoidSourceWithPolarization, self).__init__(sim_shape, vertical_diameter,
                                                                                   horizontal_diameter)
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    @property
    def polarization_degree(self):
        """
        Getter of the polarization degree.
        :return: The polarization degree, a float.
        """
        return self._polarization_degree

    @polarization_degree.setter
    def polarization_degree(self, polarization_degree: float):
        """
        Setter of the polarization degree.
        :param polarization_degree: float. New polarization degree. Must be between 0 and 1, both included.
        :return: Nothing.
        """
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    def simulate(self):
        """
        Method used to do the simulations. Redefinition of the abstract method. This method created partially polarized
        speckles and uses an "imaging" algorithm (i.e. simulate imaging speckles).
        :return: Nothing.
        """
        super(SpeckleSimulationFromEllipsoidSourceWithPolarization, self).simulate()
        sim_1 = self.previous_simulation.copy()
        super(SpeckleSimulationFromEllipsoidSourceWithPolarization, self).simulate()
        sim_2 = self.previous_simulation.copy()
        first_part = 0.5 * (1 + self._polarization_degree) * sim_1
        second_part = 0.5 * (1 - self._polarization_degree) * sim_2
        self._previous_simulation = first_part + second_part


class SpeckleSimulationFromCircularSourceWithPolarization(SpeckleSimulationFromCircularSource):
    """
    Class used to simulate static partially polarized speckles with circular shape.
    """

    # TODO: inherit from ellipsoid polarized class? Or both?

    def __init__(self, sim_shape: int, circle_diameter: float, polarization_degree: float):
        """
        Initializer of the class.
        :param sim_shape: int. Shape of the simulation, which will create simulations `sim_shape`x`sim_shape`. Must be
        strictly positive.
        :param circle_diameter: float. Diameter of the circle, related to the speckle size. Should be strictly positive.
        :param polarization_degree: float. Degree of polarization of the laser. Must be between 0 and 1, both included
        (0 is totally unpolarized and 1 is perfectly polarized).
        """
        super(SpeckleSimulationFromCircularSourceWithPolarization, self).__init__(sim_shape, circle_diameter)
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    @property
    def polarization_degree(self):
        """
        Getter of the polarization degree.
        :return: The polarization degree, a float.
        """
        return self._polarization_degree

    @polarization_degree.setter
    def polarization_degree(self, polarization_degree: float):
        """
        Setter of the polarization degree.
        :param polarization_degree: float. New polarization degree. Must be between 0 and 1, both included.
        :return: Nothing.
        """
        if not (0 <= polarization_degree <= 1):
            raise ValueError("The polarization degree must be between 0 and 1 (both included).")
        self._polarization_degree = polarization_degree

    def simulate(self):
        """
        Method used to do the simulations. Redefinition of the abstract method. This method created partially polarized
        speckles and uses an "imaging" algorithm (i.e. simulate imaging speckles).
        :return: Nothing.
        """
        super(SpeckleSimulationFromCircularSourceWithPolarization, self).simulate()
        sim_1 = self.previous_simulation.copy()
        super(SpeckleSimulationFromCircularSourceWithPolarization, self).simulate()
        sim_2 = self.previous_simulation.copy()
        first_part = 0.5 * (1 + self._polarization_degree) * sim_1
        second_part = 0.5 * (1 - self._polarization_degree) * sim_2
        self._previous_simulation = first_part + second_part


class PartiallyDevelopedSpeckleSimulation:
    """
    Class used to simulate static partially developed speckles. This class creates a sum of speckles.
    """

    def __init__(self, base_speckle_sim_class: Type[StaticSpeckleSimulation], *class_args, **class_kwargs):
        """
        Initializer of the class.
        :param base_speckle_sim_class: Type derived from `StaticSpeckleSimulation`. Base class to do simulations. Must
        be derived from `StaticSpeckleSimulation`.
        :param class_args: args. Arguments used when creating the class' instance.
        :param class_kwargs: keyword args. Keyword arguments used when creating the class' instance.
        """
        if not issubclass(base_speckle_sim_class, StaticSpeckleSimulation):
            raise TypeError("`base_speckle_class` must be a type derived from `StaticSpeckleSimulation`.")
        self._base_speckle_sim_class = base_speckle_sim_class(*class_args, **class_kwargs)
        self._previous_simulation = None
        self._means = None
        self._cargs = class_args
        self._ckwargs = class_kwargs

    @property
    def base_speckle_sim_class_and_args(self):
        """
        Getter of the class, its arguments and keyword arguments.
        :return: A tuple containing three elements. The first is the base simulation class, the second is the arguments
        used when creating the class' instance (itself a tuple), and the third element is the keyword arguments used
        when creating the class' instance (itself a dictionary).
        """
        return self._base_speckle_sim_class, self._cargs, self._ckwargs

    @base_speckle_sim_class_and_args.setter
    def base_speckle_sim_class_and_args(self, class_args_kwargs: Tuple[Type[StaticSpeckleSimulation], Tuple, Dict]):
        """
        Sette of the class, its arguments and keyword arguments.
        Arg format: (<class>, <args> (empty tuple if no args), <kwargs> (empty dict if no kwargs))
        :param class_args_kwargs: tuple. Tuple of three elements. The first must be the base simulation class (a class
        derived from `StaticSpeckleSimulation`. The second element must be a tuple of arguments. The tuple is empty
        when no arguments are needed. Finally, the third element must be a dictionary of keyword arguments. The
        dictionary is empty when no keyword arguments are needed.
        :return: Nothing.
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
        """
        Getter of the previous simulation (last one to have been created).
        :return: An array containing the previous simulation.
        """
        if self._previous_simulation is None:
            return None
        return self._previous_simulation.copy()

    @property
    def previous_simulation_means(self):
        """
        Getter of the mean of all the individual simulation used for the partially developed sum.
        :return: An array of averages. Can be `None` if no simulation done.
        """
        if self._means is None:
            return None
        return self._means.copy()

    def simulate(self, n_simulations_per_summation: int = 3, do_average: bool = False):
        """
        Method used to do the simulations. Redefinition of the abstract method. This method created partially polarized
        speckles and uses an "imaging" algorithm (i.e. simulate imaging speckles).
        :param n_simulations_per_summation: int. Integer specifying how many speckle patterns. Must be at least 1, where
        1 pattern can be fully developed.
        :param do_average: bool. Boolean specifying if we want to do the average instead of the sum of the patterns. It
        is `False` by default, so the sum is done without normalization.
        :return: Nothing.
        """
        if n_simulations_per_summation <= 0:
            msg = "The number of patterns should be at least 1. Note that 1 pattern can be fully developed depending " \
                  "on the base class."
            raise ValueError(msg)
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
        """
        Method used to display and show the intensity histogram of the sum (or mean) of the speckle patterns.
        :param n_bins: int. Number of bins to use for the histogram. Should be strictly positive. It is 256 by default.
        :param density: bool. Boolean indicating if the histogram should be normalized (i.e. the integral over the
        domain is 1). Default is `True`.
        :return: A tuple of 2 elements. The first is the array of "y" values (i.e. the probability value or counts), the
        second is the array of bin edges.
        """
        if self._previous_simulation is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        values, bin_edges, _ = plt.hist(self._previous_simulation.ravel(), n_bins, density=density)
        plt.show()
        return values, bin_edges

    def show_previous_simulation(self):
        """
        Method used to display and show the previous simulation (the last one done).
        :return: Nothing.
        """
        if self._previous_simulation is None:
            raise ValueError("No simulation to show.")
        plt.imshow(self._previous_simulation, "gray")
        plt.show()
