import numpy as np
import imageio as imio
import matplotlib.pyplot as plt
import abc


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
        plt.show()

    def _generate_phases(self, lower_bound: float = -np.pi, upper_bound: float = np.pi):
        phases = np.random.uniform(lower_bound, upper_bound, self._sim_shape)
        return np.exp(1j * phases)

    @abc.abstractmethod
    def simulate(self):
        pass

    def intensity_histogram(self, n_bins: int = 256):
        if self._previous_simulation is None:
            raise ValueError("No simulation to extract intensity histogram.")
        if n_bins <= 0:
            raise ValueError("The number of bins for the histogram must be at least 1.")
        values, bin_edges, _ = plt.hist(self._previous_simulation.ravel(), n_bins)
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
    def horizontal_property(self):
        return self._2a

    @horizontal_property.setter
    def horizontal_property(self, horizontal_diameter: float):
        if horizontal_diameter <= 0:
            raise ValueError("The vertical diameter must be strictly positive.")
        self._2a = horizontal_diameter
        self._a = horizontal_diameter / 2

    def simulate(self):
        mask = self._generate_ellipsoid_mask()
        sim_before_fft = self._generate_phases(-np.pi, np.pi)
        sim = (np.abs(np.fft.ifft2(np.fft.fft2(sim_before_fft) * mask)) ** 2).real
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


class SpeckleSimulationsUtils:

    @staticmethod
    def speckle_sizes_from_ellipse_diameters(sim_shape: int, vertical_diameter: float, horizontal_diameter: float):
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        if vertical_diameter <= 0:
            raise ValueError("The vertical diameter must be strictly positive.")
        if horizontal_diameter <= 0:
            raise ValueError("The horizontal diameter must be strictly positive.")
        return sim_shape / vertical_diameter, sim_shape / horizontal_diameter

    @staticmethod
    def ellipse_diameters_for_specific_speckle_sizes(sim_shape: int, vertical_speckle_size: float,
                                                     horizontal_speckle_size: float):
        if vertical_speckle_size < 2 or horizontal_speckle_size < 2:
            raise ValueError("The speckle sizes must be at least 2 pixels.")
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        return int(sim_shape / vertical_speckle_size), int(sim_shape / horizontal_speckle_size)

    @staticmethod
    def speckle_size_from_circle_diameter(sim_shape: int, circle_diameter: float):
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        return sim_shape / circle_diameter

    @staticmethod
    def circle_diameter_for_specific_speckle_size(sim_shape: int, speckle_size: float):
        if speckle_size < 2:
            raise ValueError("The speckle size must be at least 2 pixels.")
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        return int(sim_shape / speckle_size)


if __name__ == '__main__':
    circ_diam = SpeckleSimulationsUtils.circle_diameter_for_specific_speckle_size(1000, 4)
    speckle_size = SpeckleSimulationsUtils.speckle_size_from_circle_diameter(1000, 250)
    el = SpeckleSimulationFromEllipsoidSource(1000,
                                              *SpeckleSimulationsUtils.ellipse_diameters_for_specific_speckle_sizes(
                                                  1000, 4, 10))
    el.simulate()
    el.show_previous_simulation()
    el.save_previous_simulation("test.tif")
    print(circ_diam)
    print(speckle_size)
