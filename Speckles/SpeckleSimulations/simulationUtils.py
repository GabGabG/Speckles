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

