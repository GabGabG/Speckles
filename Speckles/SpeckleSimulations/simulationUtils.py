class SpeckleSimulationsUtils:
    """
    Class used for utility methods.
    """

    @staticmethod
    def speckle_sizes_from_ellipse_diameters(sim_shape: int, vertical_diameter: float, horizontal_diameter: float):
        """
        Static method used to compute speckles sizes from the simulation shape and the ellipsoid mask dimensions.
        :param sim_shape: int. Integer specifying what is the size of the simulation (dimensions are
        `sim_shape`x`sim_shape`). Should be strictly positive.
        :param vertical_diameter: float. Vertical diameter of the ellipsoid mask. Should be strictly positive.
        :param horizontal_diameter: float. Horizontal diameter of the ellipsoid mask. Should be strictly positive.
        :return: A tuple of floats. The first one is the vertical speckle size (approximate) and the second one is the
        horizontal speckle size (approximate).
        """
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
        """
        Static method used to compute the ellipsoid diameters for a specific simulation shape and wanted speckle shapes.
        :param sim_shape: int. Integer specifying what is the size of the simulation (dimensions are
        `sim_shape`x`sim_shape`). Should be strictly positive.
        :param vertical_speckle_size: float. Wanted vertical size of the speckles. Should be at least 2 pixels
        :param horizontal_speckle_size: float. Wanted vertical size of the speckles. Should be at least 2 pixels
        :return: A tuple of ints. The first one is the vertical diameter of the ellipsoid mask, while the second element
        is the horizontal diameter of the ellipsoid mask.
        """
        # TODO: Why ints in return?
        if vertical_speckle_size < 2 or horizontal_speckle_size < 2:
            raise ValueError("The speckle sizes must be at least 2 pixels.")
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        return int(sim_shape / vertical_speckle_size), int(sim_shape / horizontal_speckle_size)

    @staticmethod
    def speckle_size_from_circle_diameter(sim_shape: int, circle_diameter: float):
        """
        Static method used to comput the desired speckle size from a specific simulation shape and circular mask diameter.
        :param sim_shape: int. Integer specifying what is the size of the simulation (dimensions are
        `sim_shape`x`sim_shape`). Should be strictly positive.
        :param circle_diameter: float. Diameter of the circular mask. Should be strictly positive.
        :return: The (approximate) speckle size as a float.
        """
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        if circle_diameter <= 0:
            raise ValueError("The circle diameter must be strictly positive.")
        return sim_shape / circle_diameter

    @staticmethod
    def circle_diameter_for_specific_speckle_size(sim_shape: int, speckle_size: float):
        """
        Static method used to compute the circular diameter for a specific simulation shape and wanted speckle shape.
        :param sim_shape: int. Integer specifying what is the size of the simulation (dimensions are
        `sim_shape`x`sim_shape`). Should be strictly positive.
        :param speckle_size: float. Wanted size of the speckles. Should be at least 2 pixels
        :return: The (approximate) circular mask diameter as an int.
        """
        # TODO: Why return int?
        if speckle_size < 2:
            raise ValueError("The speckle size must be at least 2 pixels.")
        if sim_shape <= 0:
            raise ValueError("The simulations shape must be strictly positive")
        return int(sim_shape / speckle_size)
