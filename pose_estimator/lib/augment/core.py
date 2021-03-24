class CoreTransform:
    """
    Each instances of transform must have the following characteristic:
    > function:
        1. set_random_parameters(**kwargs):
            + random transform parameters
            + user can specify parameters in kwargs
            + save parameters in global variables like self.params
        1.1. get_random_parameters():

        2. transform_image(input_image):
            + transform input image with the self.params already generated (1. need run before 2.)
            +
        2.1. transform_coordinate(xy-coordinate)
            + transform 2d-coordinate (xy) with the self.params
    """
    def __init__(self):
        self.params  = None

    def set_random_parameters(self, **kwargs):
        raise Exception('implement in child')

    def get_random_parameters(self):
        raise Exception('implement in child')

    def transform_image(self, input_image, output_size, interpolation_mode, **kwargs):
        raise Exception('implement in child')

    def transform_coordinate(self, xy_coord, input_image, output_size, interpolation_mode, **kwargs):
        raise Exception('implement in child')