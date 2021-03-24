class AugmentFactory:
    """
    Randomly augment the input image. Two approach included until now:
    - Affine transform
    - TPS transform

    each instance of transform must have the following characteristic:
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
        pass

if __name__ == '__main__':
    pass