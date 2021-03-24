import cv2
import numpy as np
from lib.augment.core import CoreTransform
from lib.dataset.JointsDataset import get_affine_transform, affine_transform

class AffineTransform(CoreTransform):
    def __init__(self, version='affine'):
        super(AffineTransform, self).__init__()

    def set_random_parameters(self, output_size, c, s, r):
        w, h = output_size

        trans = get_affine_transform(c, s, r, output_size=(w,h))
        self.params = trans

    def get_random_parameters(self):
        return self.params

    def transform_image(self, input_image, output_size, interpolation_mode, **kwargs):
        trans   = self.params
        w, h    = output_size
        output  = cv2.warpAffine(
            input_image,
            trans,
            (int(w), int(h)),
            flags=cv2.INTER_LINEAR)

        return output

    def transform_coordinate(self, xy_coords, input_image, output_size, interpolation_mode, **kwargs):
        trans = self.params
        new_points_loc = []
        for xy_coord in xy_coords:
            x, y = xy_coord
            x = int(x)
            y = int(y)

            point_loc = np.array([x,y])
            new_point_loc = affine_transform(xy_coord, trans)
            new_points_loc += [new_point_loc]

        return np.array(new_points_loc)

if __name__ == '__main__':
    image_path = "/home/kan/Desktop/cinnamon/CharacterGAN/datasets/hor02_037_C/C/output/C0001.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    output_size = (200, 180)
    fixed_src_points = np.array([[100, 75], [80, 108]])  # xy coordinates

    affine_transform = AffineTransform(version='affine')
    affine_transform.set_random_parameters(output_size=(380, 480), c=np.array([w/2, h/2]), s=np.array([w/200., h/200.]), r=0.)

    output_image = affine_transform.transform_image(image, output_size=(380, 480), interpolation_mode='linear')

    import matplotlib.pyplot as plt
    plt.imshow(output_image)
    plt.show()