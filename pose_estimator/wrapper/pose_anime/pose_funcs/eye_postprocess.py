import cv2
import numpy as np
from skimage import measure

from lib.component.component_wrapper import ComponentWrapper
from wrapper.pose_anime.utils import imgshow

def calc_overlap(coords1, coords2, org_size, return_count_only=True):
    """
    > coords1: of size (n, 2). xy-coordinate
    > coords2: of size (m, 2). xy-coordinate
    """
    #
    w, h = org_size
    coords1_flat = coords1[:, 1] * w + coords1[:, 0]
    coords2_flat = coords2[:, 1] * w + coords2[:, 0]

    #
    intersec_points = np.intersect1d(coords1_flat, coords2_flat, assume_unique=True)

    #
    if return_count_only:
        return len(intersec_points)
    else:
        raise NotImplementedError

def components2point(components, orgsize):
    w, h = orgsize
    mask = np.zeros(shape=(h,w))

    ys = [c['coords'][:,0] for c in components]; ys = np.concatenate(ys, axis=0)
    xs = [c['coords'][:,1] for c in components]; xs = np.concatenate(xs, axis=0)
    coords = np.stack([xs, ys], axis=1)

    x, y, w, h = cv2.boundingRect(coords)
    return x + w // 2, y + h //2

class EyePostprocess:
    def __init__(self, threshold=0.9):
        self.keypoint_names = ['leye', 'reye']
        self.component_wrapper = ComponentWrapper(min_area=1, min_size=1)
        self.threshold = threshold

    def process(self, input_image, keypoints_dict):
        lkp_name = self.keypoint_names[0]
        h, w = input_image.shape[:2]

        if lkp_name not in keypoints_dict: return

        #
        kp_coord, kp_heatmap = keypoints_dict.get(lkp_name)
        kp_maxval  = kp_heatmap[kp_coord[1], kp_coord[0]]
        lkp_coords = np.stack(np.where(kp_heatmap >= 0.5 * kp_maxval)[::-1], axis=1) #xy-coordinates

        #
        padding = 80
        crop_xmin = np.min(lkp_coords[:,0]); crop_xmin = np.clip(crop_xmin - padding, 0, w)
        crop_xmax = np.max(lkp_coords[:,0]); crop_xmax = np.clip(crop_xmax + padding, 0, w)
        crop_ymin = np.min(lkp_coords[:,1]); crop_ymin = np.clip(crop_ymin - padding, 0, h)
        crop_ymax = np.max(lkp_coords[:,1]); crop_ymax = np.clip(crop_ymax + padding, 0, h)

        image_crop = np.ones_like(input_image) * 255
        image_crop[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = input_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        mask, components = self.component_wrapper.extract_on_color_image(image_crop)
        imgshow(image_crop)

        components_filtered = []
        for component in components:
            ys = component['coords'][:,0]
            xs = component['coords'][:,1]
            c_coords = np.stack([xs, ys], axis=-1)

            n_intersect = calc_overlap(lkp_coords, c_coords, return_count_only=True, org_size=(w,h)) # order is not matter
            ratio = n_intersect / float(len(c_coords))

            if ratio > self.threshold:
                print ('\t> Yeah! chosen component with label:' , component['label'], ratio)
                components_filtered += [component]

        # show
        vis_image = np.ones(shape=(h,w,3), dtype=np.uint8) * 255
        for component in components_filtered:
            ys = component['coords'][:, 0]
            xs = component['coords'][:, 1]

            vis_image[ys, xs] = input_image[ys, xs]

        print ('visualizing chosen')
        imgshow(vis_image)

        central_point = components2point(components_filtered, (w, h))
        return central_point

if __name__ == '__main__':

    pass