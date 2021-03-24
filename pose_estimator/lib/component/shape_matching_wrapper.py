import numpy as np
import cv2
from skimage import measure
from math import copysign, log10


def normalize_moment(moment):
    """
    Note that hu[0] is not comparable in magnitude as hu[6].
    We can use use a log transform given below to bring them in the same range
    """
    return -1 * copysign(1.0, moment + 1e-6) * log10(abs(moment + 1e-6))


class ShapeMatchingWrapper:
    descriptor_orb = None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __init__(self, area_epsilon=0.2):
        self.area_epsilon = area_epsilon
        self.area_ratio_range = [1 - self.area_epsilon, 1 + self.area_epsilon]
        self.pos_diff_range = [0, 200]

    @staticmethod
    def matching_template(comp_im, template, method=cv2.TM_CCOEFF_NORMED, draw_result=False):
        im = comp_im["image"]
        im_h, im_w = im.shape[:2]
        template_h, template_w = template.shape[:2]

        if im_h < template_h or im_w < template_w:
            return 0., 0.

        res = cv2.matchTemplate(im, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

        min_x, min_y = top_left
        max_x, max_y = bottom_right

        sub_im = im[min_y:max_y, min_x:max_x]
        rows, cols = np.where(sub_im > 0)

        return len(rows), max_val

    @staticmethod
    def calculate_iou(comp1, comp2, img_shape, mode=1):

        """
        :param comp1: first component, type dict.
        :param comp2: second component, type dict.
        :param img_shape: (height,width) of the original image.
        :param mode: If
            1: value = iou / np.sqrt(len(1) * len(2))
            2: value = iou / len(2)
        :return:
        """
        h, w = img_shape

        coords1 = comp1["coords"]
        coords2 = comp2["coords"]

        coords1_flatten = coords1[:, 0] * w + coords1[:, 1]
        coords2_flatten = coords2[:, 0] * w + coords2[:, 1]
        intersect_points = np.intersect1d(coords1_flatten, coords2_flatten)

        if mode == 1:
            iou_ratio = len(intersect_points) / np.sqrt(coords2.shape[0] * coords1.shape[0])
        elif mode == 2:
            iou_ratio = len(intersect_points) / coords2_flatten.shape[0]
        else:
            iou_ratio = len(intersect_points)

        return iou_ratio

    @staticmethod
    def compare_ssim(ref_img, im):
        _bbox = cv2.resize(im["image"], (50, 50))
        _ref_bbox = cv2.resize(ref_img["image"], (50, 50))

        _area_ratio = float(im["area"]) / float(ref_img["area"])
        score = measure.compare_ssim(_ref_bbox, _bbox, full=False)

        return _area_ratio, score

    @staticmethod
    def compare_central_moment(ref_comp, comp, org_shape):
        """
        Compare central moment, invarient to translation
        ------------------
        Step1: finding countours of both component
        Step2:

        Args:
            ref_comp:
            comp:
            org_shape:
        Returns:
        """
        h, w = org_shape

        # building image
        ref_im = np.zeros((h, w), dtype=np.uint8)
        ref_im[ref_comp["coords"][:, 0], ref_comp["coords"][:, 1]] = 255

        im = np.zeros((h, w), dtype=np.uint8)
        im[comp["coords"][:, 0], comp["coords"][:, 1]] = 255

        # calculate central moment then normalize
        ref_central_moment = measure.moments_central(ref_im, order=1)
        central_moment = measure.moments_central(im, order=1)
        return abs(normalize_moment(ref_central_moment[1, 1]) - normalize_moment(central_moment[1, 1]))

    @staticmethod
    def compare_orb(im1, im2, upper_bound_distance):
        descriptor = ShapeMatchingWrapper.descriptor_orb
        bf = ShapeMatchingWrapper.bf

        kp1, des1 = descriptor.detectAndCompute(im1, None)
        kp2, des2 = descriptor.detectAndCompute(im2, None)

        matches = bf.match(des1, des2)
        matches = [match for match in matches if match.distance <= upper_bound_distance]

        return matches

    def get_area(self, binary_image):
        return cv2.countNonZero(binary_image)

    def get_position(self, binary_image):
        # calculate moments of binary image
        M = cv2.moments(binary_image)

        # calculate x,y coordinate of center
        c_x = int(M["m10"] / M["m00"])
        c_y = int(M["m01"] / M["m00"])
        return np.array([c_x, c_y])

    def process(self, a, b, area_filter=True, pos_filter=True, threshold=0.01):
        """
        :param a: binary image
        :param b: binary image
        :param area_filter:
        :param pos_filter:
        :param threshold:
        :return:
        """
        # color
        if a["color"] != b["color"]:
            return False, 2

        # area
        area1, area2 = a.get("area", self.get_area(a["image"])), b.get("area", self.get_area(b["image"]))
        area_ratio = float(area2) / area1
        if (not self.area_ratio_range[0] <= area_ratio <= self.area_ratio_range[1]) and area_filter:
            return False, 2

        # position
        mean_pos1 = a.get("centroid", self.get_position(a["image"]))
        mean_pos2 = b.get("centroid", self.get_position(b["image"]))
        diff_pos = np.abs(mean_pos2 - mean_pos1)
        if not (self.pos_diff_range[0] <= diff_pos[0] <= self.pos_diff_range[1]
                and self.pos_diff_range[0] <= diff_pos[1] <= self.pos_diff_range[1]):
            if pos_filter:
                return False, 2

        # shape
        d2 = cv2.matchShapes(a["image"], b["image"], cv2.CONTOURS_MATCH_I2, 0)
        is_same = d2 <= threshold
        return is_same, d2