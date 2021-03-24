import os, sys
sys.path.append(os.path.dirname(__file__))

#
import numpy as np

#
from ai_interpolate.libs.component.component_wrapper import ComponentWrapper, get_component_color
from ai_interpolate.libs.component.shape_matching_wrapper import ShapeMatchingWrapper
from ai_interpolate.libs.common_utils.image_utils import read_image, imshow


def match_components_three_stage(components_a, components_b, matcher, is_removed):
    pairs = []

    for index_a, a in enumerate(components_a):
        matches = [(b, matcher.process(a, b)) for b in components_b]
        count_true = len([1 for match in matches if match[1][0]])
        if count_true == 0:
            continue

        distances = np.array([match[1][1] for match in matches])
        index_b = int(np.argmin(distances))
        pairs.append([index_a, index_b])

    if len(pairs) == 0:
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, threshold=0.2))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    if len(pairs) == 0 and (not is_removed):
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, pos_filter=False, threshold=0.6))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    return pairs


class ComponentMatching:
    def __init__(self):
        self.component_wrapper  = ComponentWrapper(min_area=300, min_size=10)
        self.smatch_wrapper     = ShapeMatchingWrapper()

    def match(self, im_path_a, im_path_b):
        im_a = read_image(im_path_a)
        mask_a, components_a = self.component_wrapper.extract_on_color_image(im_a)
        get_component_color(components_a, im_a, mode=ComponentWrapper.EXTRACT_COLOR)

        im_b = read_image(im_path_b)
        mask_b, components_b = self.component_wrapper.extract_on_color_image(im_b)
        get_component_color(components_b, im_b, mode=ComponentWrapper.EXTRACT_COLOR)

        positive_pairs_a2b = match_components_three_stage(components_a, components_b, self.smatch_wrapper, is_removed=False)
        positive_pairs_a2b = np.array(positive_pairs_a2b)
        positive_pairs_b2a = match_components_three_stage(components_b, components_a, self.smatch_wrapper, is_removed=False)
        positive_pairs_b2a = np.array(positive_pairs_b2a)

        return (components_a, components_b), (mask_a, mask_b), (positive_pairs_a2b, positive_pairs_b2a)

def combine_pair(positive_pair_a2b, positive_pair_b2a):
    def np2set(np_result):
        return set([tuple(e) for e in np_result.tolist()])

    if len(positive_pair_a2b) == 0 or len(positive_pair_b2a) == 0:
        if len(positive_pair_a2b) == 0: return positive_pair_b2a
        if len(positive_pair_b2a) == 0: return positive_pair_a2b

    positive_pair_a2b_ = np.concatenate([positive_pair_b2a[:,[1]], positive_pair_b2a[:,[0]]], axis=-1)
    intersect_pair = np2set(positive_pair_a2b).intersection(np2set(positive_pair_a2b_))
    intersect_pair = np.array(list(intersect_pair))

    return intersect_pair

def test1():
    import cv2

    im_path_a = "./../../../data/f0001.png"
    im_path_b = "./../../../data/f0005.png"
    im_a = read_image(im_path_a)
    im_b = read_image(im_path_b)

    components, masks, positive_pairs = ComponentMatching().match(im_path_a, im_path_b)
    components_a, components_b = components
    mask_a, mask_b = masks
    positive_pair_a2b, positive_pair_b2a = positive_pairs
    positive_pair = combine_pair(positive_pair_a2b, positive_pair_b2a)

    #
    print ('--visualize')
    print ('positive_pair: ', positive_pair)
    debug_image = np.concatenate([im_a, im_b], axis=1)
    offset_x    = im_a.shape[1]

    for index_a, index_b in positive_pair:
        _debug_image = debug_image.copy()

        component_a = components_a[index_a]
        component_b = components_b[index_b]

        centroid_a  = component_a['centroid']; y_a, x_a = centroid_a.astype(np.int32)
        centroid_b  = component_b['centroid']; y_b, x_b = centroid_b.astype(np.int32)

        print (centroid_a, centroid_b)
        cv2.circle(_debug_image, (x_a, y_a), 3, (255,0,0), thickness=1)
        cv2.circle(_debug_image, (x_b + offset_x, y_b), 3, (255,0,0), thickness=1)
        cv2.line(_debug_image, (x_a, y_a), (x_b + offset_x, y_b), color=(0,255,0), thickness=1)

        imshow(_debug_image)

if __name__ == '__main__':
    test1()