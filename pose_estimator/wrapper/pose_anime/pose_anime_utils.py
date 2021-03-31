import numpy as np
import cv2
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt

def imgshow(im):
    plt.imshow(im)
    plt.show()

class FakedArugmentPasser:
    def __init__(self, config_path, weight_path):
        self.cfg = config_path
        self.opts = ['TEST.MODEL_FILE', weight_path]

def _update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

def draw_kp_w_im(input_image, keypoints_dct, pair, radius=3, thickness=3):
    vis_image = input_image.copy()
    for k, (point,_) in keypoints_dct.items():
        cv2.circle(vis_image, point, radius=radius, color=(0, 255, 0), thickness=thickness)

    for p1_lbl, p2_lbl in pair:
        if p1_lbl in keypoints_dct and p2_lbl in keypoints_dct:
            #print('draw line btw:', p1_lbl, p2_lbl)
            cv2.line(vis_image, keypoints_dct[p1_lbl][0], keypoints_dct[p2_lbl][0], (255, 255, 0), thickness=thickness)
    return vis_image

def heatmap2image(heatmap):
    """
    heatmap: of size (h,w)
    """

    heatmap_ = np.clip(heatmap * 255, 0, 255)
    heatmap_ = heatmap_.astype(np.uint8)

    return cv2.applyColorMap(heatmap_, cv2.COLORMAP_JET)

def read_image(im_path):
    return np.asarray(Image.open(im_path).convert('RGB'))

def __simple_crop_image_by_color(np_image):
    b, r, g = cv2.split(np_image)

    sc_im = b + 256 * (g + 1) + 256 * 256 * (r + 1) # single channel
    bg_color = 0 + 256 * (0 + 1) + 256 * 256 * (0 + 1)

    labels = measure.label(sc_im, neighbors=4, background=bg_color)
    for region in measure.regionprops(labels):
        coords = np.asarray(region['coords'])

        if [5,5] in coords:
            _im = region['image'].copy().astype(np.uint8) * 255

            x, y, w, h = cv2.boundingRect(np.stack(np.where(_im == 0)[::-1], axis=-1))

            sub_im = np_image[y:y+h, x:x+w, :]
            return sub_im, (x,y,w,h)

    return np_image, None

def crop_bbox(np_image):
    np_image_crop, rect = __simple_crop_image_by_color(np_image)
    out = cv2.copyMakeBorder(np_image_crop, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return out, rect
