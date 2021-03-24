import os, sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import cv2
from skimage.measure import label
from warp_image import warp_images
from augment.core import CoreTransform
import matplotlib.pyplot as plt
def imshow(im):
    plt.imshow(im)
    plt.show()

def _get_regular_grid(image, points_per_dim, min_ratio=0.15, max_ratio=0.85):
    min_ratio = np.clip(min_ratio, 0., 0.2)
    max_ratio = np.clip(max_ratio, 0.8, 1.)
    nrows, ncols = image.shape[0], image.shape[1]
    rows = np.linspace(int(min_ratio * nrows), int((nrows) * max_ratio) - 1, points_per_dim)
    cols = np.linspace(int(min_ratio * ncols), int((ncols) * max_ratio) - 1, points_per_dim)
    rows, cols = np.meshgrid(rows, cols)
    return np.dstack([cols.flat, rows.flat])[0]

def _generate_random_vectors(image, src_points, scale):
    dst_pts = src_points + np.random.uniform(-scale, scale, src_points.shape)
    return dst_pts

def _generate_fix_coordinate(input_image, points_per_dim, scale=None):
    """
    Generate the source * target points for TPS algorithm
    Args:
        input_image:
        points_per_dim:
        scale:

    Returns:

    """
    h, w = input_image.shape[:2]
    scale = scale if scale is not None else 0.1 * w
    src_points = _get_regular_grid(input_image, points_per_dim)
    tgt_points = _generate_random_vectors(input_image, src_points, scale)

    return src_points, tgt_points

def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=True, interpolation_order=1):
    width, height = image.shape[:2]
    if keep_corners:
        corner_points = np.array(
            [[0, 0], [0, width], [height, 0], [height, width]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    out = warp_images(src_points, dst_points,
                      np.moveaxis(image, 2, 0),
                      (0, 0, width - 1, height - 1), interpolation_order=interpolation_order)
    return np.moveaxis(np.array(out), 0, 2)


def tps_warp(image, points_per_dim, scale):
    width, height = image.shape[:2]
    src = _get_regular_grid(image, points_per_dim=points_per_dim)
    dst = _generate_random_vectors(image, src, scale=scale*width)
    out = _thin_plate_spline_warp(image, src, dst)
    return out

def tps_warp_2(image, dst, src, interpolation_order=1):
    out = _thin_plate_spline_warp(image, src, dst, interpolation_order=interpolation_order)
    return out

def find_correspondence_vertex(point_loc, org_mask, dst_mask):
    x, y = point_loc
    x = int(x); y = int(y)

    org_b, org_g, org_r = org_mask[:,:,0], org_mask[:,:,1], org_mask[:,:,2]
    org_single_mask = org_b.astype(np.int64) * 255 * 255 + org_g.astype(np.int64) * 255 + org_r.astype(np.int64)

    dst_b, dst_g, dst_r = dst_mask[:,:,0], dst_mask[:,:,1], dst_mask[:,:,2]
    dst_single_mask = dst_b.astype(np.int64) * 255 * 255 + dst_g.astype(np.int64) * 255 + dst_r.astype(np.int64)

    org_color = org_single_mask[y,x]
    new_ys, new_xs =  np.where(dst_single_mask == org_color)

    n_point = len(new_ys)
    if n_point <= 0:
        print ('can not find correspondence key-points for loc.(x-%d,y-%d)' % (x, y))
        return tuple([-1., -1.])
    else:
        new_point_loc = tuple([new_xs[n_point//2], new_ys[n_point//2]])
        return new_point_loc

class TPSTransform(CoreTransform):
    def __init__(self, version):
        super(TPSTransform, self).__init__()
        self.version = version

    def check_valid(self, input_image, pose_xy_coords, output_size):
        h, w = input_image.shape[:2]
        before_mask = np.zeros(shape=(h,w), dtype=np.uint8)

        g_count = 1
        for x, y in pose_xy_coords:
            x = int(x); y = int(y)

            if 0 <= x < w and 0 <= y < h: # if the pose is label outside of image
                before_mask[y, x] = g_count
                cv2.circle(before_mask, (x, y), radius=2, color=g_count, thickness=2)
                g_count += 1

        before_mask = np.stack([before_mask, before_mask, before_mask], axis=-1)
        after_mask  = self.transform_image(before_mask, output_size, interpolation_mode='nearest')
        after_mask  = after_mask[:,:,0]

        n_unique_numbers = np.unique(after_mask.flatten())
        if (len(n_unique_numbers) - 1) != len(pose_xy_coords):
            return False

        return True

    def set_random_parameters(self, input_image, points_per_dim=3, scale_factor=0.1, **kwargs):
        h, w = input_image.shape[:2]
        src_points = _get_regular_grid(input_image, points_per_dim)
        dst_points = _generate_random_vectors(input_image, src_points, scale=scale_factor * min(w,h))
        arugments = {
            'points_per_dim': points_per_dim,
            'scale': scale_factor * min(w,h),
            'src_points': src_points,
            'dst_points': dst_points
        }

        self.params = arugments

    def get_random_parameters(self):
        return self.params

    def transform_coordinate(self, xy_coords, input_image, output_size, interpolation_mode, **kwargs):
        # build image mask
        h, w = input_image.shape[:2]
        mask = np.zeros(shape=(h,w,3), dtype=np.uint8)

        # draw mask
        g_id = 1
        for x, y in xy_coords:
            x = int(x)
            y = int(y)
            mask[y, x] = [g_id, g_id, g_id]
            cv2.circle(mask, (x,y), radius=2, color=(g_id, g_id, g_id), thickness=2)

            g_id += 1

        out_mask = self.transform_image(mask, output_size, 'nearest')

        #
        output_points = []
        for x, y in xy_coords:
            new_x, new_y = find_correspondence_vertex(point_loc=(x,y), org_mask=mask, dst_mask=out_mask)
            output_points += [(new_x, new_y)]

        return np.array(output_points)

    def transform_image(self, input_image, output_size, interpolation_mode, **kwargs):
        tps_params = self.params
        src_points = tps_params['src_points']
        dst_points = tps_params['dst_points']

        if interpolation_mode == 'linear':
            i_order = 1
            i_resize_type = cv2.INTER_CUBIC
        elif interpolation_mode == 'nearest':
            i_order = 0
            i_resize_type = cv2.INTER_NEAREST
        else:
            raise Exception('not support for interpolation mode: %s' % str(interpolation_mode))

        out = tps_warp_2(input_image, dst_points, src_points, interpolation_order=i_order)

        in_h, in_w = input_image.shape[:2]
        ou_w, ou_h = output_size

        ratio_x = in_w / ou_w
        ratio_y = in_h / ou_h
        ratio_max = max([ratio_x, ratio_y])
        scale_factor = 1. / ratio_max

        ou_est_img = cv2.resize(out, interpolation=i_resize_type,
                                dsize=(int(scale_factor * in_w), int(scale_factor * in_h)))
        ou_est_h, ou_est_w = ou_est_img.shape[:2]

        pad_h = ou_h - ou_est_h
        pad_w = ou_w - ou_est_w

        if pad_h > 0:
            ou_est_img = cv2.copyMakeBorder(ou_est_img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=0)

        if pad_w > 0:
            ou_est_img = cv2.copyMakeBorder(ou_est_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        #
        return ou_est_img

def augment_tps(np_image, xy_keypoint_coords):
    tps_transform = TPSTransform(version='tps')
    h, w = np_image.shape[:2]

    try_count = 0
    max_try = 2
    while try_count < max_try:
        tps_transform.set_random_parameters(input_image=np_image, points_per_dim=3, scale_factor=0.1)

        _is_valid = tps_transform.check_valid(input_image=np_image, pose_xy_coords=xy_keypoint_coords,
                                              output_size=(w,h))
        if _is_valid:
            break
        else:
            tps_transform.params = None

        try_count += 1

    if tps_transform.params is not None:
        new_image   = tps_transform.transform_image(np_image, output_size=(w,h), interpolation_mode='linear')
        new_coords  = tps_transform.transform_coordinate(xy_keypoint_coords, input_image=np_image, output_size=(w,h),
                                                        interpolation_mode='nearest')
        return (new_image, new_coords)

    else:
        return None

import glob, json
from copy import deepcopy
def create_more_dir_with_tps(input_dir, output_dir):
    dir_names = os.listdir(input_dir)
    for dir_name in dir_names:
        _in_dir = os.path.join(input_dir, dir_name)
        if not os.path.isdir(_in_dir): continue

        json_dir = os.path.join(_in_dir, 'labels_v1')
        image_dir = os.path.join(_in_dir, 'images')
        json_out_dir = os.path.join(output_dir, dir_name, 'labels_v1'); os.makedirs(json_out_dir, exist_ok=True)
        image_out_dir = os.path.join(output_dir, dir_name, 'images'); os.makedirs(image_out_dir, exist_ok=True)

        all_image_paths =   list(glob.glob(os.path.join(image_dir, '*.png'))) + \
                            list(glob.glob(os.path.join(image_dir, '*.jpg')))

        for image_path in all_image_paths:
            print (image_path)
            bname = os.path.basename(image_path)
            bname_wo_ext = os.path.splitext(bname)[0]

            corres_json_path = os.path.join(json_dir, "%s.json" % bname_wo_ext)
            if os.path.exists(corres_json_path):
                #
                im = cv2.imread(image_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                #
                json_data = json.load(open(corres_json_path, 'r'))
                key_point_coords = [x['points'][0] for x in json_data['shapes']]
                key_point_names  = [x['label'] for x in json_data['shapes']]

                new_result = augment_tps(im, key_point_coords)
                if new_result is None:
                    print ('> Can not TPS for %s -> skip' % bname_wo_ext)
                    continue

                new_image, new_coords = new_result
                assert len(new_coords) == len(key_point_coords)

                out_json_data = deepcopy(json_data)
                for _i, (x, y) in enumerate(new_coords):
                    x = int(x)
                    y = int(y)
                    out_json_data['shapes'][_i]['points'] = [[x, y]]
                    #cv2.circle(new_image, center=(x,y), radius=2, color=(255,0,0), thickness=2)

                #vis_image = np.concatenate([im, new_image], axis=1)
                #imshow(vis_image)
                #exit()

                out_image_path = os.path.join(image_out_dir, '%s.png' % bname_wo_ext)
                out_json_path = os.path.join(json_out_dir, '%s.json' % bname_wo_ext)

                cv2.imwrite(out_image_path, cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
                json.dump(out_json_data, open(out_json_path, 'w'), indent=4)


import _pickle as cPickle
def test_pickle(pickle_path):
    error_data = cPickle.load(open(pickle_path, 'rb'))

    data_numpy = error_data['data_numpy']
    params = error_data['params']
    h, w = data_numpy.shape[:2]
    # params['src_points'][:,0] = np.clip(params['src_points'][:,0], a_min=5., a_max=w-5)
    # params['src_points'][:,1] = np.clip(params['src_points'][:,1], a_min=5., a_max=h-5)
    # params['dst_points'][:, 0] = np.clip(params['dst_points'][:, 0], a_min=5., a_max=w-5)
    # params['dst_points'][:, 1] = np.clip(params['dst_points'][:, 1], a_min=5., a_max=h-5)


    tps_transform = TPSTransform(version='tps')
    tps_transform.params = params

    output_size = (512, 768)  # (w,h)
    expected_output_size = (data_numpy.shape[1], data_numpy.shape[0])
    augment_image = tps_transform.transform_image(input_image=data_numpy, output_size=output_size, interpolation_mode='linear')

    imshow(data_numpy)
    imshow(augment_image)

    fixed_src_points = [error_data['point'][0:2].astype(np.int).tolist()] #[[225, 218]]
    is_valid = tps_transform.check_valid(data_numpy, fixed_src_points, output_size=output_size)
    print ('is_valid:', is_valid)
    augment_keypoints = tps_transform.transform_coordinate(xy_coords=fixed_src_points, input_image=data_numpy,
                                                           output_size=output_size, interpolation_mode='nearest')

    print ('src:', fixed_src_points)
    print ('dst:', augment_keypoints)



if __name__ == '__main__':
    input_dir  = "/home/kan/data/geek_data/AnimeDrawing_Geek/Combine"
    output_dir = "/home/kan/data/geek_data/AnimeDrawing_Geek/Combine_Augment"

    create_more_dir_with_tps(input_dir, output_dir)
    exit()
    #

    import cv2
    import numpy as np

    pickle_path = "/home/kan/Desktop/19032021_11:57:00.pkl"
    #test_pickle(pickle_path=pickle_path)

    image_path = "/home/kan/Desktop/cinnamon/CharacterGAN/datasets/hor02_037_C/C/output/C0001.png"
    output_size = (512, 768)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]

    fixed_src_points = np.array([[500, 418]]) # xy coordinates

    #
    tps_transform = TPSTransform(version='tps')
    tps_transform.set_random_parameters(input_image=image, points_per_dim=3, scale_factor=0.1)

    augment_image = tps_transform.transform_image(input_image=image, output_size=output_size, interpolation_mode='linear')
    print ('original,', image.shape)
    imshow(image)

    print ('augment,', augment_image.shape)
    imshow(augment_image)

    augment_keypoints = tps_transform.transform_coordinate(xy_coords=fixed_src_points, input_image=image, output_size=output_size, interpolation_mode='nearest')
    print (fixed_src_points)
    print (augment_keypoints)
    for point in fixed_src_points:
        x, y = point
        cv2.circle(image, (x,y), radius=2, color=(255,0,0), thickness=2)
    imshow(image)

    for point in augment_keypoints:
        x, y = point
        cv2.circle(augment_image, (x, y), radius=2, color=(255, 0, 0), thickness=2)
    imshow(augment_image)
