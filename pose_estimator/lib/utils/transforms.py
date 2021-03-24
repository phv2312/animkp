# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    shift=np.array([0, 0], dtype=np.float32)
    scale_tmp = scale * 200.0 # return to the original height & width
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    # #TODO: visualize the correspondence of source & dist
    # org_w, org_h = scale_tmp
    #
    # org_h = int(org_h)
    # org_w = int(org_w)
    # dst_h = int(dst_h)
    # dst_w = int(dst_w)
    #
    # src_image = np.ones(shape=(org_h, org_w, 3), dtype=np.uint8) * 255
    # dst_image = np.ones(shape=(dst_h, dst_w, 3), dtype=np.uint8) * 255
    #
    # #
    # max_h = max(org_h, dst_h)
    # if max_h != org_h:
    #     src_image = cv2.copyMakeBorder(src_image, 0, max_h - org_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # else:
    #     dst_image = cv2.copyMakeBorder(dst_image, 0, max_h - dst_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    #
    # offset = org_w
    # vis_image = np.concatenate([src_image, dst_image], axis=1)
    #
    # for _i, (src_point, dst_point) in enumerate(zip(src, dst)):
    #     print ('pairing: src with tgt', src_point, dst_point)
    #
    #     sx, sy = src_point
    #     sx = int(sx); sy = int(sy)
    #
    #     dx, dy = dst_point
    #     dx = int(dx); dy = int(dy)
    #
    #     cv2.putText(vis_image, str(_i), (sx, sy), cv2.FONT_HERSHEY_PLAIN, 1.1, (0,255,0), 1)
    #     cv2.circle(vis_image, (sx, sy), radius=3, color=(255,0,0), thickness=2)
    #     cv2.circle(vis_image, (dx + offset, dy), radius=3, color=(255,0,0), thickness=2)
    #     cv2.line(vis_image, (sx, sy), (dx + offset, dy), color=(0,0,255), thickness=2)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(vis_image)
    # plt.show()

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img
