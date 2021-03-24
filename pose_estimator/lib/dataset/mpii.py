# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict
import cv2
import numpy as np
from scipy.io import loadmat, savemat
import random
from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
def imgshow(im):
    plt.imshow(im)
    plt.show()

def draw_keypoints(image_path, joints_3d, joints_3d_vis):
    cv2_image = cv2.imread(image_path)

    if cv2_image is not None:

        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        for joint_3d, join_3d_vis in zip(joints_3d, joints_3d_vis):
            x, y = joint_3d[:2]
            x = int(x);
            y = int(y)

            cv2.circle(cv2_image, (x, y), radius=3, color=(255, 0, 0), thickness=2)

    return cv2_image

class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.flip_pairs = [[14,15],[10,13],[9,12],[8,11],[4,7],[2,5],[3,6]] # specified for animeDrawing dataset
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
        self.aspect_ratio = .4 # don't care

        self.upper_body_ids = (0,1,2,3,4,5,6,10,14,15,16,17)
        self.lower_body_ids = (7,8,9,11,12,13)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']
            base_name  = os.path.splitext(image_name)[0]

            is_found = False
            for ext in ['.jpg', '.png']:
                image_name_ = "%s%s" % (base_name, ext)
                full_path = os.path.join(self.root, 'images', image_name_)
                if os.path.exists(full_path):
                    image_name = image_name_
                    is_found = True
                    break

            if is_found == False:
                raise Exception('can not file path for image %s ' % image_name)

            c = np.array(a['center'], dtype=np.float)
            #what is scale factor used for ? Ans: scale = original_height / 200

            s = np.array(a['scale'], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            # c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            # test viewing the original image
            image_path = os.path.join(self.root, 'images', image_name)
            if False:
                print ('test viewing original image ...')
                cv2_image  = draw_keypoints(image_path, joints_3d, joints_3d_vis)
                imgshow(cv2_image)

            if False:
                print ('test new transformation')
                from lib.augment.affine.affine_warp import AffineTransform
                from lib.augment.tps.tps_warp import TPSTransform

                # initialize image
                data_numpy = cv2.imread(
                    image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )

                if self.color_rgb:
                    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

                # set up parameter for augment
                use_augment = 'tps'
                transform_model = None
                if random.uniform(0, 1.) < self.tps_prob:
                    use_augment = 'tps'
                    transform_model = TPSTransform(version='tps')
                    transform_model.set_random_parameters(input_image=data_numpy,
                                                          points_per_dim=self.tps_params['points_per_dim'],
                                                          scale_factor=self.tps_params['scale_factor'])



                else:
                    use_augment = 'affine'
                    transform_model = AffineTransform(version='affine')
                    transform_model.set_random_parameters(output_size=self.image_size, c=c, s=s, r=0.)

                #print ('run transform:', use_augment)
                out_image = transform_model.transform_image(input_image=data_numpy, output_size=self.image_size, interpolation_mode='linear')

                # transform key-points
                for i, (joint, joint_vis) in enumerate(zip(joints_3d, joints_3d_vis)):
                    #print('joints:', joint)
                    #print('joints_vis:', joint_vis)
                    if joint_vis[0] > 0.0:
                        joint_ = transform_model.transform_coordinate(xy_coords=[joint[0:2].tolist()], input_image=data_numpy,
                                                                      output_size=self.image_size, interpolation_mode='nearest') #affine_transform(joint[0:2], trans)

                        x, y = joint_[0]
                        x = int(x)
                        y = int(y)

                        #print('x,y coordinates:', x, y)
                        cv2.circle(out_image, (x, y), radius=2, color=(255, 0, 0), thickness=2)

                #
                vis_org_image = data_numpy.copy()
                vis_org_image = cv2.resize(vis_org_image, dsize=tuple(self.image_size))
                imgshow(np.concatenate([vis_org_image, out_image], axis=1))



            # test half-body transform
            if False:
                print ('test half-body transform.')
                from .JointsDataset import fliplr_joints, get_affine_transform, affine_transform

                data_numpy = cv2.imread(
                    image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )

                if self.color_rgb:
                    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

                joints = joints_3d
                joints_vis = joints_3d_vis
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

                sf = self.scale_factor
                rf = self.rotation_factor
                r = 0.
                # s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                # r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                #     if random.random() <= 0.6 else 0

                trans = get_affine_transform(c, s, r, self.image_size)
                input = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR)

                # transform key-points
                for i, (joint, joint_vis) in enumerate(zip(joints_3d, joints_3d_vis)):
                    print('joints:', joint)
                    print('joints_vis:', joint_vis)
                    if joint_vis[0] > 0.0:
                        joint_ = affine_transform(joint[0:2], trans)

                        x, y = joint_
                        x = int(x)
                        y = int(y)

                        print('x,y coordinates:', x, y)
                        cv2.circle(input, (x, y), radius=2, color=(255, 0, 0), thickness=2)

                c_joint_ = affine_transform(c_half_body, trans)
                x, y = c_joint_; x = int(x); y = int(y)
                cv2.circle(input, (x, y), radius=10, color=(255, 0, 255), thickness=8)

                imgshow(input)

            # test flip pair
            if False:
                r = 0

                print ('test viewing flipped image ...')
                from .JointsDataset import fliplr_joints, get_affine_transform, affine_transform
                self.flip_pairs = [[0, 1]]
                data_numpy = cv2.imread(
                    image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )

                if self.color_rgb:
                    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

                original_h, original_w = data_numpy.shape[:2]
                # use this function because it already wrap all of the augmentation

                trans = get_affine_transform(c, s, r, self.image_size)
                input = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR)

                # transform key-points
                for i, (joint, joint_vis) in enumerate(zip(joints_3d, joints_3d_vis)):
                    print ('joints:', joint)
                    print ('joints_vis:', joint_vis)
                    if joint_vis[0] > 0.0:
                        joint_ = affine_transform(joint[0:2], trans)

                        x, y = joint_
                        x = int(x)
                        y = int(y)

                        print ('x,y coordinates:', x, y)
                        cv2.circle(input, (x,y), radius=2, color=(255,0,0), thickness=2)

                imgshow(input)

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']


