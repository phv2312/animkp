import sys, os
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', '..', 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
parent_path = os.path.join(this_dir, '..', '..')
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import cv2
import glob
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from pose_estimator.lib import models
from pose_estimator.lib import dataset
from pose_estimator.lib.config import cfg
from pose_estimator.lib.utils.utils import create_logger
from pose_estimator.lib.utils.transforms import get_affine_transform, flip_back, affine_transform
from pose_estimator.lib.core.inference import get_final_preds, get_max_preds


from pose_estimator.wrapper.pose_anime.pose_anime_utils import FakedArugmentPasser, _update_config, heatmap2image, draw_kp_w_im, crop_bbox, read_image, imgshow
from pose_estimator.wrapper.pose_anime.constants import joint_labels_dct, joint_labels, joint_pair, flip_pairs

class PoseAnimeInference:
    def __init__(self, config_path, weight_path, use_gpu=True):
        self.args = FakedArugmentPasser(config_path, weight_path)
        _update_config(cfg, self.args)

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        """
        Model & Transforms
        """
        self.model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False
        )

        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
#         if use_gpu and not torch.cuda.is_available():
#             print('>>> Enable GPU but <NO GPU FOUND>. Turn into cpu instead ...')
#         else:
#             print('>>> Enable GPU')

        self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def _conduct_identity_transform(self, input_image, scale_factor=1.25):
        height, width = input_image.shape[:2]

        c = [width / 2, height / 2]
        s = height / 200.
        r = 0

        c = np.array(c, dtype=np.float)
        s = np.array([width / 200., height / 200.], dtype=np.float)

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s[1]
            s = s * scale_factor

        trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
        inv_trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE, inv=1)

        return trans, inv_trans

    def __single_model_predict(self, input):
        tensor_input = self.transforms(input).unsqueeze(0).cuda()
        outputs = self.model(tensor_input)

        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        return output

    def __resize_heatmap(self, heatmap_list, org_size, inv_trans):
        heatmap_list_orgsize = []

        for _i, heatmap in enumerate(heatmap_list):
            heatmap_imsize  = cv2.resize(heatmap, dsize=tuple(cfg.MODEL.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            heatmap_orgsize = cv2.warpAffine(heatmap_imsize, inv_trans, org_size, flags=cv2.INTER_LINEAR)
            heatmap_list_orgsize += [heatmap_orgsize]

        return heatmap_list_orgsize

    def process(self, input_image, use_flip=False, threshold=0.6, postprocess=True):
        if type(input_image) is str:
            input_image = np.asarray(Image.open(input_image))

        height, width = input_image.shape[:2]
        with torch.no_grad():
            trans, inv_trans = self._conduct_identity_transform(input_image)
            input = cv2.warpAffine(input_image, trans, tuple(cfg.MODEL.IMAGE_SIZE), flags=cv2.INTER_LINEAR)
            output = self.__single_model_predict(input)

            if use_flip:
                input_flipped = np.flip(input, axis=1).copy()
                output_flipped = self.__single_model_predict(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
            else:
                output_flipped = None

            if output_flipped is not None:
                output = (output + output_flipped) * 0.5

            # get heat maps if original input size...
            heatmap_orgsize_list = self.__resize_heatmap(heatmap_list=output.cpu().numpy()[0], org_size=(width, height),
                                                         inv_trans=inv_trans)

        # get max point value per heatmap (joints)
        output_np = output.clone().cpu().numpy() # (batch, n_joint, height, width)
        hm_height, hm_width = output_np[0][0].shape

        preds, maxvals = get_max_preds(output_np)
        preds   = preds[0]
        maxvals = maxvals[0]

        if postprocess:
            import math
            for p in range(preds.shape[0]):
                hm = output_np[0][p]
                px = int(math.floor(preds[p][0] + 0.5))
                py = int(math.floor(preds[p][1] + 0.5))
                if 1 < px < hm_width-1 and 1 < py < hm_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    preds[p] += np.sign(diff) * .1

        # results
        results_dct = {}
        vis_image_w_hms = []
        vis_images = []
        for idx, ((x,y), maxval) in enumerate(zip(preds, maxvals)):
            if maxval < threshold: continue

            point_imsize  = np.array([x * 4, y * 4])
            point_orgsize = affine_transform(point_imsize, inv_trans)
            point_orgsize = tuple(point_orgsize.astype(np.int).tolist())

            # for visualizing hm
            hm = heatmap2image(heatmap_orgsize_list[idx])
            vis_image_w_hm = 0.4 * hm + 0.6 * input_image.copy()
            vis_image_w_hm = np.clip(vis_image_w_hm, 0, 255).astype(np.uint8)
            cv2.putText(vis_image_w_hm, "name: %s_maxval: %.3f" % (joint_labels[idx], maxval), (25, 25),
                        cv2.FONT_HERSHEY_TRIPLEX, .6, color=(255,255,255), thickness=1)

            # for saving result
            results_dct[joint_labels[idx]] = (point_orgsize, heatmap_orgsize_list[idx])
            vis_image_w_hms += [vis_image_w_hm]

        im_w_kp = draw_kp_w_im(input_image, results_dct, joint_pair, radius=width//230, thickness=width//230)

        vis_image = np.concatenate([im_w_kp] + vis_image_w_hms, axis=1)
        return results_dct, vis_image

def test_dir(input_dir, output_dir, from_g=True):
    os.makedirs(output_dir, exist_ok=True)
    assert os.path.isdir(input_dir), 'dir %s not existed ...' % input_dir

    config_path = "/home/kan/Desktop/cinnamon/kp_estimation/keypoint_estimation/experiments/hor01/hrnet/w48_384x288_adam_lr1e-3.yaml"
    weight_path = "/home/kan/Desktop/final_state.pth"

    pose_model = PoseAnimeInference(config_path, weight_path, use_gpu=True)
    im_paths = list(glob.glob(os.path.join(input_dir, '*.jpg'))) + list(glob.glob(os.path.join(input_dir, '*.png')))

    for im_path in im_paths:
        print ('processing %s ...' % im_path)
        bname = os.path.basename(im_path)

        if from_g:
            np_image = crop_bbox(read_image(im_path))  # for g only
        else:
            np_image = read_image(im_path)

        results_dct, vis_image = pose_model.process(input_image=np_image, use_flip=True, threshold=0.3,
                                                    postprocess=True)

        output_path = os.path.join(output_dir, bname)
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

def test_single(im_path, from_g=True):
    assert os.path.exists(im_path), 'file %s not existed ...' % im_path

    config_path = ""
    weight_path = ""

    pose_model = PoseAnimeInference(config_path, weight_path, use_gpu=True)
    if from_g:
        np_image, rect = crop_bbox(read_image(im_path)) # for g only
    else:
        np_image = read_image(im_path)

    results_dct, vis_image = pose_model.process(input_image=np_image, use_flip=True, threshold=0.6, postprocess=True)
    imgshow(vis_image)

if __name__ == '__main__':
    pass
