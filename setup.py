from setuptools import setup

requirements = open('requirements.txt').read().splitlines()

setup(name='animkp',
      description='Anime Keypoint Estimation .',
      version='0.1.0',

      packages=['pose_estimator',
                'pose_estimator.wrapper',
                'pose_estimator.wrapper.pose_anime',
                'pose_estimator.lib',
                'pose_estimator.lib.config',
                'pose_estimator.lib.utils',
                'pose_estimator.lib.core',
                'pose_estimator.lib.models',
                'pose_estimator.lib.dataset',
                'pose_estimator.lib.augment','pose_estimator.lib.augment.affine','pose_estimator.lib.augment.tps',
                'pose_estimator.lib.component',
                'pose_estimator.lib.nms',],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)