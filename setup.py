from setuptools import setup

requirements = open('requirements.txt').read().splitlines()

setup(name='animkp',
      description='Anime Keypoint Estimation .',
      version='0.1.0',

      packages=['pose_estimator', 'pose_estimator.wrapper', 'pose_estimator.wrapper.pose_anime'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)