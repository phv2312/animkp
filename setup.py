from setuptools import setup

requirements = open('requirements.txt').read().splitlines()

setup(name='animkp',
      description='Anime Keypoint Estimation .',
      version='0.1.0',

      packages=['pose_estimator'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)