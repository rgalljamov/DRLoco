from setuptools import setup

setup(name='gym_mimic_envs',
      version='0.1',
      # required packages for our env to work
      install_requires=['gym', 'mujoco-py']
)