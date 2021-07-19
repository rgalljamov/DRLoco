.. _mimic_env:

Mimic Environment
**********************

File Location: ``mujoco/gym_mimic_envs/mimic_env.py``



Purpose
===========

The MimicEnv is the base class for all our environments. It extends the MujocoEnv for the usage within the :ref:`DeepMimic Framework <deepmim>`. The main function of the class is :ref:`step() <mimic_env_step>` which runs the simulation for the specified amount of frames (cf. *frame_skip*). Within this class, we also load the :ref:`reference trajectories <ref_trajecs>` and use them for :ref:`episode initialization <dmm_rsi>` and :ref:`reward calculation <reward>`. 


Main Functions
================

.. _mimic_env_step:

step(action)
-------------------
It's main purpose is to execute an action in the environment and return the current observations of the environment's state. 

.. note::
   **Distinguish simulation and control frequencies!**
   We're running the MuJoCo simulation at 1kHz (``_sim_freq``).
   As this might be a too high frequency for the policy to deal with,
   we're using the well known trick of applying the same action for multiple (``_frame_skip``) simulation frames resulting in a control frequency of :math:`f_{ctrl} = f_{sim} / \text{frame_skip}`.


.. Automodule TEST
.. --------------------
.. .. automodule:: gym_mimic_envs.mimic_env

.. .. autoclass:: MimicEnv
..    :members: step, get_reward