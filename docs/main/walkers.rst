.. _walkers:

Walker Environments
*************************

.. important::

   So far, people can only use the framework with MuJoCo models.
   It should be actually quite easy to change the Mimic Environment to derive from a gym environment instead a mujoco gym environment and allow to use our framework with every gym env.




Straight Walker
====================


2-segmented upper body and 3d hip walker
===========================================

TODO: Find a shorter name for this model!	



How to use your own environment?
============================================

.. important::

   Do we need to limit ourselves to walker environments only? 
   Actually not at all! The only thing so far that limits us to walking is the evaluation of the agents performance. When we refactor the code to allow implementing custom evaluation, we've created a easy to use DeepMimic Framework.

1. Create a new class extending the :ref:`MimicEnv <mimic_env>` class.
	a. Override the required methods (marked as `methods to override` at the end of the :class:`MimicEnv` implementation.)
2. Register the class in `mujoco/gym_mimic_envs/__init__.py` analogously to other environments.
3. Import the class in `mujoco/gym_mimic_envs/mujoco/__init__.py`.
4. In case some problems occure, please refer to the official documentation from OpenAI here: https://github.com/openai/gym/blob/master/docs/creating-environments.md 