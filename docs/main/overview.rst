
Repository Overview
***********************

.. note::
	**DeepMimic + MuJoCo + Stable-Baselines3 = DRLoco**
	
This repository 

 * implements the :ref:`DeepMimic Framework <deepmim>` 
 * using the `PPO Algorithm <https://spinningup.openai.com/en/latest/algorithms/ppo.html>`_ from the `Stable-Baselines3 repository <https://stable-baselines3.readthedocs.io>`_
 * to obtain (deep reinforcement) learning-based controllers 
 * for :ref:`robots <walkers>` simulated with the `MuJoCo Physics Engine <http://www.mujoco.org/>`_.

 
File Structure
=================

In the following, the structure of the code is presented with a short description of what each individual script and folder is responsible for.

 * ``mocaps/`` contains our :ref:`motion capturing datasets <ref_trajecs>`

 	* Each dataset is explained in detail in :ref:`Reference Trajectories`
 
 * ``docs/`` contains the files needed to build this documentation

 * ``mujoco/`` maintains the MuJoCo environments we train our agents on.

 	*  TODO: check if we really need the whole folder structure (afaiu, it is only needed to register the gym environment... and registration is only necessary to be able to make the env by using it's string id. We might not need that.) **CHECK if we can do it the same way as described in the SB3 DOC here**: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

 * ``models/`` contains a few trained agents organized by the environment they were trained on as well as the made hyperparameter choices. 

 	* .. warning::
 		For each model, we save the PPO model as well as the **corresponding environment**. The environment is necessary, because we're normalizing the observations with a running mean and standard deviation which are maintained in the environment.  


 * $drloco/$ 

 	* 