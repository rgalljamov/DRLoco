
.. _run:

Running an agent
**********************

How do we save agents?
=======================

During the training of an agent, we save multiple checkpoints:

* initial agent: PPO with random networks directly after initialization and before training

* final agent: PPO agent after the end of the training

* agents reaching certain performance thresholds specified in $common/callback.py$.

	* reaching 50, 60, ... 90% of maximum possible imitation reward

	* reaching 50, 60, ... 90% of maximum possible episode return

.. important:
   
   Each agent is always saved with a corresponding environment. This environment has to be loaded together with the agent to run the agent. This is done automatically within this framework. 

   **Details:** We train our agents on mod:`VecNormalize` environment wrappers. These wrappers maintain a running mean and std of the observations (dimension-vise) to normalize the inputs into the policy and value function networks. Therefore, when loading an agent, we also have to wrap the corresponding environment in a VecNormalize object. This is done automatically within this framework. 
   