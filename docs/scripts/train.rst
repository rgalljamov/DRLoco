
.. _train:

Training an agent
**********************

Here, we first explain the script that prepares and start the training step by step. Thereafter, we explain the training procedure itself. 

Training Script
========================

#. Create the path for storing the model and all related info like performance metrics and model parameters. 
	
	#. The path is constructed in ``config/hypers.py`` and reflects the choice of the environment and some specific hyperparameters

#. Create the MuJoCo Gym environment

	#. We create multiple parallel instances of the environment specified by the environment id (string) in ``common/config.py``. The number of parallel environments is specified in ``common/hypers.py``.

	#. Each environment is wrapped by a :mod:`Monitor`, a gym wrapper we built to monitor joint kinematics, kinetics as well as performance metrics

	#. Each environment is then wrapped by a :mod:`SubprocVecEnv` from SB3 which creates the parallel environments

	#. Finally, a :mod:`VecNormalize` wrapper from SB3 is used to maintain a running mean and running standard deviation of the observations (dimension-wise) and the return for normalization.

	#. In summary, we get ``VecNormalize(SubprocVecEnv(MonitorWrapper(MimicEnv), n_envs))``

#. Use hyperparameters to create schedules and prepare the configuration dictionary for the network architectures

#. Create the model: a PPO agent with the specified hyperparameters and configurations

	#. In this step, also Tensorboard (TB) is launched. The TB logs go into a dedicated subfolder (``tb_logs/``) in the model folder. SB3 logs multiple PPO specific metrics like losses or entropy. In addition, we specify multiple metrics in ``common/callback.py`` that are logged to TB.

#. Logging: start W&B if necessary, print infos about the training procedure and model to the console.

#. Save initial model. The model checkpoint at this timestep is called 'init'.

#. Start the training. This single command (starting with ``model.learn(...)``) starts and executes the whole training procedure as described below.

#. After training ends, save the final model checkpoint named 'final'.

#. Close the environment, which is important to avoid multiple problems. 
	
	.. warning:: Always close the environment before ending a script in which a gym environment was instantiated.

#. At this point, we used to evaluate the model and record videos of the trained agent. After the migration from SB2 to SB3 however, the recording of the video on a remote server broke. It still should work, when training on a laptop or PC connected to a display. Also, the performance evaluation in ``eval.py`` might no longer work. Use with caution.

	.. warning:: There are two different sets of evaluation! 

	   * One is performed *during* training every N amounts of steps. This evaluation procedure is defined in ``common/callback.py``. Here, we evaluate the walking performance of the agent and track it to TB. 

	   
	   * The other evaluation used to be performed at the end of the training to evaluate the performance of the agent. It is defined in ``eval.py`` and is broken in the moment. To evaluate your agent after the training, please use/extend ``run.py``.


Training Procedure
====================

The training procedure is almost fully managed internally by Stable Baselines 3 (SB3). Our only touchpoint with it is the callback implemented in $common/callback.py$. The callback allows us to execute custom code before (:func:`_on_training_start`) and after the training (:func:`_on_training_end` as well as on every step taken in the environment (:func:`_on_step`). The former two methods are used to setup and close the :mod:`SummaryWriter` that is used to log custom metrics to Tensorboard. The most interesting inteactions with the trained agent are happening in the latter function. Hereafter, we first outline the overall training procedure as implemented in SB3 and then explain our interactions through the callback.



SB3 Training Loop
----------------------

#. Get the state observations *s* of the n parallel environments.

#. Pass the observations through the value function network and the policy network to estimate the state values and get an action *a* for each of the parallel environments.

#. Clip the actions to a specified range.

#. Execute the actions in all parallel environments, and get the rewards *r*, new observations *s'*.

#. Store the experience tuples *(s,a,r,s')* in a batch.

#. Continue steps above until the batch is full, i.e. enough experiences are collected to update the policy. 

	.. note:: Collecting enough experiences with the same policy until the batch is full is often referred to as a policy rollout.

#. Pause the training to update the policy. Sample minibatches of experiences from the batch and perform a policy update. Repeat this step for *noptepochs*.

#. Repeat steps above with the updated policy until the specified amount of steps (*mio_samples*) were collected and the training can be stopped.



Our Interactions with the Training via the Callback: Training Evaluation
-----------------------------------------------------------------------------

:func:`_on_step` is used to log training performance to TB and Weights & Biases every 100 steps (:mod:`self.skipped_steps`). In addition, it performs a strong model/walking-performance evalution every 400k (:mod:`EVAL_INTERVAL`) steps. The evaluation is described in more detail in the following.

To evaluate the walking performance, every 400k steps, we 

* pause the training and save the current model
* load the current model with the corresponding environment in a new thread
* evaluate the model for 20 episodes and 
* record multiple metrics like the *walked distance* (before falling or episode end), average speed, average episode duration before falling etc. These metrics are then all uploaded to TB and W&B in the *_det_eval* section.

