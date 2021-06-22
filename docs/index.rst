.. DRLoco documentation master file, created by
   sphinx-quickstart on Mon Jun 21 10:45:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DRLoco - DeepMimic meets MuJoCo for Locomotion
****************************************************
.. important::
	Stable-Baselines3 + MuJoCo + DeepMimic = DRLoco

.. warning::

   Do we need to limit ourselves to walker environments only? 
   Actually not at all! The only thing so far that limits us to walking is the evaluation of the agents performance. When we refactor the code to allow implementing custom evaluation, we've created a easy to use DeepMimic Framework.


How to read this documentation?
=================================
There might be something worth explaining.


TOC
====

.. toctree::
   :maxdepth: 2
   
   main/overview
   main/install
   scripts/mimic_env
   scripts/ref_trajecs
   main/mocaps
   main/walkers
   main/models
   scripts/config
   scripts/monitoring

.. toctree::
   :maxdepth: 2
   :caption: Scripts
   
   scripts/train
   scripts/run
   scripts/eval

.. toctree::
   :maxdepth: 2
   :caption: Background

   background/deepmim	


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


