.. DRLoco documentation master file, created by
   sphinx-quickstart on Mon Jun 21 10:45:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DRLoco - DeepMimic meets MuJoCo for Locomotion
****************************************************

.. note::
   **DRLoco = DeepMimic(MuJoCo, StableBaselines3)**

.. .. warning::

..    Do we need to limit ourselves to walker environments only? 
..    Actually not at all! The only thing so far that limits us to walking is the evaluation of the agents performance. When we refactor the code to allow implementing custom evaluation, we've created a easy to use DeepMimic Framework.
   

This repository for you, when:
===============================

 * you have a bipedal robot and need a controller to generate stable human-like walking
 * you have a MuJoCo model of a bipedal robot and you want it to learn to walk like a human
 * you have any kind of a legged robot and a set of reference trajectories (e.g. motion capturing) and you want a robust controller to let your robot imitate the reference motion.
 * you want to try out the DeepMimic Approach with minimal effort
 * you need high quality mocap data for Imitation Learning



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


