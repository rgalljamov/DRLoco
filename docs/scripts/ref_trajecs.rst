.. _ref_trajecs:

Reference Trajectories
************************

* TODO: mention calculation of desired walking speed.

Overview
=============

The reference trajectories (aka. reference motions or reference data) are at the hearth of the :ref:`DeepMimic Approach <deepmim>`. They provide the motion to imitate in form of a matrix :math:`R` with the rows representing the joints and the corresponding joint kinematics being stored in the columns. 

Details
=============

The base class :class:`BaseReferenceTrajectories` defines the interface between the :ref:`Mimic Environment <mimic_env>` and the reference trajectories. It maintains a position variable ``_pos`` pointing at the current position on the reference trajectories. This position should be incremented during each :func:`step` call in the environment using :func:`next`. The increment is automatically calculated from the sampling rate of the reference motion ``sample_freq`` and the control frequency ``control_freq``, both specified during the initialization of the base class.


How to use your own reference trajectories?
============================================

To use your own reference trajectories, you have to: 
 
 #. define a new class :class:`YourOwnReferenceTrajectories` which extends the base class
 #. overwrite the following methods

 	a. :func:`_load_ref_trajecs` to load your reference motion as a numpy matrix in the expected shape
 	b. :func:`_get_COM_Z_pos_index` to specify the row with the COM trajectories in the vertical direction
 	c. :func:`get_desired_walking_velocity_vector` to provide a desired COM velocity vector during training, derived from the reference data


**Example for using custom reference trajectories:** ``scripts/ref_trajecs/loco3d_trajecs.py``.


.. Code Documentation
.. =====================

.. .. automodule:: scripts.ref_trajecs.base_ref_trajecs


.. .. autoclass:: BaseReferenceTrajectories
..    :members: