import numpy as np
from drloco.ref_trajecs.straight_walk_trajecs import StraightWalkingTrajectories

class StraightWalking3dHipTrajectories(StraightWalkingTrajectories):
    def __init__(self,  qpos_indices, q_vel_indices, adaptations={}):
        super(StraightWalking3dHipTrajectories, self).__init__(qpos_indices, q_vel_indices, adaptations)

    def get_qpos(self):
        qpos = super().get_qpos()
        # add hip traversal value
        qpos = np.concatenate([qpos[:8], [-0.05], qpos[8:12], [0.05], qpos[12:]])
        return qpos

    def get_qvel(self):
        qvel = super().get_qvel()
        print('Adding hip traversal values to qvel!')

        # add hip traversal value
        qvel = np.concatenate([qvel[:8], [0], qvel[8:12], [0], qvel[12:]])
        return qvel