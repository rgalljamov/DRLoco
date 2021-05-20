'''
Interface for environments using reference trajectories.
'''
import gym, mujoco_py
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from scripts.config import config as cfgl
from scripts.config import hypers as cfg
from scripts.common.utils import log, is_remote, \
    exponential_running_smoothing as smooth, get_torque_ranges
from scripts.ref_trajecs import ReferenceTrajectories as RefTrajecs



# flag if ref trajectories are played back
_play_ref_trajecs = False

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_mujoco_viewer_on_start = True and not is_remote()

# monitor episode and training duration
step_count = 0
ep_dur = 0

class MimicEnv(MujocoEnv, gym.utils.EzPickle):
    """ The base class to derive from to train an environment using the DeepMimic Approach."""
    def __init__(self: MujocoEnv, xml_path, ref_trajecs:RefTrajecs):
        '''@param: self: gym environment class extending the MimicEnv class
           @param: xml_path: path to the mujoco environment XML file
           @param: ref_trajecs: Instance of the ReferenceTrajectory'''

        self.refs = ref_trajecs
        # set simulation and control frequency
        self._sim_freq, self._frame_skip = self.get_sim_freq_and_frameskip()

        # keep the body in the air for testing purposes
        self._FLY = False or cfg.is_mod(cfg.MOD_FLY)
        # when we evaluate a model during or after the training,
        # we might want to weaken ET conditions or monitor and plot data
        self._EVAL_MODEL = False
        # control desired walking speed
        self.FOLLOW_DESIRED_SPEED_PROFILE = False

        # track individual reward components
        self.pos_rew, self.vel_rew, self.com_rew = 0,0,0
        self.mean_epret_smoothed = 0
        # track running mean of the return and use it for ET reward
        self.ep_rews = []

        # initialize Mujoco Environment
        MujocoEnv.__init__(self, xml_path, self._frame_skip)
        # init EzPickle (think it is required to be able to save and load models)
        gym.utils.EzPickle.__init__(self)
        # make sure simulation and control run at the desired frequency
        self.model.opt.timestep = 1 / self._sim_freq
        self.control_freq = self._sim_freq / self._frame_skip
        # sync reference data with the control frequency
        self.refs.set_sampling_frequency(self.control_freq)
        # The motor torque ranges should always be specified in the config file
        # and overwrite the forcerange in the .MJCF file
        self.model.actuator_forcerange[:, :] = get_torque_ranges(*cfgl.PEAK_JOINT_TORQUES)


    def step(self, action):
        # when rendering: pause sim on startup to change rendering speed, camera perspective etc.
        global pause_mujoco_viewer_on_start
        if pause_mujoco_viewer_on_start:
            self._get_viewer('human')._paused = True
            pause_mujoco_viewer_on_start = False

        # monitor episode and training durations
        global step_count, ep_dur
        step_count += 1
        ep_dur += 1

        # hold the agent in the air
        if self._FLY:
            qpos_before = np.copy(self.sim.data.qpos)
            qvel_before = np.copy(self.sim.data.qvel)
            # get current joint angles and velocities
            qpos_set = np.copy(qpos_before)
            qvel_set = np.copy(qvel_before)
            # fix COM position, trunk rotation and corresponding velocities
            qpos_set[[0, 1, 2]] = [0, 1.2, 0]
            qvel_set[[0, 1, 2, ]] = [0, 0, 0]
            self.set_joint_kinematics_in_sim(qpos_set, qvel_set)

        action = self.rescale_actions(action)

        # when we're mirroring the policy (phase based mirroring), mirror the action
        if cfg.is_mod(cfg.MOD_MIRR_POLICY) and self.refs.is_step_left():
            action = self.mirror_action(action)

        # execute simulation with desired action for multiple steps
        self.do_simulation(action, self._frame_skip)

        # increment the current position on the reference trajectories
        self.refs.next()

        # get state observation after simulation step
        obs = self._get_obs()

        # get imitation reward
        reward = self.get_imitation_reward()

        # check if we entered a terminal state
        com_z_pos = self.sim.data.qpos[self._get_COM_indices()[-1]]
        walked_distance = self.sim.data.qpos[0]
        # was max episode duration or max walking distance reached?
        max_eplen_reached = ep_dur >= cfg.ep_dur_max or walked_distance > cfg.max_distance + 0.01
        # terminate the episode?
        done = com_z_pos < 0.5 or max_eplen_reached

        if self.is_evaluation_on():
            done = com_z_pos < 0.5 or max_eplen_reached
        else:
            terminate_early, _, _, _ = self.do_terminate_early()
            done = done or terminate_early
            if done:
                # if episode finished, recalculate the reward
                # to punish falling hard and rewarding reaching episode's end a lot
                reward = self.get_ET_reward(max_eplen_reached, terminate_early)

        # reset episode duration if episode has finished
        if done: ep_dur = 0
        # add alive bonus else
        else: reward += cfg.alive_bonus

        return obs, reward, done, {}


    def get_ET_reward(self, max_eplen_reached, terminate_early):
        """ Punish falling hard and reward reaching episode's end a lot. """

        # calculate a running mean of the ep_return
        self.mean_epret_smoothed = smooth('mimic_env_epret', np.sum(self.ep_rews), 0.5)
        self.ep_rews = []

        # reward reaching the end of the episode without falling
        # reward = expected cumulative future reward
        if max_eplen_reached:
            # estimate future cumulative reward expecting getting the mean reward per step
            mean_step_rew = self.mean_epret_smoothed / ep_dur
            act_ret_est = np.sum(mean_step_rew * np.power(cfg.gamma, np.arange(ep_dur)))
            reward = act_ret_est
        # punish for ending the episode early
        else:
            reward = -1 * self.mean_epret_smoothed

        return reward


    def rescale_actions(self, a):
        """Policy samples actions from normal Gaussian distribution around 0 with init std of 0.5.
           In this method, we rescale the actions to the actual action ranges."""
        # policy outputs (normalized) joint torques
        if cfg.env_out_torque:
            # clip the actions to the range of [-1,1]
            a = np.clip(a, -1, 1)
            # scale the actions with joint peak torques
            # *2: peak torques are the same for both sides
            a *= cfgl.PEAK_JOINT_TORQUES * 2

        # policy outputs target angles for PD position controllers
        else:
            if cfg.is_mod(cfg.MOD_PI_OUT_DELTAS):
                # qpos of actuated joints
                qpos_act_before_step = self.get_qpos(True, True)
                # unnormalize the normalized deltas
                a *= self.get_max_qpos_deltas()
                # add the deltas to current position
                a = qpos_act_before_step + a
        return a



    def get_sim_freq_and_frameskip(self):
        """
        What is the frequency the simulation is running at
        and how many frames should be skipped during step(action)?
        """
        skip_n_frames = cfg.SIM_FREQ/cfg.CTRL_FREQ
        assert skip_n_frames.is_integer(), \
            f"Please check the simulation and control frequency in the config! " \
            f"The simulation frequency should be an integer multiple of the control frequency." \
            f"Your simulation frequency is {cfg.SIM_FREQ} and control frequency is {cfg.CTRL_FREQ}"
        return cfg.SIM_FREQ, int(skip_n_frames)

    def get_joint_kinematics(self, exclude_com=False, concat=False):
        '''Returns qpos and qvel of the agent.'''
        qpos = np.copy(self.sim.data.qpos)
        qvel = np.copy(self.sim.data.qvel)
        if exclude_com:
            qpos = self._remove_by_indices(qpos, self._get_COM_indices())
            qvel = self._remove_by_indices(qvel, self._get_COM_indices())
        if concat:
            return np.concatenate([qpos, qvel]).flatten()
        return qpos, qvel

    def _exclude_joints(self, qvals, exclude_com, exclude_not_actuated_joints):
        """ Takes qpos or qvel as input and outputs only a portion of the values
            dependent on which indices has to be excluded. """
        if exclude_not_actuated_joints:
            qvals = self._remove_by_indices(qvals, self._get_not_actuated_joint_indices())
        elif exclude_com:
            qvals = self._remove_by_indices(qvals, self._get_COM_indices())
        return qvals

    def get_qpos(self, exclude_com=False, exclude_not_actuated_joints=False):
        qpos = np.copy(self.sim.data.qpos)
        return self._exclude_joints(qpos, exclude_com, exclude_not_actuated_joints)

    def get_qvel(self, exclude_com=False, exclude_not_actuated_joints=False):
        qvel = np.copy(self.sim.data.qvel)
        return self._exclude_joints(qvel, exclude_com, exclude_not_actuated_joints)

    def get_ref_qpos(self, exclude_com=False, exclude_not_actuated_joints=False):
        qpos = self.refs.get_qpos()
        return self._exclude_joints(qpos, exclude_com, exclude_not_actuated_joints)

    def get_ref_qvel(self, exclude_com=False, exclude_not_actuated_joints=False):
        qvel = self.refs.get_qvel()
        return self._exclude_joints(qvel, exclude_com, exclude_not_actuated_joints)

    def activate_evaluation(self):
        self._EVAL_MODEL = True

    def is_evaluation_on(self):
        return self._EVAL_MODEL

    def do_fly(self, fly=True):
        self._FLY = fly

    def get_actuator_torques(self, abs_mean=False):
        tors = np.copy(self.sim.data.actuator_force)
        return np.mean(np.abs(tors)) if abs_mean else tors

    def get_force_ranges(self):
        return np.copy(self.model.actuator_forcerange)

    def get_qpos_ranges(self):
        ctrl_ranges = np.copy(self.model.actuator_ctrlrange)
        low = ctrl_ranges[:, 0]
        high = ctrl_ranges[:, 1]
        return low, high


    def get_max_qpos_deltas(self):
        """Returns the scalars needed to rescale the normalized actions from the agent."""
        # get max allowed deviations in actuated joints
        max_vels = self._get_max_actuator_velocities()
        # double max deltas for better perturbation recovery. To keep balance,
        # the agent might require to output angles that are not reachable and saturate the motors.
        SCALE_MAX_VELS = 2
        max_qpos_deltas = SCALE_MAX_VELS * max_vels / self.control_freq
        return max_qpos_deltas


    def playback_ref_trajectories(self, timesteps=2000, pd_pos_control=False):
        global _play_ref_trajecs
        _play_ref_trajecs = True

        from gym_mimic_envs.monitor import Monitor
        env = Monitor(self)

        env.reset()
        if pd_pos_control:
            for i in range(timesteps):
                # hold com x and z position and trunk rotation constant
                FLIGHT = self._FLY
                ignore_not_actuated_joints = True and not FLIGHT

                if not ignore_not_actuated_joints:
                    # position servos do not actuate all joints
                    # for those joints, we still have to set the joint positions by hand
                    ref_qpos = self.get_ref_qpos()
                    ref_qvel = self.get_ref_qvel()
                    qpos = self.get_qpos()
                    qvel = self.get_qvel()

                    if FLIGHT:
                        ref_qpos[[0,1,2]] = [0, 1.5, 0]
                        qvel[[0,1]] = 0
                    # set the not actuated joint position from refs
                    not_actuated_is = self._get_not_actuated_joint_indices()
                    qpos[not_actuated_is] = ref_qpos[not_actuated_is]
                    qvel[not_actuated_is] = ref_qvel[not_actuated_is]
                    self.set_joint_kinematics_in_sim(qpos, qvel)

                    # self.sim.data.qvel[:] = 0
                    # fix all other joints to focus tuning control gains of a single joint pair
                    # self.sim.data.qpos[[0, 2, 3, 4, 6, 7]] = 0
                    # self.sim.data.qvel[[0, 2, 3, 4, 6, 7]] = 0

                # follow desired trajecs
                des_qpos = self.get_ref_qpos(exclude_not_actuated_joints=True)
                # des_qpos[0] = 0
                # des_qpos[1] = 0.4
                # if i % 60 == 0:
                #     self.reset()
                if FLIGHT:
                    self.sim.data.qpos[[0, 1, 2]] = [0, 1.5, 0]
                    self.sim.data.qvel[[0, 1, 2]] = 0
                obs, reward, done, _ = env.step(des_qpos)
                # obs, reward, done, _ = self.step(np.ones_like(des_qpos)*(0))
                # ankle tune:
                # obs, reward, done, _ = self.step([0, 0, -0.3, 0, 0, -0.3])
                # knee tune:
                # obs, reward, done, _ = self.step([0, 0.2, 0, 0, 0.2, 0])
                # obs, reward, done, _ = self.step([-0.5, 1, -.2, -0.5, 1, -.2])
                self.render()
                if done:
                    self.reset()
        else:
            for i in range(timesteps):
                self.refs.next()
                ZERO_INPUTS = False
                if ZERO_INPUTS:
                    qpos, qvel = self.get_joint_kinematics()
                    qpos= np.zeros_like(qpos)
                    qvel = np.zeros_like(qvel)
                    self.set_joint_kinematics_in_sim(qpos, qvel)
                else:
                    self.set_joint_kinematics_in_sim()
                # self.step(np.zeros_like(self.action_space.sample()))
                self.sim.forward()
                self.render()

        _play_ref_trajecs = False
        self.close()
        raise SystemExit('Environment intentionally closed after playing back trajectories.')


    def set_joint_kinematics_in_sim(self, qpos=None, qvel=None):
        """ Specify the desired qpos and qvel and do a forward step in the simulation
            to set the specified qpos and qvel values. """
        old_state = self.sim.get_state()
        if qpos is None or qvel is None:
            qpos, qvel = self.refs.get_ref_kinmeatics()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)

    def activate_speed_control(self, desired_walking_speeds):
        self.FOLLOW_DESIRED_SPEED_PROFILE = True
        self.desired_walking_speeds = desired_walking_speeds
        self.i_speed = 0

    def _get_obs(self):
        qpos, qvel = self.get_joint_kinematics()
        # remove COM x position as the action should be independent of it
        qpos = qpos[1:]

        if self.FOLLOW_DESIRED_SPEED_PROFILE:
            self.desired_walking_speed = self.desired_walking_speeds[self.i_speed]
            self.i_speed += 1
            if self.i_speed >= len(self.desired_walking_speeds): self.i_speed = 0
        else:
            # TODO: during evaluation when speed control is inactive, we should just specify a constant speed
            #  when speed control is not active, set the speed to a constant value from the config
            #  During training, we still should use the step vel from the mocap!
            self.desired_walking_speed = self.refs.get_step_velocity()

        phase = self.refs.get_phase_variable()

        obs = np.concatenate([np.array([phase, self.desired_walking_speed]), qpos, qvel]).ravel()

        # when we mirror the policy (phase based mirr), mirror left step
        if cfg.is_mod(cfg.MOD_MIRR_POLICY) and self.refs.is_step_left():
            obs = self.mirror_obs(obs)

        return obs


    def mirror_obs(self, obs):
        is3d = True
        if is3d:
            # 3D Walker obs indices:
            #           0: phase, 1: des_vel, 2: com_y, 3: com_z,
            #           4: trunk_rot_x, 5: trunk_rot_y, 6: trunk_rot_z,
            #           7: hip_ang_r_sag, 8: hip_ang_r_front, 9: knee_ang_r, 10: ankle_ang_r,
            #           11: hip_ang_l_sag, 12: hip_ang_l_front 13: knee_ang_l, 14: ankle_ang_l,
            #           15: com_x_vel, 16: com_y_vel, 17:com_z_vel,
            #           18: trunk_x_ang_vel, 19: trunk_y_ang_vel, 20: trunk_z_ang_vel,
            #           21: hip_sag_vel_r, 22: hip_front_vel_r, 23: knee_vel_r, 24: ankle_vel_r,
            #           25: hip_sag_vel_l, 26: hip_front_vel_l, 27: knee_vel_l, 28: ankle_vel_l
            mirred_obs_indices = [0, 1, 2, 3,
                                  4, 5, 6,
                                  11, 12, 13, 14,
                                  7, 8, 9, 10,
                                  15, 16, 17,
                                  18, 19, 20,
                                  25, 26, 27, 28,
                                  21, 22, 23, 24]
            # some observations and actions retain the same absolute value but change the sign
            negate_obs_indices = [2, 4, 6, 8, 12, 16, 18, 20, 22, 26]
        else:
            # 2D Walker obs indices:
            #           0: phase, 1: des_vel, 2: com_z, 3: trunk_rot,
            #           4: hip_ang_r, 5: knee_ang_r, 6: ankle_ang_r,
            #           7: hip_ang_l, 8: knee_ang_l, 9: ankle_ang_l,
            #           10: com_x_vel, 11:com_z_vel, 12: trunk_ang_vel,
            #           13: hip_vel_r, 14: knee_vel_r, 15: ankle_vel_r,
            #           16: hip_vel_l, 17: knee_vel_l, 18: ankle_vel_l
            mirred_obs_indices = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6,
                                  10, 11, 12, 16, 17, 18, 13, 14, 15]

        obs_mirred = obs[mirred_obs_indices]

        if is3d:
            obs_mirred[negate_obs_indices] *= -1

        return obs_mirred


    def mirror_action(self, acts):
        is3d = '3d' in cfg.env_abbrev or '3pd' in cfg.env_abbrev
        if is3d:
            mirred_acts_indices = [4, 5, 6, 7, 0, 1, 2, 3]
            # some observations and actions retain the same absolute value but change the sign
            negate_act_indices = [1, 5]
        else:
            mirred_acts_indices = [3, 4, 5, 0, 1, 2]

        try:
            acts_mirred = acts[mirred_acts_indices]
            if is3d:
                acts_mirred[negate_act_indices] *= -1
        except Exception as ex:
            log('WARNING! Could not mirror actions',
                [f'Exception: {ex}', f'Actions: {acts}'])
            acts_mirred = acts


        return acts_mirred


    def randomize_dynamics(self):
        model = self.model
        dir(model)
        print(dir(model))
        exit(33)
        debug = True

    def reset_model(self):

        # self.randomize_dynamics()

        # get desired qpos and qvel from the refs (also include the trunk COM positions and vels)
        qpos, qvel = self.get_init_state(not self.is_evaluation_on() and not self.FOLLOW_DESIRED_SPEED_PROFILE)
        # apply the refs kinematics to the simulation
        self.set_state(qpos, qvel)

        # applying refs kinematics to the model might init the model
        # without ground contact or with significant ground penetration
        # Set the following constant to True to initialize walker always in ground contact.
        # CAUTION: This requires your mujoco model to have sites on their foot soul.
        # See mujoco/gym_mimic_envs/mujoco/assets/walker3d_flat_feet.xml for an example.
        OPTIMIZE_GROUND_CONTANT_ON_INITIALIZATION = True
        if OPTIMIZE_GROUND_CONTANT_ON_INITIALIZATION:
            # we determine the lowest foot position (4 sites at the bottom of each foot in the XML file)
            foot_corner_positions = self.data.site_xpos
            assert foot_corner_positions is not None, \
                "In order to optimize foot-ground-contact on episode initialization, " \
                "please add sites to each corner of your walker's feet (MJCF file)!"
            lowest_foot_z_pos = np.min(foot_corner_positions[:, -1])
            # and shift the trunks COM Z position to have contact between ground and lowest foot point
            qpos[2] -= lowest_foot_z_pos
            # set the new state with adjusted trunk COM position in the simulation
            self.set_state(qpos, qvel)

        # sanity check: reward should be around 1 after initialization
        rew = self.get_imitation_reward()
        assert (rew > 0.95 * cfg.rew_scale) if not self._FLY else 0.5, \
            f"Reward should be around 1 after RSI, but was {rew}!"

        # set the reference trajectories to the next state,
        # otherwise the first step after initialization has always zero error
        self.refs.next()

        # get and return current observations
        obs = self._get_obs()
        return obs


    def get_init_state(self, random=True):
        ''' Random State Initialization:
            @returns: qpos and qvel of a random step at a random position'''
        return self.refs.get_random_init_state() if random \
            else self.refs.get_deterministic_init_state()


    def get_ref_kinematics(self, exclude_com=False, concat=False):
        qpos, qvel = self.refs.get_ref_kinmeatics()
        if exclude_com:
            qpos = self._remove_by_indices(qpos, self._get_COM_indices())
            qvel = self._remove_by_indices(qvel, self._get_COM_indices())
        if concat:
            return np.concatenate([qpos, qvel]).flatten()
        return qpos, qvel


    def get_pose_reward(self):
        # get sim and ref joint positions excluding com position
        qpos, _ = self.get_joint_kinematics(exclude_com=True)
        ref_pos, _ = self.get_ref_kinematics(exclude_com=True)
        if self._FLY:
            ref_pos[0] = 0

        dif = qpos - ref_pos
        dif_sqrd = np.square(dif)
        sum = np.sum(dif_sqrd)
        pose_rew = np.exp(-3 * sum)
        return pose_rew

    def get_vel_reward(self):
        _, qvel = self.get_joint_kinematics(exclude_com=True)
        _, ref_vel = self.get_ref_kinematics(exclude_com=True)

        if self._FLY:
            # set trunk angular velocity to zero
            ref_vel[0] = 0

        difs = qvel - ref_vel
        dif_sqrd = np.square(difs)
        dif_sum = np.sum(dif_sqrd)
        vel_rew = np.exp(-0.05 * dif_sum)
        return vel_rew

    def get_com_reward(self):
        qpos, qvel = self.get_joint_kinematics()
        ref_pos, ref_vel = self.get_ref_kinematics()
        com_is = self._get_COM_indices()
        com_pos, com_ref = qpos[com_is], ref_pos[com_is]
        dif = com_pos - com_ref
        dif_sqrd = np.square(dif)
        sum = np.sum(dif_sqrd)
        com_rew = np.exp(-16 * sum)
        return com_rew


    def _remove_by_indices(self, list, indices):
        """
        Removes specified indices from the passed list and returns it.
        """
        new_list = [item for i, item in enumerate(list) if i not in indices]
        return np.array(new_list)


    def get_imitation_reward(self):
        """ DeepMimic imitation reward function """

        # get rew weights from rew_weights_string
        weights = [float(digit)/10 for digit in cfg.rew_weights]

        w_pos, w_vel, w_com, w_pow = weights
        pos_rew = self.get_pose_reward()
        vel_rew = self.get_vel_reward()
        com_rew = self.get_com_reward()
        pow_rew = self.get_energy_reward() if w_pow != 0 else 0

        self.pos_rew, self.vel_rew, self.com_rew = pos_rew, vel_rew, com_rew

        imit_rew = w_pos * pos_rew + w_vel * vel_rew + w_com * com_rew + w_pow * pow_rew

        return imit_rew


    def do_terminate_early(self):
        """
        Early Termination based on multiple checks:
        - does the character exceeds allowed trunk angle deviations
        - does the characters COM in Z direction is below a certain point
        - does the characters COM in Y direction deviated from straight walking
        """
        if _play_ref_trajecs:
            return False

        qpos = self.get_qpos()
        ref_qpos = self.refs.get_qpos()
        com_indices = self._get_COM_indices()
        trunk_ang_indices = self._get_trunk_rot_joint_indices()

        com_height = qpos[com_indices[-1]]
        com_y_pos = qpos[com_indices[1]]
        trunk_angs = qpos[trunk_ang_indices]
        ref_trunk_angs = ref_qpos[trunk_ang_indices]

        # calculate if trunk saggital angle is out of allowed range
        max_pos_sag = 0.3
        max_neg_sag = -0.05
        is2d = len(trunk_ang_indices) == 1
        if is2d:
            trunk_ang_saggit = trunk_angs # is the saggital ang
            trunk_ang_sag_exceeded = trunk_ang_saggit > max_pos_sag or trunk_ang_saggit < max_neg_sag
            trunk_ang_exceeded = trunk_ang_sag_exceeded
        else:
            max_front_dev = 0.2
            max_axial_dev = 0.5 # should be much smaller but actually doesn't really hurt performance
            trunk_ang_front, trunk_ang_saggit, trunk_ang_axial = trunk_angs
            front_dev, sag_dev, ax_dev = np.abs(trunk_angs - ref_trunk_angs)
            trunk_ang_front_exceeded = front_dev > max_front_dev
            trunk_ang_axial_exceeded = ax_dev > max_axial_dev
            trunk_ang_sag_exceeded = trunk_ang_saggit > max_pos_sag or trunk_ang_saggit < max_neg_sag
            trunk_ang_exceeded = trunk_ang_sag_exceeded or trunk_ang_front_exceeded \
                                 or trunk_ang_axial_exceeded

        # check if agent has deviated too much in the y direction
        is_drunk = np.abs(com_y_pos) > 0.2

        # is com height too low (e.g. walker felt down)?
        min_com_height = 0.75
        com_height_too_low = com_height < min_com_height
        if self._FLY: com_height_too_low = False

        terminate_early = com_height_too_low or trunk_ang_exceeded or is_drunk
        return terminate_early, com_height_too_low, trunk_ang_exceeded, is_drunk


    # ----------------------------
    # Methods we override:
    # ----------------------------

    def close(self):
        # calls MujocoEnv.close()
        super().close()

    # ----------------------------
    # Methods to override:
    # ----------------------------

    def _get_COM_indices(self):
        """
        Needed to distinguish between joint and COM kinematics.

        Returns a list of indices pointing at COM joint position/index
        in the considered robot model, e.g. [0,1,2]

        Caution: Do not include trunk rotational joints here.
        """
        raise NotImplementedError

    def _get_trunk_rot_joint_indices(self):
        """ Return the indices of the rotational joints of the trunk."""
        raise NotImplementedError

    def _get_not_actuated_joint_indices(self):
        """
        Needed for playing back reference trajectories
        by using position servos in the actuated joints.

        @returns a list of indices specifying indices of
        joints in the considered robot model that are not actuated.
        Example: return [0,1,2]
        """
        raise NotImplementedError

    def _get_max_actuator_velocities(self):
        """
        Needed to calculate the max mechanical power for the energy efficiency reward.
        :returns: A numpy array containing the maximum absolute velocity peaks
                  of each actuated joint. E.g. np.array([6, 12, 12, 6, 12, 12])
        """
        raise NotImplementedError


    def has_ground_contact(self):
        """
        :returns: two booleans indicating ground contact of left and right foot.
        Example: [True, False] means left foot has ground contact, [True, True] indicates double stance.
        """
        raise NotImplementedError
