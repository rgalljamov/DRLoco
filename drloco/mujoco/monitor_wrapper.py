import gym
import numpy as np
import seaborn as sns
from drloco.common.utils import config_pyplot, is_remote, \
    exponential_running_smoothing as smooth, change_plot_properties
from drloco.mujoco.mimic_env import MimicEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# length of the buffer containing sim and ref trajecs for comparison
_trajec_buffer_length = 2000

PLOT_REF_DISTRIB =  False

class Monitor(gym.Wrapper):

    def __init__(self, env: MimicEnv):
        try:
            env_type = type(env)
            env_is_mimic_env = isinstance(env, MimicEnv)
            env_is_subproc = isinstance(env.venv, SubprocVecEnv)
            env_is_normalize = isinstance(env, VecNormalize)
            env_is_dummy = isinstance(env, DummyVecEnv)
        except:
            pass
        if isinstance(env, MimicEnv):
            self.env = env
        elif isinstance(env, DummyVecEnv):
            self.env = env.envs[0]

        super(Monitor, self).__init__(self.env)

        self.kinem_labels = self.env.refs.get_kinematics_labels()
        self.num_dofs = len(self.kinem_labels)

        # self.num_dofs = self.env.observation_space.high.size
        self.num_actions = self.env.action_space.high.size
        # do we want to control walking speed
        self.SPEED_CONTROL = False

        self.setup_containers()

        self.plt = config_pyplot(fig_size=True)


    def setup_containers(self):
        self.ep_len = 0
        self.reward = 0
        self.ep_len_smoothed = 0
        self.rewards = []
        # mean reward per step, calculated at each episode end
        self.mean_reward_smoothed = 0
        self.returns = []
        self.ep_ret_smoothed = 0
        self.ep_lens = []
        self.grfs_left = []
        self.grfs_right = []
        self.moved_distance = 0
        # track phases during initialization and ET
        self.et_positions = []
        self.rsi_positions = []
        # which phases lead to episode lengths smaller than the running average
        self.difficult_rsi_phases = []
        # track reward components
        self.ep_pos_rews, self.ep_vel_rews, self.ep_com_rews = [], [], []
        self.mean_ep_pos_rew_smoothed, self.mean_ep_vel_rew_smoothed, \
        self.mean_ep_com_rew_smoothed = 0,0,0

        # monitor energy efficiency
        self.ep_torques_abs = []
        self.mean_abs_ep_torque_smoothed = 0
        self.median_abs_torque_smoothed = 0

        # monitor sim and ref trajecs for comparison (sim/ref, kinem_indices, timesteps)
        # 3 and 4 in first dimension are for mean and std of ref trajec distribution
        self.trajecs_buffer = np.zeros((4, self.num_dofs, _trajec_buffer_length))
        # monitor episode terminations
        self.dones_buf = np.zeros((_trajec_buffer_length,))
        # monitor the actions at actuated joints (PD target angles)
        self.action_buf = np.zeros((self.num_actions, _trajec_buffer_length))
        # monitor the joint torques
        self.torque_buf = np.zeros((self.num_actions, _trajec_buffer_length))
        # monitor desired walking speed
        self.speed_buf = np.zeros((_trajec_buffer_length,))

        self.left_step_distrib, self.right_step_distrib = None, None


    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        if self.ep_len == 0:
            self.init_pos = self.env.refs._pos
            self.rsi_positions.append(self.init_pos)
        self.ep_len += 1

        self.reward = reward
        self.rewards.append(reward)
        self.ep_pos_rews.append(self.env.pos_rew)
        self.ep_vel_rews.append(self.env.vel_rew)
        self.ep_com_rews.append(self.env.com_rew)

        self.ep_torques_abs.append(self.env.get_actuator_torques(True))

        if done:
            # get phase ET was detected at
            et_pos = self.env.refs._pos
            self.et_positions.append(et_pos)

            ep_rewards = self.rewards[-self.ep_len:]
            mean_reward = np.mean(ep_rewards[:-1])
            self.mean_reward_smoothed = smooth('rew', mean_reward)
            self.mean_ep_pos_rew_smoothed = smooth('ep_pos_rew', np.mean(self.ep_pos_rews))
            self.mean_ep_vel_rew_smoothed = smooth('ep_vel_rew', np.mean(self.ep_vel_rews))
            self.mean_ep_com_rew_smoothed = smooth('ep_com_rew', np.mean(self.ep_com_rews))

            ep_return = np.sum(ep_rewards)
            self.returns.append(ep_return)
            self.ep_ret_smoothed = smooth('ep_ret', ep_return, 0.25)

            self.ep_lens.append(self.ep_len)
            self.ep_len_smoothed = smooth('ep_len', self.ep_len, 0.75)
            if self.ep_len < self.ep_len_smoothed*0.75:
                self.difficult_rsi_phases.append(self.init_pos)
            self.ep_len = 0

            self.moved_distance = self.env.get_walked_distance()
            # self.moved_distance_smooth = smooth('dist', self.env.data.qpos[0], 0.25)

            self.mean_abs_ep_torque_smoothed = \
                smooth('mean_ep_tor', np.mean(self.ep_torques_abs), 0.75)
            self.median_abs_torque_smoothed = \
                smooth('med_ep_tor', np.median(self.ep_torques_abs), 0.75)
            self.ep_torques_abs = []


        COMPARE_TRAJECS = False and not is_remote()
        if COMPARE_TRAJECS:

            # save sim and ref trajecs in a buffer for comparison
            sim_trajecs = self.env.get_joint_kinematics(concat=True)
            ref_trajecs = self.env.get_ref_kinematics(concat=True)
            # fifo approach, replace oldest entry with the newest one
            self.trajecs_buffer = np.roll(self.trajecs_buffer, -1, axis=2)
            self.trajecs_buffer[0, :, -1] = sim_trajecs
            self.trajecs_buffer[1, :, -1] = ref_trajecs

            # do the same with the dones
            self.dones_buf = np.roll(self.dones_buf, -1)
            self.dones_buf[-1] = done
            # and with the desired walking speed
            self.speed_buf = np.roll(self.speed_buf, -1)
            self.speed_buf[-1] = self.env.desired_walking_speed
            # save actions
            self.action_buf = np.roll(self.action_buf, -1, axis=1)
            self.action_buf[:, -1] = action
            # save joint toqrues
            self.torque_buf = np.roll(self.torque_buf, -1, axis=1)
            self.torque_buf[:, -1] = self.get_actuator_torques()

            # plot trajecs when the buffers are filled
            try: self.trajecs_recorded += 1
            except: self.trajecs_recorded = 1
            if self.trajecs_recorded % (1 * _trajec_buffer_length) == 0:
                self.compare_sim_ref_trajecs()

        return obs, reward, done, _


    def compare_sim_ref_trajecs(self):
        """
        Plot simulation and reference trajectories in a single figure
        to compare them.
        """
        plt = self.plt
        plt.rcParams.update({'figure.autolayout': False})
        plt.rcParams['figure.figsize'] = (19.2, 6.8)
        plt.subplots_adjust(top=0.974, bottom=0.13, left=0.06, right=0.978, hspace=0.15, wspace=0.44)
        font_size, _, _ = change_plot_properties(-4, -2, -2, 1)
        sns.set_style("whitegrid", {'axes.edgecolor':'#ffffff00'})
        names = ['Simulation'] # line names (legend)
        second_y_axis_pos = 1.0

        ONLY_ACTUATED_JOINTS = True
        if ONLY_ACTUATED_JOINTS:
            # both legs
            # inds = list(range(6,14)) + list(range(20, 27))
            # right leg only
            inds = list(range(6,10)) + list(range(20, 24))
            self.trajecs_buffer = self.trajecs_buffer[:, inds, :]
            self.kinem_labels = self.kinem_labels[inds]
            plt.rcParams.update({'figure.autolayout': False})

        if self.SPEED_CONTROL:
            # plt.rcParams.update({'axes.labelsize': 14})
            num_joints = 2
            rows, cols = 3, 1
            # only plot com x pos and velocity
            inds = [0, 9]
            self.trajecs_buffer = self.trajecs_buffer[:, inds, :]
            self.kinem_labels = self.kinem_labels[inds]
            y_labels = ['Moved Distance [m]', 'COM X Vel [m/s]']
        else:
            num_joints = len(self.kinem_labels)
            cols = 5
            rows = int((num_joints+1)/cols) + 1
            if ONLY_ACTUATED_JOINTS:
                cols = 4
                rows = 2
        # plot sim trajecs
        trajecs = self.trajecs_buffer[0,:,:]
        # collect axes to reuse them for overlaying multiple plots
        axes = []
        # collect different lines to place the legend in a separate subplot
        lines = []
        for i_joint in range(num_joints):
            try: axes.append(plt.subplot(rows, cols, i_joint + 1, sharex=axes[i_joint-1]))
            except: axes.append(plt.subplot(rows, cols, i_joint + 1))
            trajec = trajecs[i_joint, :]
            line = plt.plot(trajec)
            # show episode ends
            plt.rcParams['lines.linewidth'] = 1
            plt.vlines(np.argwhere(self.dones_buf).flatten()+1,
                       np.min(trajec), np.max(trajec), colors='#cccccc', linestyles='dashed')
            plt.rcParams['lines.linewidth'] = 2
            if self.SPEED_CONTROL:
                plt.ylabel(y_labels[i_joint])
            else:
                plt.ylabel(f'{i_joint+1}. ' + self.kinem_labels[i_joint])
        lines.append(line[0])

        # plot ref trajec distributions (mean + 2std)
        if PLOT_REF_DISTRIB:
            trajecs = self.trajecs_buffer[2,:,:]
            stds = self.trajecs_buffer[3,:,:]
            for i_joint in range(num_joints):
                trajec = trajecs[i_joint, :]
                std = stds[i_joint, :]
                line = axes[i_joint].plot(trajec)
                axes[i_joint].fill_between(range(len(trajec)), trajec+std, trajec-std,
                                           color='orange', alpha=0.5)
            lines.append(line[0])
            names.append('Reference Distribution\n(mean $\pm$ 2std)')

        PLOT_REFS = True
        if PLOT_REFS:
            trajecs = self.trajecs_buffer[1, :, :]
            for i_joint in range(num_joints):
                trajec = trajecs[i_joint, :]
                line = axes[i_joint].plot(trajec, color='red' if PLOT_REF_DISTRIB else 'orange')

            lines.append(line[0])
            names.append('Reference')

        def plot_actions(buffer, name, line_color='#777777'):
            with sns.axes_style("white", {"axes.edgecolor": '#ffffff00',
                                          "ytick.color":'#ffffff00'}):
                i_not_actuated = self.env._get_not_actuated_joint_indices()
                i_actuated = 0
                plt.rcParams['lines.linewidth'] = 1
                for i_joint in range(num_joints):
                    if i_joint in i_not_actuated:
                        continue
                    if i_actuated >= buffer.shape[0]:
                        break
                    act_plt = axes[i_joint].twinx()
                    act_plt.spines['right'].set_position(('axes', second_y_axis_pos))
                    line = act_plt.plot(buffer[i_actuated, :], line_color+'77')
                    act_plt.tick_params(axis='y', labelcolor=line_color)
                    i_actuated += 1
                plt.rcParams['lines.linewidth'] = 2
            lines.append(line[0])
            names.append(name)

        PLOT_TORQUES = False
        if PLOT_TORQUES:
            plot_actions(self.torque_buf/1000, "Joint Torque [kNm]")
            second_y_axis_pos = 1.12

        PLOT_ACTIONS = False
        if PLOT_ACTIONS:
            plot_actions(self.action_buf, 'PD Target', '#ff0000')

        # remove x ticks from upper graphs
        for i_graph in (range(len(axes) - cols + 1) if not ONLY_ACTUATED_JOINTS else range(4)):
            axes[i_graph].tick_params(axis='x', which='both',
                                      labelbottom=False)

        if self.SPEED_CONTROL:
            plt.subplot(rows, cols, 3, sharex=axes[-1])
            plt.plot(self.speed_buf)
            plt.ylabel('Desired Walking Speed [m/s]')
            plt.xlabel('Simulation Timesteps []')
            axes[0].legend(lines, names)
        elif ONLY_ACTUATED_JOINTS:
            axes[-1].legend(lines, names, loc='upper right')
            plt.gcf().text(0.5, 0.04, r'Simulation Timesteps', ha='center', fontsize=font_size+2)
            plt.subplots_adjust(top=0.974, bottom=0.13, left=0.06, right=0.978, hspace=0.15, wspace=0.44)
            axes[-1].set_xlim([1400, 2000])
        else:
            # plot the legend in a separate subplot
            with sns.axes_style("white", {"axes.edgecolor": 'white'}):
                legend_subplot = plt.subplot(rows, cols, num_joints + 2)
                legend_subplot.set_xticks([])
                legend_subplot.set_yticks([])
                legend_subplot.legend(lines, names, bbox_to_anchor=(
                    1.2 if PLOT_REF_DISTRIB else 1, 1.075 if PLOT_REF_DISTRIB else 1))

            PLOT_REWS = False
            if PLOT_REWS:
                # add rewards and returns
                from drloco.config.config import rew_scale, alive_bonus
                rews = np.copy(self.rewards[-_trajec_buffer_length:])
                rews -= alive_bonus
                rews /= rew_scale

                rew_plot = plt.subplot(rows, cols, len(axes) + 1, sharex=axes[-1])
                rew_plot.plot(rews)
                # rew_plot.set_ylim(np.array([-0.075, 1.025]))
                # plot episode terminations
                plt.vlines(np.argwhere(self.dones_buf).flatten() + 1,
                           0, 1, colors='#cccccc')
                # plot episode returns
                ret_plot = rew_plot.twinx().twiny()
                ret_plot.plot(self.returns, '#77777777')
                ret_plot.tick_params(axis='y', labelcolor='#77777777')
                ret_plot.set_xticks([])

                plt.title('Rewards & Returns')

        # fix title overlapping when tight_layout is true
        plt.gcf().tight_layout() # rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.25, hspace=0.4)
        PD_TUNING = False
        if PD_TUNING:
            rew_plot.set_xlim([-5, 250])
            dampings = self.env.model.dof_damping[3:].astype(int).tolist()
            kps = self.env.model.actuator_gainprm[:,0].astype(int).tolist()
            mean_rew = int(1000 * np.mean(self.rewards[-_trajec_buffer_length:]))
            plt.suptitle(f'PD Gains Tuning:   rew={mean_rew}    kp={kps}    kd={dampings}')
        elif self.SPEED_CONTROL:
            plt.suptitle('Simulation and Reference Joint Kinematics over Time')

        plt.show()
        if self.env.is_evaluation_on() or PD_TUNING:
            raise SystemExit('Planned exit after closing trajectory comparison plot.')