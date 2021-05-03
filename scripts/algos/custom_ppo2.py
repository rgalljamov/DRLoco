import time, traceback
import numpy as np

from stable_baselines import PPO2
from scripts.common.utils import log
from scripts.common import config as cfg
from scripts.behavior_cloning.dataset import get_obs_and_delta_actions

# imports required to copy the learn method
from stable_baselines import logger
from stable_baselines.common.math_util import safe_mean
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common import explained_variance, SetVerbosity, TensorboardWriter


def mirror_experiences(rollout, ppo2=None):
    obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
    assert obs.shape[0] == cfg.batch_size
    assert states is None
    assert len(ep_infos) == 0

    is3d = cfg.env_is3d
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
        mirred_acts_indices = [4, 5, 6, 7, 0, 1, 2, 3]
        # some observations and actions retain the same absolute value but change the sign
        negate_obs_indices = [2, 4, 6, 8, 12, 16, 18, 20, 22, 26]
        negate_act_indices = [1, 5]
    else:
        # 2D Walker obs indices:
        #           0: phase, 1: des_vel, 2: com_z, 3: trunk_rot,
        #           4: hip_ang_r, 5: knee_ang_r, 6: ankle_ang_r,
        #           7: hip_ang_l, 8: knee_ang_l, 9: ankle_ang_l,
        #           10: com_x_vel, 11:com_z_vel, 12: trunk_ang_vel,
        #           13: hip_vel_r, 14: knee_vel_r, 15: ankle_vel_r,
        #           16: hip_vel_l, 17: knee_vel_l, 18: ankle_vel_l
        mirred_acts_indices = [3, 4, 5, 0, 1, 2]
        mirred_obs_indices = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6,
                              10, 11, 12, 16, 17, 18, 13, 14, 15]

    obs_mirred = obs[:, mirred_obs_indices]
    acts_mirred = actions[:, mirred_acts_indices]

    if is3d:
        obs_mirred[:, negate_obs_indices] *= -1
        acts_mirred[:, negate_act_indices] *= -1

    QUERY_NETS = cfg.is_mod(cfg.MOD_QUERY_NETS)
    if QUERY_NETS:
        parameters = ppo2.get_parameter_list()
        parameter_values = np.array(ppo2.sess.run(parameters))
        pi_w0, pi_w1, pi_w2 = parameter_values[[0, 2, 8]]
        pi_b0, pi_b1, pi_b2 = parameter_values[[1, 3, 9]]
        vf_w0, vf_w1, vf_w2 = parameter_values[[4, 6, 13]]
        vf_b0, vf_b1, vf_b2 = parameter_values[[5, 7, 14]]
        pi_logstd = parameter_values[10]
        pi_std = np.exp(pi_logstd)
        def relu(x): return np.maximum(x, 0)
        # get values of the mirrored observations
        def get_value(obs):
            vf_hid1 = relu(np.matmul(obs,vf_w0) + vf_b0)
            vf_hid2 = relu(np.matmul(vf_hid1, vf_w1) + vf_b1)
            values = np.matmul(vf_hid2, vf_w2) + vf_b2
            return values.flatten()

        def get_action_means(obs):
            pi_hid1 = relu(np.matmul(obs,pi_w0) + pi_b0)
            pi_hid2 = relu(np.matmul(pi_hid1, pi_w1) + pi_b1)
            means = np.matmul(pi_hid2, pi_w2) + pi_b2
            return means

        values_test = get_value(obs)
        values_mirred_obs = get_value(obs_mirred)

        def neglogp(acts, mean, logstd):
            std = np.exp(logstd)
            return 0.5 * np.sum(np.square((acts - mean) / std), axis=-1) \
                   + 0.5 * np.log(2.0 * np.pi) * np.array(acts.shape[-1], dtype=np.float) \
                   + np.sum(logstd, axis=-1)

        if not cfg.is_mod(cfg.MOD_QUERY_VF_ONLY):
            act_means = get_action_means(obs)
            act_means_mirred = get_action_means(obs_mirred)

            neglogpacs_test = neglogp(actions, act_means, pi_logstd)
            neglogpacs_mirred = neglogp(acts_mirred, act_means_mirred, pi_logstd)

            # log('Logstd', [f'logstd = {pi_logstd}', f'std = {pi_std}'])

            percentiles = [50, 75, 90, 95, 99, 100]
            if np.random.randint(0, 100, 1) == 77:
                log('Neglogpacs Comparison (before clipping!)',
                [f'neglogpacs orig: min {np.min(neglogpacs)}, '
                  f'mean {np.mean(neglogpacs)}, max {np.max(neglogpacs)}',
                  f'neglogpacs mirred: min {np.min(neglogpacs_mirred)}, '
                  f'mean {np.mean(neglogpacs_mirred)}, '
                  f'max {np.max(neglogpacs_mirred)}',
                  f'---\npercentiles {percentiles}:',
                  f'orig percentiles: {np.percentile(neglogpacs, percentiles)}',
                  f'mirred percentiles: {np.percentile(neglogpacs_mirred, percentiles)}',
                  ])

            # this doesn't work! we should rather delete actions that are too unprobable under pi!
            CLIP_NEGLOGPACS = False
            if CLIP_NEGLOGPACS:
                # limit neglogpacs_mirred to be not bigger than the max neglogpacs
                # otherwise the action distribution stay too wide
                max_allowed_neglogpac = 5 * np.percentile(neglogpacs, 99)
                min_allowed_neglogpac = 2 * np.min(neglogpacs) # np.percentile(neglogpacs, 1)
                neglogpacs_mirred = np.clip(neglogpacs_mirred,
                                            min_allowed_neglogpac, max_allowed_neglogpac)

            residuals_neglogpacs = neglogpacs - neglogpacs_test
            residuals_values = values - values_test

            difs_neglogpacs = neglogpacs_mirred - neglogpacs
            difs_values = values_mirred_obs - values

            log('Differences between original and mirrored experiences',
                [f'neglogpacs: min {np.min(difs_neglogpacs)} max {np.max(difs_neglogpacs)}\n'
                 f'values: min {np.min(difs_values)} max {np.max(difs_values)}'])

            if not ( (residuals_neglogpacs < 0.01).all() and (residuals_values < 0.01).all() ):
                log('WARNING!', ['Residuals exceeded allowed amplitude of 0.01',
                                 f'Neglogpacs: mean {np.mean(residuals_neglogpacs)}, max {np.max(residuals_neglogpacs)}',
                                 f'Values: mean {np.mean(residuals_values)}, max {np.max(residuals_values)}',
                                 ])

    obs = np.concatenate((obs, obs_mirred), axis=0)
    actions = np.concatenate((actions, acts_mirred), axis=0)

    if QUERY_NETS:
        values = np.concatenate((values, values_mirred_obs.flatten()))
        neglogpacs = np.concatenate((neglogpacs,
                                     neglogpacs_mirred.flatten()
                                     if not cfg.is_mod(cfg.MOD_QUERY_VF_ONLY)
                                     else neglogpacs))
    else:
        values = np.concatenate((values, values))
        neglogpacs = np.concatenate((neglogpacs, neglogpacs))

    # the other values should stay the same for the mirrored experiences
    returns = np.concatenate((returns, returns))
    masks = np.concatenate((masks, masks))
    true_reward = np.concatenate((true_reward, true_reward))

    # remove mirrored experiences with too high neglogpacs
    FILTER_MIRRED_EXPS = cfg.is_mod(cfg.MOD_QUERY_NETS) and not cfg.is_mod(cfg.MOD_QUERY_VF_ONLY)
    if FILTER_MIRRED_EXPS:
        n_mirred_exps = int(len(neglogpacs) / 2)
        max_allowed_neglogpac = 5 * np.percentile(neglogpacs[:n_mirred_exps], 99)
        delete_act_indices = np.where(neglogpacs[n_mirred_exps:] > max_allowed_neglogpac)[0] + n_mirred_exps
        if np.random.randint(0, 10, 1)[0] == 7:
            log(f'Deleted {len(delete_act_indices)} mirrored actions '
                f'with neglogpac > {max_allowed_neglogpac}')

        obs = np.delete(obs, delete_act_indices, axis=0)
        actions = np.delete(actions, delete_act_indices, axis=0)
        returns = np.delete(returns, delete_act_indices, axis=0)
        masks = np.delete(masks, delete_act_indices, axis=0)
        values = np.delete(values, delete_act_indices, axis=0)
        true_reward = np.delete(true_reward, delete_act_indices, axis=0)
        neglogpacs = np.delete(neglogpacs, delete_act_indices, axis=0)

    # assert true_reward.shape[0] == cfg.batch_size*2
    # assert obs.shape[0] == cfg.batch_size*2

    return obs, returns, masks, actions, values, \
           neglogpacs, states, ep_infos, true_reward


def generate_experiences_from_refs(rollout, ref_obs, ref_acts):
    """
    Generate experiences from reference trajectories.
    - obs and actions can be used without a change. TODO: obs and acts should be normalized by current running stats
    - predicted state values are estimated as the mean value of taken experiences. TODO: query VF network
    - neglogpacs, -log[p(a|s)], are estimated by using the smallest probability of taken experiences. TODO: query PI network
    - returns are estimated by max return of taken (s,a)-pairs
    TODO: Mirror refs
    """

    obs, returns, masks, actions, values, neglogpacs, \
    states, ep_infos, true_reward = rollout

    n_ref_exps = ref_obs.shape[0]
    ref_returns = np.ones((n_ref_exps,), dtype=np.float32) * np.max(returns)
    ref_values = np.ones((n_ref_exps,), dtype=np.float32) * np.mean(values)
    ref_masks = np.array([False] * n_ref_exps)
    ref_neglogpacs = np.ones_like(ref_values) * np.mean(neglogpacs)

    obs = np.concatenate((obs, ref_obs), axis=0)
    actions = np.concatenate((actions, ref_acts), axis=0)
    returns = np.concatenate((returns, ref_returns))
    masks = np.concatenate((masks, ref_masks))
    values = np.concatenate((values, ref_values))
    neglogpacs = np.concatenate((neglogpacs, ref_neglogpacs))

    return obs, actions, returns, masks, values, neglogpacs

class CustomPPO2(PPO2):
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        # log('Using CustomPPO2!')

        self.mirror_experiences = cfg.is_mod(cfg.MOD_MIRROR_EXPS)
        # to investigate the outputted actions in the monitor env
        self.last_actions = None

        if cfg.is_mod(cfg.MOD_REFS_REPLAY):
            # load obs and actions generated from reference trajectories
            self.ref_obs, self.ref_acts = get_obs_and_delta_actions(norm_obs=True, norm_acts=True, fly=False)

        if cfg.is_mod(cfg.MOD_EXP_REPLAY):
            self.replay_buf = np.ndarray((cfg.replay_buf_size,), dtype=object)

        super(CustomPPO2, self).__init__(policy, env, gamma, n_steps, ent_coef, learning_rate, vf_coef,
                                         max_grad_norm, lam, nminibatches, noptepochs, cliprange, cliprange_vf,
                                         verbose, tensorboard_log, _init_setup_model, policy_kwargs,
                                         full_tensorboard_log, seed, n_cpu_tf_sess)

    def exp_replay(self, rollout):
        obs, returns, masks, actions, values, neglogpacs, \
        states, ep_infos, true_reward = rollout

        QUERY_NETS = cfg.is_mod(cfg.MOD_QUERY_NETS)

        if QUERY_NETS:
            # get current PI and VF network parameters
            parameters = self.get_parameter_list()
            parameter_values = np.array(self.sess.run(parameters))
            pi_w0, pi_w1, pi_w2 = parameter_values[[0, 2, 8]]
            pi_b0, pi_b1, pi_b2 = parameter_values[[1, 3, 9]]
            vf_w0, vf_w1, vf_w2 = parameter_values[[4, 6, 13]]
            vf_b0, vf_b1, vf_b2 = parameter_values[[5, 7, 14]]
            pi_logstd = parameter_values[10]

            def relu(x): return np.maximum(x, 0)

            # get values of the mirrored observations
            def get_value(obs):
                vf_hid1 = relu(np.matmul(obs, vf_w0) + vf_b0)
                vf_hid2 = relu(np.matmul(vf_hid1, vf_w1) + vf_b1)
                values = np.matmul(vf_hid2, vf_w2) + vf_b2
                return values.flatten()

            def get_action_means(obs):
                pi_hid1 = relu(np.matmul(obs, pi_w0) + pi_b0)
                pi_hid2 = relu(np.matmul(pi_hid1, pi_w1) + pi_b1)
                means = np.matmul(pi_hid2, pi_w2) + pi_b2
                return means

            def neglogp(acts, mean, logstd):
                std = np.exp(logstd)
                return 0.5 * np.sum(np.square((acts - mean) / std), axis=-1) \
                       + 0.5 * np.log(2.0 * np.pi) * np.array(acts.shape[-1], dtype=np.float) \
                       + np.sum(logstd, axis=-1)

        for old_rollout in self.replay_buf:
            if old_rollout is None: continue

            self.prev_obs, self.prev_returns, self.prev_masks, self.prev_actions, \
            self.prev_values, self.prev_neglogpacs, self.prev_states, \
            self.prev_ep_infos, self.prev_true_reward = old_rollout

            if QUERY_NETS:
                self.prev_values = get_value(self.prev_obs)
                if not cfg.is_mod(cfg.MOD_QUERY_VF_ONLY):
                    act_means = get_action_means(self.prev_obs)
                    self.prev_neglogpacs = neglogp(self.prev_actions, act_means, pi_logstd)

                percentiles = [50, 75, 90, 95, 99, 100]
                if np.random.randint(0, 100, 1) == 77:
                    log('Neglogpacs Comparison (before clipping!)',
                        [f'neglogpacs orig: min {np.min(neglogpacs)}, '
                         f'mean {np.mean(neglogpacs)}, max {np.max(neglogpacs)}',
                         f'neglogpacs prev: min {np.min(self.prev_neglogpacs)}, '
                         f'mean {np.mean(self.prev_neglogpacs)}, '
                         f'max {np.max(self.prev_neglogpacs)}',
                         f'---\npercentiles {percentiles}:',
                         f'orig percentiles: {np.percentile(neglogpacs, percentiles)}',
                         f'prev percentiles: {np.percentile(self.prev_neglogpacs, percentiles)}',
                         ])

            obs = np.concatenate((obs, self.prev_obs))
            actions = np.concatenate((actions, self.prev_actions))
            returns = np.concatenate((returns, self.prev_returns))
            masks = np.concatenate((masks, self.prev_masks))
            values = np.concatenate((values, self.prev_values))
            neglogpacs = np.concatenate((neglogpacs, self.prev_neglogpacs))

        # remove mirrored experiences with too high neglogpacs
        FILTER_MIRRED_EXPS = True and QUERY_NETS and not cfg.is_mod(cfg.MOD_QUERY_VF_ONLY)
        if FILTER_MIRRED_EXPS:
            n_fresh_exps = int(len(neglogpacs) / (cfg.replay_buf_size+1))
            max_allowed_neglogpac = 5 * np.percentile(neglogpacs[:n_fresh_exps], 99)
            delete_act_indices = np.where(neglogpacs[n_fresh_exps:] > max_allowed_neglogpac)[0] + n_fresh_exps
            if np.random.randint(0, 10, 1)[0] == 7:
                log(f'Deleted {len(delete_act_indices)} mirrored actions '
                    f'with neglogpac > {max_allowed_neglogpac}')

            obs = np.delete(obs, delete_act_indices, axis=0)
            actions = np.delete(actions, delete_act_indices, axis=0)
            returns = np.delete(returns, delete_act_indices, axis=0)
            masks = np.delete(masks, delete_act_indices, axis=0)
            values = np.delete(values, delete_act_indices, axis=0)
            true_reward = np.delete(true_reward, delete_act_indices, axis=0)
            neglogpacs = np.delete(neglogpacs, delete_act_indices, axis=0)

        # add the current rollout in the replay buffer
        self.replay_buf = np.roll(self.replay_buf, shift=1)
        self.replay_buf[0] = rollout
        # self.prev_obs, self.prev_returns, self.prev_masks, self.prev_actions, \
        # self.prev_values, self.prev_neglogpacs, self.prev_states, \
        # self.prev_ep_infos, self.prev_true_reward = rollout

        return obs, returns, masks, actions, values, \
               neglogpacs, states, ep_infos, true_reward

    # ----------------------------------
    # OVERWRITTEN CLASSES
    # ----------------------------------

    def setup_model(self):
        """ Overwritten to double the batch size when experiences are mirrored. """
        super(CustomPPO2, self).setup_model()
        if self.mirror_experiences:
            log('Mirroring observations and actions to improve sample-efficiency.')
            self.n_batch *= 2

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        """
        Just copied from the stable_baselines.ppo2 implementation.
        Goal is to change some parts of it later.
        """
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            callback.on_training_start(locals(), globals())

            for update in range(1, n_updates + 1):
                minibatch_size = cfg.minibatch_size # self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()

                # try getting rollout 3 times
                tried_rollouts = 0
                while tried_rollouts < 1:
                    try:
                        # true_reward is the reward without discount
                        rollout = self.runner.run(callback)
                        break
                    except BrokenPipeError as bpe:
                        raise BrokenPipeError(f'Catched Broken Pipe Error.')
                    except Exception as ex:
                        # tried_rollouts += 1
                        # obs, returns, masks, actions, values, neglogpacs, \
                        # states, ep_infos, true_reward = rollout
                        # log(f'Rollout failed {tried_rollouts} times!',
                        #     [f'Catched exception: {ex}',
                        #      f'obs.shape: {obs.shape}',
                        #      f'ret.shape: {returns.shape}'])
                        traceback.print_exc()
                        # if isinstance(ex, BrokenPipeError):
                        #     # copy-pasted from the old blog here:
                        #     # http://newbebweb.blogspot.com/2012/02/python-head-ioerror-errno-32-broken.html
                        #     from signal import signal, SIGPIPE, SIG_DFL
                        #     signal(SIGPIPE, SIG_DFL)
                        #     print('Executing fix: Importing signal and disabling BrokenPipeError.')
                        #     for _ in range(10000):
                        #         print('', end='')

                # reset count once, rollout was successful
                tried_rollouts = 0

                # Unpack
                if self.mirror_experiences:
                    obs, returns, masks, actions, values, neglogpacs, \
                    states, ep_infos, true_reward = mirror_experiences(rollout, self)
                elif cfg.is_mod(cfg.MOD_EXP_REPLAY):
                    obs, returns, masks, actions, values, neglogpacs, \
                    states, ep_infos, true_reward = self.exp_replay(rollout)
                else:
                    obs, returns, masks, actions, values, neglogpacs, \
                    states, ep_infos, true_reward = rollout



                self.last_actions = actions

                if np.random.randint(low=1, high=20) == 7:
                    log(f'Values and Returns of collected experiences: ',
                    [f'min returns:\t{np.min(returns)}', f'min values:\t\t{np.min(values)}',
                     f'mean returns:\t{np.mean(returns)}', f'mean values:\t{np.mean(values)}',
                     f'max returns:\t{np.max(returns)}', f'max values:\t\t{np.max(values)}'])

                if cfg.is_mod(cfg.MOD_REFS_REPLAY):
                    # load ref experiences and treat them as real experiences
                    obs, actions, returns, masks, values, neglogpacs = \
                        generate_experiences_from_refs(rollout, self.ref_obs, self.ref_acts)

                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                self.n_batch = obs.shape[0]
                self.nminibatches = self.n_batch / minibatch_size
                if self.n_batch % minibatch_size != 0:
                    log("CAUTION!", ['Last minibatch might be too small!',
                                     f'Batch Size: \t{self.n_batch}',
                                     f'Minibatch Size:\t{minibatch_size}',
                                     f'Modulo: \t\t {self.n_batch % minibatch_size}'])
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    n_epochs = self.noptepochs
                    for epoch_num in range(n_epochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, minibatch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // minibatch_size)
                            end = start + minibatch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = minibatch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

            callback.on_training_end()
            return self
