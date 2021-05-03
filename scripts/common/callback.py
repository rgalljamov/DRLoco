from os import makedirs, remove, rename
import numpy as np
import wandb

from stable_baselines3 import PPO
from scripts.common import config as cfg, utils
from stable_baselines3.common.callbacks import BaseCallback

# define intervals/criteria for saving the model
# save everytime the agent achieved an additional 10% of the max possible return
MAX_RETURN = cfg.ep_dur_max * 1 * cfg.rew_scale
EP_RETURN_INCREMENT = 0.1 * MAX_RETURN
# 10% of max possible reward
MEAN_REW_INCREMENT = 0.1 * cfg.rew_scale

# define evaluation interval
EVAL_MORE_FREQUENT_THRES = 3.2e6
EVAL_INTERVAL_RARE = 400e3 if not cfg.DEBUG else 10e3
EVAL_INTERVAL_FREQUENT = 200e3
EVAL_INTERVAL_MOST_FREQUENT = 100e3
EVAL_INTERVAL = EVAL_INTERVAL_RARE

class TrainingMonitor(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingMonitor, self).__init__(verbose)
        # to control how often to save the model
        self.times_surpassed_ep_return_threshold = 0
        self.times_surpassed_mean_reward_threshold = 0
        # control evaluation
        self.n_steps_after_eval = EVAL_INTERVAL
        self.n_saved_models = 0
        self.moved_distances = []
        self.mean_walked_distance = 0
        self.min_walked_distance = 0
        self.mean_episode_duration = 0
        self.min_episode_duration = 0
        self.mean_walking_speed = 0
        self.min_walking_speed = 0
        self.mean_reward_means = 0
        self.count_stable_walks = 0
        self.summary_score = 0
        self.has_reached_stable_walking = False
        # collect the frequency of failed walks during evaluation
        self.failed_eval_runs_indices = []
        # log data less frequently
        self.skip_n_steps = 100
        self.skipped_steps = 99

    def _on_training_start(self) -> None:
        self.env = self.training_env

    def _on_step(self) -> bool:
        if cfg.DEBUG and self.num_timesteps > cfg.MAX_DEBUG_STEPS:
            raise SystemExit(f"Planned Exit after {cfg.MAX_DEBUG_STEPS} due to Debugging mode!")

        # reset the collection of episode lengths after 1M steps
        # goal: see a distribution of ep lens of last 1M steps,
        # ... not of the whole training so far...
        if self.num_timesteps % 2e6 < 10:
            self.env.set_attr('ep_lens', [])
            self.env.set_attr('et_phases', [])
            self.env.set_attr('difficult_rsi_phases', [])

        self.n_steps_after_eval += 1 * cfg.n_envs

        # skip n steps to reduce logging interval and speed up training
        if self.skipped_steps < self.skip_n_steps:
            self.skipped_steps += 1
            return True

        global EVAL_INTERVAL

        if self.n_steps_after_eval >= EVAL_INTERVAL and not cfg.DEBUG:
            self.n_steps_after_eval = 0
            walking_stably = self.eval_walking()
            # terminate training when stable walking has been learned
            if walking_stably:
                import wandb
                # log required num of steps to wandb
                if not self.has_reached_stable_walking:
                    wandb.run.summary['steps_to_convergence'] = self.num_timesteps
                    wandb.log({'log_steps_to_convergence': self.num_timesteps})
                    self.has_reached_stable_walking = True
                utils.log("WE COULD FINISH TRAINING EARLY!",
                          [f'Agent learned to stably walk '
                           f'after {self.num_timesteps} steps'
                           f'with mean step reward of {self.mean_reward_means}!'])

            if self.mean_walked_distance >= 20:
                EVAL_INTERVAL = EVAL_INTERVAL_RARE
            elif self.mean_walked_distance >= 10:
                EVAL_INTERVAL = EVAL_INTERVAL_MOST_FREQUENT
            elif self.mean_walked_distance >= 5:
                EVAL_INTERVAL = EVAL_INTERVAL_FREQUENT

        ep_len = self.get_mean('ep_len_smoothed')
        ep_ret = self.get_mean('ep_ret_smoothed')
        mean_rew = self.get_mean('mean_reward_smoothed')

        # avoid logging data during first episode
        if ep_len < {400: 60, 200:30, 50:8, 100:15}[cfg.CTRL_FREQ]:
            return True

        if not cfg.DEBUG: self.log_to_tb(mean_rew, ep_len, ep_ret)
        # do not save a model if its episode length was too short
        if ep_len > 1500:
            self.save_model_if_good(mean_rew, ep_ret)

        # reset counter of skipped steps after data was logged
        self.skipped_steps = 0

        return True


    def get_mean(self, attribute_name):
        try:
            values = self.env.get_attr(attribute_name)
            mean = np.mean(values)
            return mean
        except: return 0.333


    def log_to_tb(self, mean_rew, ep_len, ep_ret):
        moved_distance = self.get_mean('moved_distance_smooth')
        mean_abs_torque_smoothed = self.get_mean('mean_abs_ep_torque_smoothed')

        # Log scalar values
        NEW_LOG_STRUCTURE = True
        if NEW_LOG_STRUCTURE:
            logs = [
                tf.Summary.Value(tag='_det_eval/1. Summary Score',
                                 simple_value=self.summary_score),
                tf.Summary.Value(tag='_det_eval/2. stable walks count',
                                 simple_value=self.count_stable_walks),
                tf.Summary.Value(tag='_det_eval/4. mean eval distance',
                                 simple_value=self.mean_walked_distance),
                tf.Summary.Value(tag='_det_eval/5. MIN eval distance',
                                 simple_value=self.min_walked_distance),
                tf.Summary.Value(tag='_det_eval/3. mean step reward',
                                 simple_value=self.mean_reward_means),
                tf.Summary.Value(tag='_det_eval/6. mean episode duration',
                                 simple_value=self.mean_episode_duration),
                # tf.Summary.Value(tag='_det_eval/6. MIN episode duration',
                #                  simple_value=self.min_episode_duration),
                tf.Summary.Value(tag='_det_eval/7. mean walking speed',
                                 simple_value=self.mean_walking_speed),
                # tf.Summary.Value(tag='_det_eval/8. MIN walking speed',
                #                  simple_value=self.min_walking_speed),

                # tf.Summary.Value(tag='_monit/1. ',
                #                  simple_value=),

                tf.Summary.Value(tag='_train/1. moved distance (stochastic, smoothed 0.25)',
                                 simple_value=moved_distance/cfg.max_distance),
                tf.Summary.Value(tag='_train/2. episode length (smoothed 0.75)',
                                 simple_value=ep_len/cfg.ep_dur_max),
                tf.Summary.Value(tag='_train/3. step reward (smoothed 0.25)',
                                 simple_value=(mean_rew-cfg.alive_bonus)/cfg.rew_scale),
                tf.Summary.Value(tag='_train/4. episode return (smoothed 0.75)',
                                 simple_value=(ep_ret-ep_len*cfg.alive_bonus)/(cfg.ep_dur_max*cfg.rew_scale)),

                # tf.Summary.Value(tag='acts/2. mean abs episode joint torques (smoothed 0.75)',
                #                  simple_value=mean_abs_torque_smoothed)
            ]

            wandb.log({"_det_eval/1. walked distances": wandb.Histogram(
                np_histogram=np.histogram(self.moved_distances, bins=20))}, step=self.num_timesteps)

            # log reward components
            mean_ep_pos_rew = self.get_mean('mean_ep_pos_rew_smoothed')
            mean_ep_vel_rew = self.get_mean('mean_ep_vel_rew_smoothed')
            mean_ep_com_rew = self.get_mean('mean_ep_com_rew_smoothed')
            logs += [tf.Summary.Value(tag=f'_rews/1. mean ep pos rew ({cfg.n_envs}envs, smoothed 0.9)',
                                      simple_value=mean_ep_pos_rew),
                     tf.Summary.Value(tag=f'_rews/2. mean ep vel rew ({cfg.n_envs}envs, smoothed 0.9)',
                                      simple_value=mean_ep_vel_rew),
                     tf.Summary.Value(tag=f'_rews/3. mean ep com rew ({cfg.n_envs}envs, smoothed 0.9)',
                                      simple_value=mean_ep_com_rew),
                     ]

            model = self.model
            parameters = model.get_parameter_list()
            parameters = [param for param in parameters if 'logstd' in param.name]
            if cfg.is_mod(cfg.MOD_CONST_EXPLORE):
                logstd = cfg.init_logstd
            else:
                logstd = np.array(model.sess.run(parameters))[0]
            std = np.exp(logstd)
            mean_std = np.mean(std)
            std_of_stds = np.std(std)

            logs += [tf.Summary.Value(
                tag='acts/1. mean std of 8 action distributions', simple_value=mean_std)]

            actions = model.last_actions
            if actions is not None:
                abs_actions = np.abs(actions)
                pos_actions = actions[actions>=0]
                neg_actions = actions[actions<0]
                mean_action = np.mean(abs_actions)
                # monitor how many actions were saturated
                sat_acts_indices = np.where(abs_actions > 0.95)
                sat_acts_percentage = np.size(sat_acts_indices[0]) / np.size(actions)
                # std_actions = np.std(abs_actions)
                logs += [tf.Summary.Value(tag='acts/2. mean abs action (over batch and all dims)',
                                 simple_value=mean_action),
                         # tf.Summary.Value(tag='acts/a. mean action (over batch and all dims)',
                         #                  simple_value=np.mean(actions)),
                         tf.Summary.Value(tag='acts/4. mean POS action (over batch and all dims)',
                                 simple_value=np.mean(pos_actions)),
                         tf.Summary.Value(tag='acts/5. mean NEG action (over batch and all dims)',
                                 simple_value=np.mean(neg_actions)),

                         tf.Summary.Value(tag='acts/3. saturated actions percentage (all acts in a batch)',
                                 simple_value=sat_acts_percentage),
                         # tf.Summary.Value(tag='_own/0. std abs actions (whole batch and all dims)',
                         #         simple_value=std_actions)
                         ]
                wandb.log({"_hist/actions": wandb.Histogram(
                    np_histogram=np.histogram(actions, bins=200))}, step=self.num_timesteps)

                # get ET and RSI phases
                et_phases = self.env.get_attr('et_phases')
                et_phases_flat = [phase for env in et_phases for phase in env]
                # rsi_phases = self.env.get_attr('rsi_phases')
                # rsi_phases_flat = [phase for env in rsi_phases for phase in env]

                wandb.log({"_hist/ET_phases": wandb.Histogram(
                    np_histogram=np.histogram(et_phases_flat, bins=250))}, step=self.num_timesteps)
                # wandb.log({"_hist/RSI_phases": wandb.Histogram(
                #     np_histogram=np.histogram(rsi_phases_flat, bins=200))}, step=self.num_timesteps)
                if len(self.failed_eval_runs_indices) > 0:
                    wandb.log({"_hist/trials_below_20m": wandb.Histogram(
                        np_histogram=np.histogram(self.failed_eval_runs_indices,
                                                  bins=20, range=(0,19)))},
                        step=self.num_timesteps)

                ep_lens = self.env.get_attr('ep_lens')
                ep_lens = [ep_len for env_lens in ep_lens for ep_len in env_lens]
                wandb.log({"_hist/ep_lens": wandb.Histogram(
                    np_histogram=np.histogram(ep_lens, bins=40))}, step=self.num_timesteps)

                difficult_rsi_phases = self.env.get_attr('difficult_rsi_phases')
                difficult_rsi_phases = [phase for env_phases in difficult_rsi_phases for phase in env_phases]
                wandb.log({"_hist/difficult_rsi_phases": wandb.Histogram(
                    np_histogram=np.histogram(difficult_rsi_phases, bins=250, range=(0,1)))},
                    step=self.num_timesteps)

            if False: # np.random.randint(low=1, high=500) == 77:
                utils.log(f'Logstd after {int(self.num_timesteps/1e3)}k timesteps:',
                          [f'mean std: {mean_std}', f'std of stds: {std_of_stds}'])

            summary = tf.Summary(value=logs)
            self.locals['writer'].add_summary(summary, self.num_timesteps)


    def save_model_if_good(self, mean_rew, ep_ret):
        if cfg.DEBUG: return
        def get_mio_timesteps():
            return int(self.num_timesteps/1e6)

        ep_ret_thres = 0.6 * MAX_RETURN \
                       + int(EP_RETURN_INCREMENT * (self.times_surpassed_ep_return_threshold + 1))
        if ep_ret > ep_ret_thres:
            utils.save_model(self.model, cfg.save_path,
                             'ep_ret' + str(ep_ret_thres) + f'_{get_mio_timesteps()}M')
            self.times_surpassed_ep_return_threshold += 1
            print(f'NOT Saving model after surpassing EPISODE RETURN of {ep_ret_thres}.')
            # print('Model Path: ', cfg.save_path)

        # normalize reward
        mean_rew = (mean_rew - cfg.alive_bonus)/cfg.rew_scale
        mean_rew_thres = 0.4 * cfg.rew_scale \
                         + MEAN_REW_INCREMENT * (self.times_surpassed_mean_reward_threshold + 1)
        if mean_rew > (mean_rew_thres):
            # utils.save_model(self.model, cfg.save_path,
            #                  'mean_rew' + str(int(100*mean_rew_thres)) + f'_{get_mio_timesteps()}M')
            self.times_surpassed_mean_reward_threshold += 1
            print(f'NOT Saving model after surpassing MEAN REWARD of {mean_rew_thres}.')
            print('Model Path: ', cfg.save_path)


    def eval_walking(self):
        """
        Test the deterministic version of the current model:
        How far does it walk (in average and at least) without falling?
        @returns: If the training can be stopped as stable walking was achieved.
        """
        moved_distances, mean_rewards, ep_durs, mean_com_x_vels = [], [], [], []
        # save current model
        checkpoint = f'{int(self.num_timesteps/1e5)}'
        model_path, env_path = \
            utils.save_model(self.model, cfg.save_path, checkpoint, full=False)

        # load current model
        eval_model = PPO.load(load_path=model_path)

        eval_env = utils.load_env(checkpoint, cfg.save_path, cfg.env_id)
        mimic_env = eval_env.venv.envs[0].env
        mimic_env.activate_evaluation()

        # evaluate deterministically
        utils.log(f'Starting model evaluation, checkpoint {checkpoint}')
        obs = eval_env.reset()
        eval_n_times = cfg.EVAL_N_TIMES if self.num_timesteps > 1e6 else 10
        for i in range(eval_n_times):
            ep_dur = 0
            walked_distance = 0
            rewards = []
            while True:
                ep_dur += 1
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                if done:
                    moved_distances.append(walked_distance)
                    mean_rewards.append(np.mean(rewards))
                    ep_durs.append(ep_dur)
                    mean_com_x_vel = walked_distance/(ep_dur/cfg.CTRL_FREQ)
                    mean_com_x_vels.append(mean_com_x_vel)
                    break
                else:
                    # we cannot get the walked distance after episode termination,
                    # as when done=True is returned, the env was already reseted.
                    walked_distance = mimic_env.data.qpos[0]
                    # undo reward normalization, don't save last reward
                    reward = reward * np.sqrt(eval_env.ret_rms.var + 1e-8)
                    rewards.append(reward[0])

        # calculate min and mean walked distance
        self.moved_distances = moved_distances
        self.mean_walked_distance = np.mean(moved_distances)
        self.min_walked_distance = np.min(moved_distances)
        self.mean_episode_duration = np.mean(ep_durs)/cfg.ep_dur_max
        self.min_episode_duration = np.min(ep_durs)

        # calculate mean walking speed
        self.mean_walking_speed = np.mean(mean_com_x_vels)
        self.min_walking_speed = np.min(mean_com_x_vels)

        # calculate the average mean reward
        self.mean_reward_means = np.mean(mean_rewards)
        # normalize it
        self.mean_reward_means = (self.mean_reward_means - cfg.alive_bonus)/cfg.rew_scale

        # how many times 20m were reached
        min_required_distance = 20 if not cfg.is_mod(cfg.MOD_REFS_RAMP) else 9
        runs_below_min_distance = np.where(np.array(moved_distances) < min_required_distance)[0]
        count_runs_reached_min_distance = eval_n_times - len(runs_below_min_distance)
        runs_no_falling = np.where(
            (np.array(ep_durs) == cfg.ep_dur_max)
            & (np.array(moved_distances) >= 0.5*min_required_distance))[0]
        if eval_n_times == cfg.EVAL_N_TIMES:
            self.failed_eval_runs_indices = runs_below_min_distance.tolist()
        self.count_stable_walks = max(count_runs_reached_min_distance, len(runs_no_falling))
        dt = EVAL_INTERVAL / (
            EVAL_INTERVAL_RARE if self.num_timesteps < EVAL_MORE_FREQUENT_THRES else
            EVAL_INTERVAL_FREQUENT)
        self.summary_score += dt * 4 * self.mean_reward_means ** 2 \
                              * (self.count_stable_walks / cfg.EVAL_N_TIMES) ** 4 \

        if False: # runs_20m >= 20 and not cfg.is_mod(cfg.MOD_MIRR_QUERY_VF_ONLY):
            cfg.modification += f'/{cfg.MOD_QUERY_VF_ONLY}'
            utils.log('Starting to query VF only!',
                      [f'Stable walks: {count_runs_reached_min_distance}',
                       f'Mean distance: {self.mean_walked_distance}'])

        ## delete evaluation model if stable walking was not achieved yet
        # or too many models were saved already
        were_enough_models_saved = self.n_saved_models >= 5
        # or walking was not human-like
        walks_humanlike = self.mean_reward_means >= 0.5 * (1+self.n_saved_models/10)
        # print('Mean rewards during evaluation of the deterministic model: ', mean_rewards)
        min_dist = int(self.min_walked_distance)
        mean_dist = int(self.mean_walked_distance)
        # walked 10 times at least 20 meters without falling
        has_achieved_stable_walking = min_dist > 20
        # in average stable for 20 meters but not all 20 trials were over 20m
        has_reached_high_mean_distance = mean_dist > 20
        is_stable_humanlike_walking = self.count_stable_walks == eval_n_times and walks_humanlike
        # retain the model if it is good else delete it
        retain_model = is_stable_humanlike_walking and not were_enough_models_saved
        distances_report = [f'Min walked distance: {min_dist}m',
                            f'Mean walked distance: {mean_dist}m']
        if retain_model:
            utils.log('Saving Model:', distances_report)
            # rename model: add distances to the models names
            dists = f'_min{min_dist}mean{mean_dist}'
            new_model_path = model_path[:-4] + dists +'.zip'
            new_env_path = env_path + dists
            rename(model_path, new_model_path)
            rename(env_path, new_env_path)
            self.n_saved_models += 1
        else:
            utils.log('Deleting Model:', distances_report +
                      [f'Mean step reward: {self.mean_reward_means}',
                       f'Runs below 20m: {runs_below_min_distance}'])
            remove(model_path)
            remove(env_path)

        return is_stable_humanlike_walking


def _save_rews_n_rets(locals):
    # save all rewards and returns of the training, batch wise
    path_rews = cfg.save_path + 'metrics/train_rews.npy'
    path_rets = cfg.save_path + 'metrics/train_rets.npy'

    try:
        # load already saved rews and rets
        rews = np.load(path_rews)
        rets = np.load(path_rets)
        # combine saved with new rews and rets
        rews = np.concatenate((rews, locals['true_reward']))
        rets = np.concatenate((rets, locals['returns']))
    except Exception:
        rews = locals['true_reward']
        rets = locals['returns']

    # save
    np.save(path_rets, np.float16(rets))
    np.save(path_rews, np.float16(rews))




def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    Used to log relevant information during training
    :param _locals: (dict)
    :param _globals: (dict)
    """

    # save all rewards and returns of the training, batch wise
    _save_rews_n_rets(_locals)

    # Log other data about every 200k steps
    # todo: calc as a function of batch for ppo
    #  when updating stable-baselines doesn't provide another option
    #  and check how often TD3 and SAC raise the callback.
    saving_interval = 390 if cfg.use_default_hypers else 6
    n_updates = _locals['update']
    if n_updates % saving_interval == 0:

        model = _locals['self']
        utils.save_pi_weights(model, n_updates)

        # save the model and environment only for every second update (every 400k steps)
        if n_updates % (2*saving_interval) == 0:
            # save model
            model.save(save_path=cfg.save_path + 'models/model_' + str(n_updates))
            # save env
            env_path = cfg.save_path + 'envs/' + 'env_' + str(n_updates)
            makedirs(env_path)
            # save Running mean of observations and reward
            env = model.get_env()
            env.save_running_average(env_path)
            utils.log("Saved model after {} updates".format(n_updates))

    return True