import os.path
import wandb

import torch as th

import scripts.config.config
from scripts import eval
from scripts.config import config as cfgl
from scripts.config import hypers as cfg
from scripts.common import utils
from scripts.common.schedules import LinearDecay, ExponentialSchedule
from scripts.callback import TrainingMonitor
# from scripts.common.distributions import LOG_STD_MIN, LOG_STD_MAX

# from scripts.algos.custom_ppo2 import CustomPPO2
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from scripts.custom.policies import CustomActorCriticPolicy


def init_wandb(model):
    batch_size = model.n_steps * model.n_envs
    params = {
        "path": cfg.save_path,
        "mod": cfg.modification,
        "ctrl_freq": cfg.CTRL_FREQ,
        "lr0": cfg.lr_start,
        "lr1": cfg.lr_final,
        'hid_sizes': cfg.hid_layer_sizes,
        # 'peak_joint_torques': cfg.peak_joint_torques,
        'walker_xml_file': cfg.walker_xml_file,
        "noptepochs": cfg.noptepochs,
        "batch_size": batch_size,
        "cfg.batch_size": cfg.batch_size,
        # "n_mini_batches": model.nminibatches,
        "cfg.minibatch_size": cfg.minibatch_size,
        # "mini_batch_size": int(batch_size / model.nminibatches),
        "mio_steps": cfg.mio_samples,
        "ent_coef": model.ent_coef,
        "ep_dur": cfg.ep_dur_max,
        "imit_rew": cfg.rew_weights,
        "logstd": cfg.init_logstd,
        # "min_logstd": LOG_STD_MIN,
        # "max_logstd": LOG_STD_MAX,
        "env": cfg.env_abbrev,
        "gam": model.gamma,
        "lam": model.gae_lambda,
        "n_envs": model.n_envs,
        "seed": model.seed,
        "policy": model.policy,
        "n_steps": model.n_steps,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        # "nminibatches": model.nminibatches,
        "clip0": cfg.clip_start,
        "clip1": cfg.clip_end,
        }

    wandb.init(config=params, sync_tensorboard=True, name=cfgl.WB_RUN_NAME,
               project=cfgl.WB_PROJECT_NAME, notes=cfgl.WB_RUN_DESCRIPTION)


def train():

    # create model directories
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
        os.makedirs(cfg.save_path + 'metrics')
        os.makedirs(cfg.save_path + 'models')
        os.makedirs(cfg.save_path + 'models/params')
        os.makedirs(cfg.save_path + 'envs')

    # setup environment
    env = utils.vec_env(cfg.env_id, norm_rew=True, num_envs=cfg.n_envs)

    # setup model/algorithm
    training_timesteps = int(cfg.mio_samples * 1e6)
    lr_start = cfg.lr_start
    lr_end = cfg.lr_final

    learning_rate_schedule = LinearDecay(lr_start, lr_end).value
    clip_schedule = ExponentialSchedule(cfg.clip_start, cfg.clip_end, cfg.clip_exp_slope).value

    use_custom_policy = cfg.is_mod(cfg.MOD_CUSTOM_POLICY)
    policy_kwargs = {'log_std_init':cfg.init_logstd} if use_custom_policy else \
                    {'net_arch': [{'vf': cfg.hid_layer_sizes, 'pi': cfg.hid_layer_sizes}],
                    'activation_fn': th.nn.Tanh, 'log_std_init':cfg.init_logstd}

    model = PPO(CustomActorCriticPolicy if use_custom_policy else MlpPolicy,
                       env, verbose=1,
                       n_steps = cfg.batch_size//cfg.n_envs, # num of steps per env per update
                       batch_size=cfg.minibatch_size, # minibatch size (batch size per training step)
                       policy_kwargs=policy_kwargs,
                       learning_rate=learning_rate_schedule, ent_coef=cfg.ent_coef,
                       gamma=cfg.gamma, n_epochs=cfg.noptepochs,
                       clip_range_vf=clip_schedule if cfg.is_mod(cfg.MOD_CLIPRANGE_SCHED) else cfg.cliprange,
                       clip_range=clip_schedule if cfg.is_mod(cfg.MOD_CLIPRANGE_SCHED) else cfg.cliprange,
                       tensorboard_log=cfg.save_path + 'tb_logs/')

    # init wandb
    if not cfg.DEBUG:
        init_wandb(model)

    # print model path and modification parameters
    utils.log('RUN DESCRIPTION: \n' + cfgl.WB_RUN_DESCRIPTION)
    utils.log('Training started',
              ['Model: ' + cfg.save_path, 'Modifications: ' + cfg.modification])

    # save model and weights before training
    if not cfg.DEBUG:
        utils.save_model(model, cfg.save_path, scripts.config.config.init_checkpoint)

    # train model
    model.learn(total_timesteps=training_timesteps, callback=TrainingMonitor())

    # save model after training
    utils.save_model(model, cfg.save_path, scripts.config.config.final_checkpoint)

    # close environment
    env.close()

    # evaluate last saved model
    eval.eval_model()


if __name__ == '__main__':
    train()