# suppress the annoying TF Warnings at startup
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scripts.plots.data_struct import Approach
from scripts.plots.plot import plot_violin as violin
from scripts.common.utils import config_pyplot, change_plot_properties
from scripts.plots import plot
import seaborn as sns
import numpy as np
plt = config_pyplot(fig_size=False)

APD_BSLN = 'pd_bsln'
APD_NORM_ANGS = 'pd_norm_angs'
APD_NORM_ANGS_EXPMORE = 'pd_norm_angs_expmore'
APD_NORM_ANGS_SIGN_EXPMORE = 'pd_norm_angs_sign_expmore'
APD_NORM_DELTA = 'pd_norm_delta'
APT_BSLN = 'trq_bsln'
APT_BSLN_HALF_BS = 'trq_bsln_half_bs'
APT_CLIP_DEC = 'clip_dec'
APT_MRR_STEPS = 'mrr_steps'
APT_DUP = 'mrr_exps'
APT_SYM_DUP = 'mrr_exps_sym'
APT_EXP_REPLAY = 'exp_replay'
APT_EXP_REPLAY_QUERY_VF = 'exp_replay_vf'


run_names_dict = {APD_BSLN: 'BSLN, init std = 1',
                  APD_NORM_ANGS: 'BSLN - normed target angles',
                  APD_NORM_ANGS_EXPMORE: 'BSLN - normed target angles - init std = 1',
                  APD_NORM_ANGS_SIGN_EXPMORE: 'BSLN - SIGN normed target angles - init std = 1',
                  APD_NORM_DELTA: 'normed deltas',
                  APT_BSLN: 'bsln',
                  APT_BSLN_HALF_BS:  'bsln, half exps',
                  APT_CLIP_DEC:  'exp clip decay (VF too): 0.55 - 0.1',
                  APT_MRR_STEPS:  'MRR steps, half BS',
                  APT_DUP:  'mirr exps - no query nets',
                  APT_SYM_DUP:  'SYM mirr exps - no query nets',
                  APT_EXP_REPLAY:  'NEW Replay BUF1, NO query, ent_coef-0.0075, adjust BS',
                  APT_EXP_REPLAY_QUERY_VF:  'NEW Replay BUF1, query VF, ent_coef-0.0075, adjust BS'
}

approach_names_dict = {APD_BSLN: 'Baseline ($\sigma^2_0 = 1.0$)',
                  APD_NORM_ANGS: 'Normed Target Angles ($\sigma^2_0 = 0.5$)',
                  APD_NORM_ANGS_EXPMORE: 'Normed Target Angles ($\sigma^2_0 = 1.0$)',
                  APD_NORM_ANGS_SIGN_EXPMORE: 'Sign-Normed Target Angles ($\sigma^2_0 = 1.0$)',
                  APD_NORM_DELTA: 'Normalized Angle Deltas ($\sigma^2_0 = 0.5$)',
                  APT_BSLN: 'Baseline',
                   APT_BSLN_HALF_BS: r'Baseline ($\frac{1}{2}$ BS)',
                   APT_CLIP_DEC: 'Cliprange Decay',
                   APT_MRR_STEPS: 'Mirror (PHASE)',
                   APT_DUP: 'Mirror (DUP)',
                   APT_SYM_DUP: 'Mirror (DUP, SYM)',
                   APT_EXP_REPLAY: 'Experience Replay',
                   APT_EXP_REPLAY_QUERY_VF: 'Experience Replay (query VF)'
                       }

# metric labels
MET_SUM_SCORE = '_det_eval/1. AUC stable walks count'
MET_STABLE_WALKS = '_det_eval/2. stable walks count'
MET_STEP_REW = '_det_eval/3. mean step reward'
MET_MEAN_DIST = '_det_eval/4. mean eval distance'
MET_REW_POS = '_rews/1. mean ep pos rew (8envs, smoothed 0.9)'
MET_REW_VEL = '_rews/2. mean ep vel rew (8envs, smoothed 0.9)'
MET_REW_COM = '_rews/3. mean ep com rew (8envs, smoothed 0.9)'
MET_TRAIN_EPRET = '_train/4. episode return (smoothed 0.75)'
MET_TRAIN_STEPREW = '_train/3. step reward (smoothed 0.25)'
MET_TRAIN_DIST = '_train/1. moved distance (stochastic, smoothed 0.25)'
MET_TRAIN_EPLEN = '_train/2. episode length (smoothed 0.75)'
MET_STEPS_TO_CONV = 'log_steps_to_convergence'
MET_PI_STD = 'acts/1. mean std of 8 action distributions'
# MET_SAT_ACTS = 'acts/3. saturated actions percentage (all acts in a batch)'

metric_labels = [MET_SUM_SCORE, MET_STABLE_WALKS, MET_STEP_REW, MET_MEAN_DIST, MET_REW_POS, MET_REW_VEL,
           MET_REW_COM, MET_TRAIN_EPRET, MET_TRAIN_STEPREW, MET_TRAIN_DIST, MET_TRAIN_EPLEN,
           MET_STEPS_TO_CONV, MET_PI_STD]

metric_names_dict = {MET_SUM_SCORE: 'Summary Score', MET_STABLE_WALKS: '# Stable Episodes',
                MET_STEP_REW: 'Mean Imitation Reward', MET_MEAN_DIST: 'Eval Distance [m]',
                MET_REW_POS: 'Average Position Reward', MET_REW_VEL: 'Average Velocity Reward', 
                MET_REW_COM: 'Average COM Reward', MET_TRAIN_EPRET: 'Normalized Episode Return',
                MET_TRAIN_STEPREW: 'Train Step Reward', MET_TRAIN_DIST: 'Train Distance [m]',
                MET_TRAIN_EPLEN:'Train Episode Length', MET_STEPS_TO_CONV:'Steps to Convergence',
                MET_PI_STD: 'Policy STD'}


def check_data_for_completeness(approach, mio_train_steps=16):
    n_metrics = len(approach.metrics)
    change_plot_properties(font_size=-4, tick_size=-4, legend_fontsize=-2)
    for i, metric in enumerate(approach.metrics):
        subplot = plt.subplot(3, int(n_metrics / 3) + 1, i + 1)
        if isinstance(metric.mean, np.ndarray) and len(metric.mean) > 1:
            x = np.linspace(0, mio_train_steps*1e6, len(metric.mean))
            subplot.plot(x, metric.mean)
            subplot.plot(x, metric.mean_fltrd)
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, metric.mean_fltrd + metric.std, metric.mean_fltrd - metric.std,
                                 color=mean_color, alpha=0.25)
            tick_distance_mio = mio_train_steps / 4
        else:
            subplot.scatter(metric.data, np.arange(len(metric.data)))
        subplot.title.set_text(metric_names_dict[metric.label])
        subplot.set_xticks(np.arange(5) * tick_distance_mio * 1e6)
        subplot.set_xticklabels([f'{x}M' for x in np.arange(5) * tick_distance_mio])
    plt.show()


def download_approach_data(approach_name, project_name, run_name=None):
    train_steps = 16 if project_name == 'pd_approaches' else 8
    if run_name is None: run_name = run_names_dict[approach_name]
    approach = Approach(approach_name, project_name, run_name, metric_labels)
    check_data_for_completeness(approach, train_steps)
    approach.save()

def download_multiple_approaches(project_name, approach_names: list):
    run_names = [run_names_dict[approach] for approach in approach_names]
    for i in range(len(run_names)):
        approach_name = approach_names[i]
        print('Downloading approach:', approach_name)
        download_approach_data(approach_name, project_name, run_names[i])

def download_approaches():
    project_name = "pd_approaches"
    # ap_names = [APD_BSLN, APD_NORM_ANGS, APD_NORM_ANGS_EXPMORE,
    #             APD_NORM_ANGS_SIGN_EXPMORE, APD_NORM_DELTA]
    project_name = "final3d_trq"
    ap_names = [APT_BSLN_HALF_BS, APT_CLIP_DEC, APT_MRR_STEPS ,
                APT_DUP ,APT_SYM_DUP,APT_EXP_REPLAY,APT_EXP_REPLAY_QUERY_VF,]
    download_multiple_approaches(project_name, ap_names)



def show_summary_score_advantages():
    # to show how good the summary score distinguishes between runs
    ap_names = [APD_NORM_ANGS_EXPMORE, APD_NORM_ANGS_SIGN_EXPMORE, APD_NORM_DELTA]
    metric_labels = [MET_SUM_SCORE, MET_TRAIN_EPRET, MET_STABLE_WALKS, MET_STEP_REW]
    aps = [Approach(name) for name in ap_names]
    n_metrics = len(metric_labels)
    change_plot_properties(font_size=-2, tick_size=-2, line_width=+1)
    for i, metric_label in enumerate(metric_labels):
        subplot = plt.subplot(1, n_metrics, i + 1)
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd * 0
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}M' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])
            # subplot.set_xlabel(r'Training Timesteps [x$10^6$]')
    plt.show()


def compare_all_metrics():
    plt = config_pyplot(fig_size=1)
    ap_names = [APD_BSLN, APD_NORM_ANGS, APD_NORM_DELTA]
    # approach_names_dict[APT_BSLN] = 'Joint Torque'
    # approach_names_dict[APD_BSLN] = 'Target Angles'
    aps = [Approach(name) for name in ap_names]
    # metric_labels = [MET_TRAIN_EPRET, MET_STABLE_WALKS, MET_STEP_REW]
    n_metrics = len(metric_labels)
    subplots = []
    change_plot_properties(font_size=-4, tick_size=-4, legend_fontsize=-6, line_width=+1)
    for i, metric_label in enumerate(metric_labels):
        subplot = plt.subplot(1, n_metrics, i + 1) if len(metric_labels) < 5 \
            else plt.subplot(3, int(n_metrics / 3) + 1, i + 1)
        subplots.append(subplot)
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                subplot.set_ylabel(metric_names_dict[metric.label][-20:])
                continue
            n_steps = metric.train_duration_mio * 1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 4])
            subplot.set_ylabel(metric_names_dict[metric.label][-20:])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
    # subplots[1].set_xlabel(r'Training Timesteps [x$10^6$]')
    subplots[-1].legend([approach_names_dict[ap] for ap in ap_names],
                        bbox_to_anchor=(1.2, 0.8))
    plt.show()


def compare_action_spaces():
    ap_names = [APD_BSLN, APD_NORM_ANGS, APD_NORM_DELTA, APT_BSLN]
    # ap_names = [APD_BSLN, APT_BSLN]
    approach_names_dict[APT_BSLN] = 'Joint Torque'
    approach_names_dict[APD_BSLN] = 'Target Angles'
    approach_names_dict[APD_NORM_ANGS] = 'Target Angles'
    approach_names_dict[APD_BSLN] = 'Target Angles'
    aps = [Approach(name) for name in ap_names]
    # metric_labels = [MET_TRAIN_EPRET, MET_STABLE_WALKS, MET_STEP_REW]
    n_metrics = len(metric_labels)
    subplots = []
    change_plot_properties(font_size=-4, tick_size=-4, legend_fontsize=-2, line_width=+1)
    for i, metric_label in enumerate(metric_labels):
        subplot = plt.subplot(1, n_metrics, i + 1) if len(metric_labels) < 5 \
                    else plt.subplot(3, int(n_metrics / 3) + 1, i + 1)
        subplots.append(subplot)
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])
    subplots[1].set_xlabel(r'Training Timesteps [x$10^6$]')
    subplots[-1].legend([approach_names_dict[ap] for ap in ap_names],
                        bbox_to_anchor=(1.2, 0.8))
    plt.show()


def compare_baselines_8plots():
    ap_names = [APD_BSLN, APT_BSLN]
    approach_names_dict[APT_BSLN] = 'Joint Torque'
    approach_names_dict[APD_BSLN] = 'Target Angles'
    aps = [Approach(name) for name in ap_names]
    metric_labels = [MET_SUM_SCORE, MET_STABLE_WALKS, MET_STEP_REW,
                     MET_REW_POS, MET_REW_VEL,
                     MET_TRAIN_EPRET, MET_TRAIN_DIST, MET_TRAIN_EPLEN]
    n_metrics = len(metric_labels)
    subplots = []
    change_plot_properties(font_size=-4, tick_size=-4, legend_fontsize=-2, line_width=+1)
    for i, metric_label in enumerate(metric_labels):
        subplot = plt.subplot(1, n_metrics, i + 1) if len(metric_labels) < 5 \
                    else plt.subplot(2, 4, i + 1)
        subplots.append(subplot)
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])
    subplots[1].set_xlabel(r'Training Timesteps [x$10^6$]')
    subplots[-1].legend([approach_names_dict[ap] for ap in ap_names])
    plt.show()

def plot_return_only():
    plt = config_pyplot(fig_size=0.5)
    ap_names = [APD_NORM_ANGS, APD_NORM_DELTA, APT_BSLN]
    # ap_names = [APT_BSLN, APT_BSLN_HALF_BS, APT_DUP, APT_MRR_STEPS]
    approach_names_dict[APD_BSLN] = 'Target Angles'
    approach_names_dict[APT_BSLN] = 'Joint Torques (Ours)'
    approach_names_dict[APD_NORM_ANGS] = 'Target Angles'
    approach_names_dict[APD_NORM_DELTA] = 'Angle Deltas (Ours)'
    aps = [Approach(name) for name in ap_names]
    max_dur = 1e6 * max([ap.train_duration_mio for ap in aps])
    metric_labels = [MET_STABLE_WALKS, MET_STEP_REW, MET_SUM_SCORE, MET_TRAIN_EPRET]
    # metric_labels = [MET_REW_POS, MET_REW_VEL, MET_REW_COM]
    n_metrics = len(metric_labels)
    plot_rews = n_metrics == 3


    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=0, tick_size=-2, legend_fontsize=-3, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})

    # heights for the lines
    line_start = np.zeros(n_metrics)
    line_start[2] = 0.1
    line_stop = [30, 20, 0.81, 0.8]

    fig, subplots = plt.subplots(1, 1) #n_metrics)
    for i, metric_label in enumerate(metric_labels):
        if i != 3: continue
        subplot = subplots#[i]
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])

            if plot_rews:
                conv_timestep = metric.approach.steps_to_conv_mean
                train_dur = (metric.approach.train_duration_mio * 1e6)
                n_points = len(metric.mean_fltrd)
                index = int(n_points * conv_timestep / train_dur)
                mean_conv = metric.mean_fltrd[index]

                subplot.scatter(conv_timestep, mean_conv, s=100, marker='o', color=mean_color, zorder=10)
                subplot.plot(np.linspace(0, conv_timestep, 3), np.ones(3) * mean_conv,
                             c=mean_color, linestyle='--', linewidth=2, zorder=1)

                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)

                    # horizontal
                    subplot.plot(np.linspace(0, rew75_timestep, 3),
                        np.ones(3) * mean75, c=mean_color, linestyle=':', linewidth=2, zorder=0)

            # Mark important points in the reward and return graph
            if i in [1,3] and len(metric_labels) == 4:
                # show rew at convergence
                conv_timestep = metric.approach.steps_to_conv_mean
                train_dur = (metric.approach.train_duration_mio * 1e6)
                n_points = len(metric.mean_fltrd)
                index = int(n_points * conv_timestep / train_dur)
                mean_conv = metric.mean_fltrd[index]

                subplot.scatter(conv_timestep, mean_conv, s=100, marker='o', color=mean_color, zorder=10)
                subplot.plot(np.linspace(0, conv_timestep, 3), np.ones(3)*mean_conv,
                             c=mean_color, linestyle='--', linewidth=2, zorder=0)

            # Plot the 75% line
            if i == 1 and len(metric_labels) == 4:
                # subplot.plot(np.linspace(0, train_dur, 3), np.ones(3) * 0.75,
                #              c='gray', linestyle=':', linewidth=1.25, zorder=0)
                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)
                    subplot.text(rew75_timestep, 0.05, f'{metric.approach.steps_to_75rew_mean}',
                             ha='center', fontsize=tick_size+2, color=mean_color)
                    # vertical
                    subplot.plot(np.ones(3) * rew75_timestep,
                              np.linspace(0.105, mean75, 3), c=mean_color,
                              linestyle=':', linewidth=2, zorder=0)

            # Plot horizontal 75% line in the return graph
            if i == 3 and len(metric_labels) == 4:
                # subplot.plot(np.linspace(0, train_dur, 3), np.ones(3) * 0.75,
                #              c='gray', linestyle=':', linewidth=1.25, zorder=0)
                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='s', color=mean_color, zorder=10)

                    # horizontal
                    subplot.plot(np.linspace(0, rew75_timestep, 3),
                                 np.ones(3) * mean75, c=mean_color,
                                 linestyle=':', linewidth=3.5, zorder=0)



                # # Plot vertical lines indicating convergence and 75% rew
                # # vertical
                # subplot.plot(np.ones(3)*conv_timestep,
                #              np.linspace(line_start[i], mean_conv,3), c=mean_color,
                #              linestyle='--', linewidth=1.4, zorder=0)
                # # horizontal
                # subplot.plot(np.linspace(0, max_dur, 3), np.ones(3)*mean_conv,
                #              c=mean_color, linestyle='--', linewidth=1.5, zorder=0)


    x_label = r'Training Timesteps [x$10^6$]'
    # plt.gcf().tight_layout(rect=[0.1, 0.5, 0.95, 1])
    fig.text(0.5, 0.04, x_label, ha='center', fontsize=font_size-1)
    plt.subplots_adjust(wspace=0.33, top=0.99, left=0.05, right=0.99, bottom=0.18)
    legend_texts = [approach_names_dict[ap] for ap in ap_names]
    # assert 'ours' in ''.join(legend_texts).lower()
    # if plot_rews:
    #     subplots[-1].legend(np.array(subplots[-1].get_lines())[[0,2,5] if len(aps) == 3 else [0,3,6,9]], legend_texts, fancybox=True, framealpha=0.6,
    #                         loc='upper right')
    # else:
    #     subplots[-1].legend(subplots[2].get_lines(), legend_texts, fancybox=True, framealpha=0.6,
    #                         loc='upper left', bbox_to_anchor=(0.04, 0.42))
    plt.show()


def compare_main_plots():
    plt = config_pyplot(fig_size=0.5)
    # ap_names = [APD_NORM_ANGS, APD_NORM_DELTA, APT_BSLN]
    # ap_names = [APT_BSLN, APT_BSLN_HALF_BS, APT_DUP, APT_MRR_STEPS]
    ap_names = [APT_BSLN, APT_DUP, APT_MRR_STEPS]

    # ap_names = [APD_BSLN, APD_NORM_DELTA, APT_BSLN, APT_MRR_STEPS, APT_DUP]
    # approach_names_dict[APT_BSLN_HALF_BS] = 'Baseline (1/2 BS)'
    # approach_names_dict[APD_BSLN] = 'Baseline\nTarget Angles'
    # approach_names_dict[APD_NORM_DELTA] = 'Angle\nDeltas'
    # approach_names_dict[APT_BSLN] = 'Torque'
    # approach_names_dict[APT_MRR_STEPS] = 'Mirror\nPolicy'
    # approach_names_dict[APT_DUP] = 'Mirror\nExperiences'

    approach_names_dict[APD_BSLN] = 'Target Angles'
    approach_names_dict[APT_BSLN] = 'Joint Torques'
    approach_names_dict[APD_NORM_ANGS] = 'Target Angles'
    approach_names_dict[APD_NORM_DELTA] = 'Angle Deltas'
    aps = [Approach(name) for name in ap_names]
    max_dur = 1e6 * max([ap.train_duration_mio for ap in aps])
    metric_labels = [MET_STABLE_WALKS, MET_STEP_REW, MET_SUM_SCORE, MET_TRAIN_EPRET]
    # metric_labels = [MET_REW_POS, MET_REW_VEL, MET_REW_COM]
    n_metrics = len(metric_labels)
    plot_rews = n_metrics == 3


    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=-4, tick_size=-4, legend_fontsize=-3, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})

    # heights for the lines
    line_start = np.zeros(n_metrics)
    line_start[2] = 0.1
    line_stop = [30, 20, 0.81, 0.8]

    fig, subplots = plt.subplots(1, n_metrics)
    for i, metric_label in enumerate(metric_labels):
        subplot = subplots[i]
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            # mean = metric.data[0,:]
            subplot.plot(x, mean)
            mean_color = subplot.get_lines()[-1].get_color()
            show_std = True
            if show_std:
                std = metric.std_fltrd
                subplot.fill_between(x, mean + std, mean - std,
                                     color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 2e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 2])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])
            # subplot.set_ylabel(f'({"abcd"[i]}) ' + metric_names_dict[metric.label])

            if plot_rews:
                conv_timestep = metric.approach.steps_to_conv_mean
                train_dur = (metric.approach.train_duration_mio * 1e6)
                n_points = len(metric.mean_fltrd)
                index = int(n_points * conv_timestep / train_dur)
                mean_conv = metric.mean_fltrd[index]

                subplot.scatter(conv_timestep, mean_conv, s=100, marker='o', color=mean_color, zorder=10)
                subplot.plot(np.linspace(0, conv_timestep, 3), np.ones(3) * mean_conv,
                             c=mean_color, linestyle='--', linewidth=2, zorder=1)

                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)

                    # horizontal
                    subplot.plot(np.linspace(0, rew75_timestep, 3),
                        np.ones(3) * mean75, c=mean_color, linestyle=':', linewidth=2, zorder=0)

            # Mark important points in the stable walks graph
            if i == 0 and len(metric_labels) == 4:
                # show rew at convergence
                conv_timestep = metric.approach.steps_to_conv_mean
                subplot.set_ylim([-0.5, 22])
                subplot.scatter(conv_timestep, 0, s=90, marker='o', color=mean_color, zorder=10)
                subplot.plot(np.ones(3) * conv_timestep, np.linspace(0, 20, 3),
                             c=mean_color, linestyle='--', linewidth=1.5, zorder=0)

            # Mark important points in the reward and return graph
            if i in [1,3] and len(metric_labels) == 4:
                # show rew at convergence
                conv_timestep = metric.approach.steps_to_conv_mean
                train_dur = (metric.approach.train_duration_mio * 1e6)
                n_points = len(metric.mean_fltrd)
                index = int(n_points * conv_timestep / train_dur)
                mean_conv = metric.mean_fltrd[index]

                subplot.scatter(conv_timestep, mean_conv, s=90, marker='o', color=mean_color, zorder=10)
                subplot.plot(np.linspace(0, conv_timestep, 3), np.ones(3)*mean_conv,
                             c=mean_color, linestyle='--', linewidth=1.5, zorder=0)

            # Plot the 75% line
            if False: # i == 1 and len(metric_labels) == 4:
                # subplot.plot(np.linspace(0, train_dur, 3), np.ones(3) * 0.75,
                #              c='gray', linestyle=':', linewidth=1.25, zorder=0)
                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)
                    subplot.text(rew75_timestep, 0.05, f'{metric.approach.steps_to_75rew_mean}',
                             ha='center', fontsize=tick_size+2, color=mean_color)
                    # vertical
                    subplot.plot(np.ones(3) * rew75_timestep,
                              np.linspace(0.105, mean75, 3), c=mean_color,
                              linestyle=':', linewidth=2, zorder=0)

            # Plot horizontal 75% line in the return graph
            if i == 3 and len(metric_labels) == 4:
                # subplot.plot(np.linspace(0, train_dur, 3), np.ones(3) * 0.75,
                #              c='gray', linestyle=':', linewidth=1.25, zorder=0)
                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)

                    # horizontal
                    subplot.plot(np.linspace(0, rew75_timestep, 3),
                                 np.ones(3) * mean75, c=mean_color,
                                 linestyle=':', linewidth=2, zorder=0)



                # # Plot vertical lines indicating convergence and 75% rew
                # # vertical
                # subplot.plot(np.ones(3)*conv_timestep,
                #              np.linspace(line_start[i], mean_conv,3), c=mean_color,
                #              linestyle='--', linewidth=1.4, zorder=0)
                # # horizontal
                # subplot.plot(np.linspace(0, max_dur, 3), np.ones(3)*mean_conv,
                #              c=mean_color, linestyle='--', linewidth=1.5, zorder=0)


    x_label = r'Training Timesteps [x$10^6$]'
    # plt.gcf().tight_layout(rect=[0.1, 0.5, 0.95, 1])
    fig.text(0.5, 0.04, x_label, ha='center', fontsize=font_size-1)
    plt.subplots_adjust(wspace=0.33, top=0.99, left=0.05, right=0.99, bottom=0.18)
    plt.show()
    exit(33)
    legend_texts = [approach_names_dict[ap] for ap in ap_names]
    # assert 'ours' in ''.join(legend_texts).lower()
    if plot_rews:
        subplots[-1].legend(np.array(subplots[-1].get_lines())[[0,2,5] if len(aps) == 3 else [0,3,6,9]], legend_texts, fancybox=True, framealpha=0.6,
                            loc='upper right')
    else:
        subplots[-1].legend(subplots[2].get_lines(), legend_texts, fancybox=True, framealpha=0.6,
                            loc='upper left', bbox_to_anchor=(0.04, 0.42))
    plt.show()



def compare_main_torque_plots():
    plt = config_pyplot(fig_size=0.5)
    ap_names = [APT_BSLN, APT_BSLN_HALF_BS, APT_DUP, APT_MRR_STEPS]
    approach_names_dict[APD_BSLN] = 'Target Angles'
    # approach_names_dict[APT_BSLN] = 'Joint Torques (Ours)'
    approach_names_dict[APD_NORM_ANGS] = 'Target Angles'
    approach_names_dict[APD_NORM_DELTA] = 'Angle Deltas (Ours)'
    aps = [Approach(name) for name in ap_names]
    train_dur = 1e6 * max([ap.train_duration_mio for ap in aps])
    metric_labels = [MET_STABLE_WALKS, MET_STEP_REW, MET_SUM_SCORE, MET_TRAIN_EPRET]
    # metric_labels = [MET_REW_POS, MET_REW_VEL, MET_REW_COM]
    n_metrics = len(metric_labels)
    plot_rews = n_metrics == 3


    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=0, tick_size=-2, legend_fontsize=-3, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})

    # heights for the lines
    line_start = np.zeros(n_metrics)
    line_start[2] = 0.1
    line_stop = [30, 20, 0.81, 0.8]

    fig, subplots = plt.subplots(1, n_metrics)
    for i, metric_label in enumerate(metric_labels):
        subplot = subplots[i]
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for i_met, metric in enumerate(metrics):
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 2e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 2])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])
            # subplot.set_ylabel(f'({"abcd"[i]}) ' + metric_names_dict[metric.label])

            n_points = len(metric.mean_fltrd)

            if plot_rews:
                conv_timestep = metric.approach.steps_to_conv_mean
                train_dur = (metric.approach.train_duration_mio * 1e6)
                n_points = len(metric.mean_fltrd)
                index = int(n_points * conv_timestep / train_dur)
                mean_conv = metric.mean_fltrd[index]

                subplot.scatter(conv_timestep, mean_conv, s=100, marker='o', color=mean_color, zorder=10)
                subplot.plot(np.linspace(0, conv_timestep, 3), np.ones(3) * mean_conv,
                             c=mean_color, linestyle='--', linewidth=2, zorder=1)

                # show time after 75% of reward was reached
                # but only when it is reached at all
                # rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                # if rew75_timestep != train_dur:
                #     index = int(n_points * rew75_timestep / train_dur)
                #     mean75 = metric.mean_fltrd[index]
                #     # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                #     subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)
                #
                #     # horizontal
                #     subplot.plot(np.linspace(0, rew75_timestep, 3),
                #         np.ones(3) * mean75, c=mean_color, linestyle=':', linewidth=2, zorder=0)

            # Mark important points in the stable walks graph
            if i == 0 and len(metric_labels) == 4:
                # show rew at convergence
                conv_timestep = metric.approach.steps_to_conv_mean
                subplot.set_ylim([-0.5, 22])
                subplot.scatter(conv_timestep, 0, s=90, marker='o', color=mean_color, zorder=10)
                subplot.plot(np.ones(3) * conv_timestep, np.linspace(0, 20, 3),
                             c=mean_color, linestyle='--', linewidth=1.5, zorder=0)

            # Mark important points in the reward and return graph
            if i in [1,3] and len(metric_labels) == 4:
                # show rew at convergence
                conv_timestep = metric.approach.steps_to_conv_mean
                train_dur = (metric.approach.train_duration_mio * 1e6)
                index = int(n_points * conv_timestep / train_dur)
                mean_conv = metric.mean_fltrd[index]

                subplot.scatter(conv_timestep, mean_conv, s=90, marker='o', color=mean_color, zorder=1)
                subplot.plot(np.linspace(0, conv_timestep, 3), np.ones(3)*mean_conv,
                             c=mean_color, linestyle='--', linewidth=1.5, zorder=0)

            # Plot the 75% vertical line in the reward plot
            if i == 1 and len(metric_labels) == 4:
                # subplot.plot(np.linspace(0, train_dur, 3), np.ones(3) * 0.75,
                #              c='gray', linestyle=':', linewidth=1.25, zorder=0)
                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)
                    xpos = rew75_timestep
                    if i_met == 2:
                        xpos -= 4e5
                    elif i_met == 3:
                        xpos += 4e5
                    subplot.text(xpos, 0.15, f'{metric.approach.steps_to_75rew_mean}',
                             ha='center', fontsize=tick_size+2, color=mean_color)
                    # vertical
                    subplot.plot(np.ones(3) * rew75_timestep,
                              np.linspace(0.205, mean75, 3), c=mean_color,
                              linestyle=':', linewidth=2, zorder=0)

            # Plot horizontal 75% line in the return graph
            if i == 3 and len(metric_labels) == 4:
                # subplot.plot(np.linspace(0, train_dur, 3), np.ones(3) * 0.75,
                #              c='gray', linestyle=':', linewidth=1.25, zorder=0)
                # show time after 75% of reward was reached
                # but only when it is reached at all
                rew75_timestep = 1e6 * metric.approach.steps_to_75rew_mean
                if rew75_timestep != train_dur:
                    index = int(n_points * rew75_timestep / train_dur)
                    mean75 = metric.mean_fltrd[index]
                    # subplot.scatter(rew75_timestep, 0.07, s=75, marker='v', color=mean_color)
                    subplot.scatter(rew75_timestep, mean75, s=120, marker='X', color=mean_color, zorder=10)

                    # horizontal
                    subplot.plot(np.linspace(0, rew75_timestep, 3),
                                 np.ones(3) * mean75, c=mean_color,
                                 linestyle=':', linewidth=2, zorder=0)

            if i == 2:
                subplot.set_xlim([0, 8e6])


                # # Plot vertical lines indicating convergence and 75% rew
                # # vertical
                # subplot.plot(np.ones(3)*conv_timestep,
                #              np.linspace(line_start[i], mean_conv,3), c=mean_color,
                #              linestyle='--', linewidth=1.4, zorder=0)
                # # horizontal
                # subplot.plot(np.linspace(0, max_dur, 3), np.ones(3)*mean_conv,
                #              c=mean_color, linestyle='--', linewidth=1.5, zorder=0)

    x_label = r'Training Timesteps [x$10^6$]'
    # plt.gcf().tight_layout(rect=[0.1, 0.5, 0.95, 1])
    fig.text(0.5, 0.04, x_label, ha='center', fontsize=font_size-1)
    plt.subplots_adjust(wspace=0.33, top=0.99, left=0.05, right=0.99, bottom=0.18)
    legend_texts = [approach_names_dict[ap] for ap in ap_names]
    # assert 'ours' in ''.join(legend_texts).lower()
    if plot_rews:
        # before [0,3,6,9]
        subplots[-1].legend(np.array(subplots[-1].get_lines())[[0,2,5] if len(aps) == 3 else [0,2,4,6]], legend_texts, fancybox=True, framealpha=0.6,
                            loc='upper right')
    else:
        subplots[2].legend(subplots[2].get_lines(), legend_texts, fancybox=True, framealpha=0.6,
                            loc='upper left', bbox_to_anchor=(0.04, 1))
    plt.show()



def compare_rewards():
    raise NotImplementedError('Use compare_main_plots() with changed metrics to plot the results!')
    plt = config_pyplot(fig_size=0.5)
    ap_names = [APD_NORM_ANGS, APD_NORM_DELTA, APT_BSLN]
    approach_names_dict[APD_NORM_ANGS] = 'Target Angles'
    approach_names_dict[APT_BSLN] = 'Joint Torques (Ours)'
    approach_names_dict[APD_NORM_DELTA] = 'Angle Deltas (Ours)'
    aps = [Approach(name) for name in ap_names]
    max_dur = 1e6 * max([ap.train_duration_mio for ap in aps])
    # metric_labels = [MET_SUM_SCORE, MET_STABLE_WALKS, MET_STEP_REW, MET_TRAIN_EPRET]
    metric_labels = [MET_REW_POS, MET_REW_VEL, MET_REW_COM]
    n_metrics = len(metric_labels)

    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=0, tick_size=-2, legend_fontsize=-3, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})

    # heights for the lines
    line_start = np.zeros(n_metrics)
    line_start[2] = 0.1
    line_stop = [30, 20, 0.81, 0.8]

    fig, subplots = plt.subplots(1, n_metrics)
    for i, metric_label in enumerate(metric_labels):
        subplot = subplots[i]
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])

            # plot vertical lines indicating convergence and 75% rew
            # but only in the reward plot
            if False and i == 2:
                conv_timestep = metric.approach.steps_to_conv_mean
                index = int(len(metric.mean_fltrd)*conv_timestep/(metric.approach.train_duration_mio*1e6))
                stop = metric.mean_fltrd[index]
                # vertical
                # subplot.plot(np.ones(3)*conv_timestep,
                #              np.linspace(line_start[i], stop,3), c=mean_color,
                #              linestyle='--', linewidth=1.4, zorder=0)
                # horizontal
                subplot.plot(np.linspace(0, max_dur, 3), np.ones(3)*stop,
                             c=mean_color, linestyle='--', linewidth=1.5, zorder=0)


    x_label = r'Training Timesteps [x$10^6$]'
    # plt.gcf().tight_layout(rect=[0.1, 0.5, 0.95, 1])
    fig.text(0.5, 0.04, x_label, ha='center', fontsize=font_size-1)
    plt.subplots_adjust(wspace=0.4, top=0.99, left=0.05, right=0.99, bottom=0.18)
    legend_texts = [approach_names_dict[ap] for ap in ap_names]
    assert 'ours' in ''.join(legend_texts).lower()
    subplots[-1].legend(legend_texts, fancybox=True, framealpha=0.6)
    plt.show()


def compare_baselines_rews():
    ap_names = [APD_BSLN, APT_BSLN]
    approach_names_dict[APT_BSLN] = 'Joint Torque'
    approach_names_dict[APD_BSLN] = 'Target Angles'
    aps = [Approach(name) for name in ap_names]
    metric_labels = [MET_REW_POS, MET_REW_VEL, MET_REW_COM]
    n_metrics = len(metric_labels)
    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=-2, tick_size=-3, legend_fontsize=-2, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})
    fig, subplots = plt.subplots(1, n_metrics)
    for i, metric_label in enumerate(metric_labels):
        subplot = subplots[i]
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])
    x_label = r'Training Timesteps [x$10^6$]'
    # plt.gcf().tight_layout(rect=[0.1, 0.5, 0.95, 1])
    fig.text(0.5, 0.04, x_label, ha='center', fontsize=font_size-1)
    # subplots[1].set_xlabel(x_label)
    plt.subplots_adjust(wspace=0.4, top=0.99, left=0.04, right=0.99, bottom=0.18)
    subplots[-1].legend([approach_names_dict[ap] for ap in ap_names])
    plt.show()


def compare_baselines_training_curves():
    # for the APPENDIX
    ap_names = [APD_BSLN, APT_BSLN]
    approach_names_dict[APT_BSLN] = 'Joint Torque'
    approach_names_dict[APD_BSLN] = 'Target Angles'
    aps = [Approach(name) for name in ap_names]
    metric_labels = [MET_TRAIN_EPRET, MET_TRAIN_STEPREW, MET_TRAIN_DIST, MET_TRAIN_EPLEN]
    n_metrics = len(metric_labels)
    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=-2, tick_size=-3, legend_fontsize=-2, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})
    fig, subplots = plt.subplots(1, n_metrics)
    for i, metric_label in enumerate(metric_labels):
        subplot = subplots[i]
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            if not isinstance(metric.mean, np.ndarray) or not len(metric.mean) > 1:
                subplot.scatter(metric.data, np.arange(len(metric.data)))
                continue
            n_steps = metric.train_duration_mio*1e6
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_ylabel(metric_names_dict[metric.label])
    x_label = r'Training Timesteps [x$10^6$]'
    # plt.gcf().tight_layout(rect=[0.1, 0.5, 0.95, 1])
    fig.text(0.5, 0.04, x_label, ha='center', fontsize=font_size-1)
    # subplots[1].set_xlabel(x_label)
    plt.subplots_adjust(wspace=0.4, top=0.99, left=0.04, right=0.99, bottom=0.18)
    subplots[-1].legend([approach_names_dict[ap] for ap in ap_names])
    plt.show()


def compare_baselines_violin():
    # for the APPENDIX
    ap_names = [APD_BSLN, APT_BSLN]
    approach_names_dict[APT_BSLN] = 'Joint Torque'
    approach_names_dict[APD_BSLN] = 'Target Angles'
    aps = [Approach(name) for name in ap_names]
    metric_label = MET_STEPS_TO_CONV
    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=2, tick_size=+3, legend_fontsize=-2, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})

    metrics = [metric for ap in aps for metric in ap.metrics
               if metric.label == metric_label]
    names = [approach_names_dict[ap] for ap in ap_names]
    means = []
    hist_data = []
    for metric in metrics:
        means.append(metric.mean)
        hist_data.append(metric.data)
    violin(names, means, hist_data, '',
           metric_names_dict[metric.label] + ' [x$10^6$]', text_size=font_size)
    tick_distance_mio = 2
    arange = np.arange(2, 8)
    plt.gca().set_yticks(arange * tick_distance_mio * 1e6)
    plt.gca().set_yticklabels([f'{x}' for x in arange * tick_distance_mio])

    # plt.subplots_adjust(wspace=0.4, top=0.99, left=0.04, right=0.99, bottom=0.18)
    plt.show()


def compare_violins():
    plt = config_pyplot(1)
    # for the APPENDIX
    ap_names = [APD_NORM_ANGS, APD_NORM_DELTA, APT_BSLN]
    ap_names = [APT_BSLN, APT_BSLN_HALF_BS, APT_DUP, APT_MRR_STEPS]
    ap_names = [APD_BSLN, APD_NORM_DELTA, APT_BSLN, APT_MRR_STEPS, APT_DUP]
    approach_names_dict[APT_BSLN_HALF_BS] = 'Baseline (1/2 BS)'
    approach_names_dict[APD_BSLN] = 'Baseline\nTarget Angles'
    approach_names_dict[APD_NORM_DELTA] = 'Angle\nDeltas'
    approach_names_dict[APT_BSLN] = 'Torque'
    approach_names_dict[APT_MRR_STEPS] = 'Mirror\nPolicy'
    approach_names_dict[APT_DUP] = 'Mirror\nExperiences'

    aps = [Approach(name) for name in ap_names]
    metric_label = MET_STEPS_TO_CONV
    font_size, tick_size, legend_size = \
        change_plot_properties(font_size=5, tick_size=7, line_width=+1)
    plt.rcParams.update({'figure.autolayout': False})

    metrics = [metric for ap in aps for metric in ap.metrics
               if metric.label == metric_label]
    names = [approach_names_dict[ap] for ap in ap_names]
    means = []
    hist_data = []
    for i, metric in enumerate(metrics):
        means.append(metric.mean)
        hist_data.append(metric.data)
    violin(names, means, hist_data, '',
           metric_names_dict[metric.label] + ' [x$10^6$]', text_size=font_size)
    # arange = np.arange(2, 7) * 2
    arange = np.arange(1, 7) * 2
    plt.gca().set_yticks(arange * 1e6)
    plt.gca().set_yticklabels([f'{x}' for x in arange])
    # plt.subplots_adjust(wspace=0.4, top=0.99, left=0.04, right=0.99, bottom=0.18)
    plt.show()


def plot_metrics_table():
    # first get all metrics of interest
    ap_names = [APD_NORM_ANGS, APD_NORM_DELTA, APT_BSLN]
    ap_names = [APT_BSLN, APT_BSLN_HALF_BS, APT_DUP, APT_MRR_STEPS]
    # approach_names_dict[APT_BSLN] = 'Joint Torque'
    # approach_names_dict[APD_BSLN] = 'Target Angles'
    aps = [Approach(name) for name in ap_names]
    for ap in aps:
        print('\nApproach:', ap.name)
        print(f'{ap.final_sum_score_mean} \pm {ap.final_sum_score_std} \n'
              f'{ap.steps_to_conv_mean} \pm {ap.steps_to_conv_std} \n'
              f'{ap.rews_at_conv_mean} \pm {ap.rews_at_conv_std} \n'
              f'{ap.steps_to_75rew_mean} \pm {ap.steps_to_75rew_std} \n'
              f'{ap.rews_at_end_mean} \pm {ap.rews_at_end_std} \n'
              )


if __name__ == '__main__':
    # download_approach_data(APT_BSLN, 'final3d_trq')
    # plot_metrics_table()
    # show_summary_score_advantages()
    compare_main_plots()
    # plot_return_only()
    # compare_main_torque_plots()
    # compare_rewards()
    # compare_violins()
    # compare_baselines_training_curves()
    # compare_baselines_rews()
    # compare_baselines_8plots()
    # compare_baselines_violin()
    # compare_action_spaces()
    # compare_all_metrics()
    # compare_pd_violin()
    # download_approaches()
    # download_approach_data(APD_NORM_ANGS, 'pd_approaches')
