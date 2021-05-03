from scripts.mocap.ref_trajecs import ReferenceTrajectories, labels as refs_labels, SAMPLE_FREQ
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

IS_3D = True

if IS_3D:
    from gym_mimic_envs.mujoco.mimic_walker3d import \
        qpos_indices, qvel_indices, ref_trajec_adapts
else:
    from gym_mimic_envs.mujoco.mimic_walker2d import \
        qpos_indices, qvel_indices, ref_trajec_adapts


def get_refs(refs=None):
    if refs is None:
        refs = ReferenceTrajectories(
            qpos_indices, qvel_indices, ref_trajec_adapts)
    return refs


def get_joint_mocap_stats(refs=None, plot=False, std_only=False, save_path=None):
    """
    Calculates the means and stds of reference joint kinematics of the MimicWalker2D.
    Statistics are computed for each leg separately for the CONSTANT SPEED TRAJECS!

    :return: means_left, means_right, stds_left, stds_right
    Returns the mean trajectories of a left and right step with corresponding stds.
    """
    if refs is None:
        refs = get_refs()

    refs.reset()
    data = refs.data # 30 x (19, n_step_points)

    # to be able to get mean and std of each trajectory,
    # we need to make all steps equally long (crop to min step length)
    step_lens = [step.shape[1] for step in data]
    min_len = np.min(step_lens)
    data = [step[:, :min_len] for step in data]
    data = np.array(data, dtype=np.float)

    # extract only relevant joints
    indices = qpos_indices + qvel_indices
    data = data[:, indices, :]

    # now, split data into left and right steps
    left_indices = refs.left_step_indices
    right_indices = np.array(left_indices) - 1
    data_left = data[left_indices, :, :]
    data_right = data[right_indices, :, :]

    means_left = np.mean(data_left, axis=0)
    stds_left = np.std(data_left, axis=0)
    means_right = np.mean(data_right, axis=0)
    stds_right = np.std(data_right, axis=0)

    means_all = np.concatenate([means_right, means_left], axis=1)
    stds_all = np.concatenate([stds_right, stds_left], axis=1)

    if save_path is not None:
        np.savez(save_path, means=means_all, std=stds_all,
                 means_left=means_left, stds_left=stds_left,
                 means_right=means_right, stds_right=stds_right)

    # plot figure in full screen mode (scaled down aspect ratio of my screen)
    plt.rcParams['figure.figsize'] = (19.2, 10.8)
    plt.rcParams['figure.autolayout'] = False

    PLOT_ANGLE_DERIVS = False # broken when putting both steps together
    if plot:
        n_rows = 6 if IS_3D else 5
        n_cols = 5 if IS_3D else 4
        for i_sbplt, i in enumerate(indices):
            try:
                subplt = plt.subplot(n_rows, n_cols, i_sbplt + 1, sharex=subplt)
            except:
                subplt = plt.subplot(n_rows, n_cols, i_sbplt + 1)
            curve = means_all[i_sbplt, :]
            std = 2*stds_all[i_sbplt, :]
            if i < 15 or i > 28:
                line_blue = plt.plot(curve)
                plt.fill_between(range(len(curve)), curve-std, curve+std, alpha=0.5)
            else:
                # plot vels in orange
                line_red = plt.plot(curve, 'red')
                plt.fill_between(range(len(curve)), curve-std, curve+std, color='red', alpha=0.25)
            plt.title(f'{i} - {refs_labels[i]}')

            # plot the derivatives to easier find corresponding velocities
            if PLOT_ANGLE_DERIVS and i < 15:
                velplt = subplt.twinx()
                line_orange = velplt.plot(np.gradient(curve, 1 / SAMPLE_FREQ), 'darkorange')
                velplt.tick_params(axis='y', labelcolor='darkorange')

        # collect different lines to place the legend in a separate subplot
        if PLOT_ANGLE_DERIVS:
            lines = [line_blue[0], line_orange[0], line_red[0]]
            labels = ['Joint Positions [rad]',
                      'Joint Position Derivatives [rad/s]',
                      'Joint Velocities (Dataset) [rad/s]']
        else:
            lines = [line_blue[0], line_red[0]]
            labels = ['Joint Positions [rad]',
                      'Joint Velocities (Dataset) [rad/s]']

        # # fix title overlapping when tight_layout is true
        plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.55, hspace=0.5)
        plt.suptitle('Reference Trajectory Statistics (mean $\pm$ 2std, two steps)')
        # plot the legend in a separate subplot
        with sns.axes_style("white", {"axes.edgecolor": 'white'}):
            legend_subplot = plt.subplot(n_rows, n_cols, i_sbplt + 2)
            legend_subplot.set_xticks([])
            legend_subplot.set_yticks([])
            legend_subplot.legend(lines, labels,
                                  bbox_to_anchor=(1.25, 1.15))

        plt.show()

    if std_only:
        return stds_all, stds_left, stds_right
    else:
        return means_all, stds_all, means_left, means_right, stds_left, stds_right


def get_mocap_stds(refs=None, save_stds=False):
    """ returns: the std for each dimension of the mocap data. """
    if refs is None:
        refs = get_refs()

    refs.reset()
    # (num_data_points x refs_dim)
    data_matrix = np.concatenate(refs.data.tolist(), axis=1).transpose()
    # convert to float array
    data = np.array(data_matrix, dtype=np.float32)
    # get stds as max allowed devitations
    stds = np.std(data, axis=0)
    if save_stds:
        np.save('/assets/ref_trajecs/distributions/3d_mocap_stds_const_speed_400hz.npy', stds)
    return stds

def determine_max_allowed_deviations(stds, save=False):
    max_stds = np.max(stds, axis=1)
    if save:
        np.save('/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/assets/'
                'ref_trajecs/distributions/3d_max_qpos_qvel_stds_const_speed_400hz', max_stds)


if __name__ == '__main__':
    path_mocap_stats = None  # add path here to save mocap statistics!
    # path_mocap_stats = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/assets/' \
    #                    'ref_trajecs/distributions/3d_distributions_constant_speed_400hz.npz'
    means_all, stds_all, means_left, means_right, stds_left, stds_right = \
                    get_joint_mocap_stats(plot=True, save_path=path_mocap_stats)
    determine_max_allowed_deviations(stds_all, save=False)
    mean_stds = np.mean(np.hstack([stds_left, stds_right]), axis=1)
    debug = False
