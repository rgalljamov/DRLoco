import numpy as np
import scipy.io as spio
import seaborn as sns
from scripts.common import utils
from scripts.common.config import abs_project_path

plt = utils.config_pyplot(font_size=12, tick_size=12, legend_fontsize=13)
# plt.rcParams.update({'figure.autolayout': False})

# load matlab data, containing trajectories of 250 steps
dir_path = abs_project_path
# file_path = 'assets/ref_trajecs/Trajecs_Ramp_Slow_400Hz_EulerTrunkAdded.mat'
file_path = 'assets/ref_trajecs/Trajecs_Constant_Speed_400Hz.mat'
# file_path = 'assets/ref_trajecs/original/Traj_Ramp_Slow_final.mat'
file_path = 'assets/ref_trajecs/original/Traj_Ramp_Slow_1000Hz.mat'

SAMPLE_FREQ = 1000
assert str(SAMPLE_FREQ) in file_path, 'Have you set the right sample frequency!?'



data = spio.loadmat(dir_path+file_path, squeeze_me=True)

# 250 steps, shape (250,1), where 1 is an array with kinematic data
data = data['Data']
# flatten the array to have dim (steps,)
data = data.flatten()
print('Number of steps recorded: ', np.size(data))

# first step (37 dims, 281 timesteps)
step = data[3]
dofs, timesteps = step.shape

def get_com_pos_all_steps():
    com_pos = []
    for step in range(len(data)):
        com_pos.extend(data[step][0])
    return com_pos

def get_com_vel_all_steps():
    com_vels = []
    mean_vels = []
    for step in range(len(data)):
        vels = data[step][15]
        mean_vel = np.mean(vels)
        print(f'{step} \t {mean_vel}')
        mean_vels.extend(np.ones_like(vels)*mean_vel)
        com_vels.extend(vels)
    return com_vels, mean_vels

PLOT_COM_STATS = False
if PLOT_COM_STATS:
    com_pos_all = get_com_pos_all_steps()
    plt.plot(com_pos_all)
    plt.show()
    #
    com_vels, mean_vels = get_com_vel_all_steps()
    plt.plot(com_vels)
    plt.plot(mean_vels)
    plt.show()
    # exit(33)

test_refs = False
if test_refs:
    from scripts.mocap.ref_trajecs import ReferenceTrajectories as RT

    rt = RT(range(15), range(15,29))
    rt._step = rt.data[0]
    compos, comvel = rt.get_com_kinematics_full()
    step = rt._step
    dofs, timesteps = step.shape
    # step[0:3,:] -= compos


# label every trajectory with the corresponding name
labels = ['COM Pos (X)', 'COM Pos (Y)', 'COM Pos (Z)',
          'Trunk Rot (quat,w)', 'Trunk Rot (quat,x)', 'Trunk Rot (quat,y)', 'Trunk Rot (quat,z)',
          'Ang Hip Frontal R', 'Ang Hip Sagittal R',
          'Ang Knee R', 'Ang Ankle R',
          'Ang Hip Frontal L', 'Ang Hip Sagittal L',
          'Ang Knee L', 'Ang Ankle L',

          'COM Vel (X)', 'COM Vel (Y)', 'COM Vel (Z)',
          'Trunk Ang Vel (X)', 'Trunk Ang Vel (Y)', 'Trunk Ang Vel (Z)',
          'Vel Hip Frontal R', 'Vel Hip Sagittal R',
          'Vel Knee R', 'Vel Ankle R',
          'Vel Hip Frontal L', 'Vel Hip Sagittal L',
          'Vel Knee L', 'Vel Ankle L',

          'Foot Pos L (X)', 'Foot Pos L (Y)', 'Foot Pos L (Z)',
          'Foot Pos R (X)', 'Foot Pos R (Y)', 'Foot Pos R (Z)',

          'GRF R [N]', 'GRF L [N]',
          'Trunk Rot (euler,x)', 'Trunk Rot (euler,y)', 'Trunk Rot (euler,z)',
          ]

if file_path == 'assets/ref_trajecs/Trajecs_Constant_Speed_400Hz.mat':
    # have no GRFs included
    labels.remove('GRF R [N]')
    labels.remove('GRF L [N]')

# plot figure in full screen mode (scaled down aspect ratio of my screen)
plt.rcParams['figure.figsize'] = (19.2, 10.8)

for i in range(dofs):
    try: subplt = plt.subplot(8,5,i+1, sharex=subplt)
    except: subplt = plt.subplot(8,5,i+1)
    curve = step[i, :]
    if i < 15 or i > 28:
        line_blue = plt.plot(curve)
    else:
        # plot vels in orange
        line_red = plt.plot(curve, 'red')
    plt.title(f'{i} - {labels[i]}')

    # plot the derivatives to easier find corresponding velocities
    if i < 15:
        velplt = subplt.twinx()
        line_orange = velplt.plot(np.gradient(curve, 1 / SAMPLE_FREQ), 'darkorange')
        velplt.tick_params(axis='y', labelcolor='darkorange')

    # remove x labels from first rows
    # if i < 32:
    #     plt.xticks([])

# collect different lines to place the legend in a separate subplot
lines = [line_blue[0], line_orange[0], line_red[0]]
# plot the legend in a separate subplot
with sns.axes_style("white", {"axes.edgecolor": 'white'}):
    legend_subplot = plt.subplot(8, 5, i + 2)
    legend_subplot.set_xticks([])
    legend_subplot.set_yticks([])
    legend_subplot.legend(lines, ['Joint Positions [rad]',
                                  'Joint Position Derivatives [rad/s]',
                                  'Joint Velocities (Dataset) [rad/s]'],
                          bbox_to_anchor=(1.15, 1.05) )

# # fix title overlapping when tight_layout is true
# plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])
# plt.subplots_adjust(wspace=0.55, hspace=0.5)
# plt.suptitle('Trajectories')

plt.show()

"""
Insights:
- the first 15 dimensions are joint positions
- the next 14 the corresponding velocities
- reduced dim of velocities due to usage of quaternions for freejoints
-- rotation in quaternions is 4D, the corresponding angular velocities 3D
"""