import numpy as np
import scipy.io as spio
import seaborn as sns
from drloco.common import utils
from matplotlib import pyplot as plt
from drloco.common.utils import get_project_path

# plt = utils.config_pyplot(font_size=0, tick_size=0, legend_fontsize=0)
# plt.rcParams.update({'figure.autolayout': False})

# load matlab data, containing trajectories of 250 steps
dir_path = get_project_path()
file_path = 'assets/mocaps/loco3d/loco3d_guoping.mat'

data = spio.loadmat(dir_path+file_path, squeeze_me=True)

kin_labels = data['rowNameIK']
angles = data['angJoi']
ang_vels = data['angDJoi']
n_dofs = len(kin_labels)
SAMPLE_FREQ = 500 # fskin

labels = kin_labels

# plot figure in full screen mode (scaled down aspect ratio of my screen)
plt.rcParams['figure.figsize'] = (19.2, 10.8)
plt.rcParams.update({'figure.autolayout': True})


for i in range(n_dofs):
    try: subplt = plt.subplot(8,5,i+1)
    except: subplt = plt.subplot(8,5,i+1)
    line_blue = plt.plot(angles[i, 1000:2000])
    velplt = subplt.twinx()
    line_orange = velplt.plot(ang_vels[i, 1000:2000], 'darkorange')
    velplt.tick_params(axis='y', labelcolor='darkorange')
    plt.title(f'{i} - {labels[i]}')
    # remove x ticks and labels from first rows
    if i < n_dofs - 5:
        subplt.set_xticks([])
    else:
        subplt.set_xticks(range(0, 1001, 200))
        subplt.set_xlabel('Timesteps [1/500s]')

plt.rcParams.update({'figure.autolayout': True})


# collect different lines to place the legend in a separate subplot
lines = [line_blue[0], line_orange[0]]
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
