import numpy as np
from scripts.common import utils

plt = utils.import_pyplot()

xs = np.linspace(1,100,10000)
sins = np.sin(xs)
x_noisy = []
x_rms = []
s_rms = []
sins_noisy = []
x_exps = []
sins_exps = []
sins_clean_exps = []

plt.plot(sins)

for s in sins:
    sins_clean_exps.append(utils.exponential_running_smoothing('sin_clean', s, 0.25))
    s += np.random.normal(0,0.25)
    sins_noisy.append(s)
    s_rms.append(utils.running_mean('sin', s))
    sins_exps.append(utils.exponential_running_smoothing('sin_noisy', s, 0.25))
plt.plot(sins_noisy)
plt.plot(s_rms)
plt.plot(sins_exps)
plt.plot(sins_clean_exps)
plt.legend(['orig', 'orig (noisy)', 'run_mean', 'exp_smooth (of noisy data)', 'exp_smooth (no noise)'])
plt.show()

plt.figure()
for x in xs:
    x += np.random.normal(0,5)
    x_rm = utils.running_mean('x', x)
    x_exps.append(utils.exponential_running_smoothing('x', x, 0.25))
    x_noisy.append(x)
    x_rms.append(x_rm)
plt.plot(x_noisy)
plt.plot(x_rms)
plt.plot(x_exps)
plt.legend(['orig', 'run_mean', 'exp_smooth'])
plt.show()
exit(33)