from scripts.plots.wandb_api import Api
from scripts.common import utils
import numpy as np
import os

class Metric:
    def __init__(self, label, approach, train_duration_mio=None):
        self.label = label
        self.approach = approach
        self.approach_name = approach.name
        self.train_duration_mio = train_duration_mio
        # runs x recorded points for metric
        self.data = []

    def append_run(self, run):
        self.data.append(run)

    def convert_data_to_np(self):
        # to account for scalar values like steps_to_conv
        if not (isinstance(self.data[0], list) or isinstance(self.data[0], np.ndarray)):
            self.data = np.array(self.data)
            return
        # cut all lists to the same minimum length
        # but avoid runs that failed too quickly
        while True:
            lens = [len(values) for values in self.data]
            min_len = np.min(lens)
            max_len = np.max(lens)
            # when the minimum run length is too short, delete this run
            if (max_len - min_len) > 0.05 * max_len:
                # delete the run with the too short length
                index = lens.index(min_len)
                failed_run_data = self.data.pop(index)
                assert min_len == len(failed_run_data)
                print(f'Removed a run with min len of {min_len} where max is {max_len}')
            else: break

        data = [values[-min_len:] for values in self.data]
        self.data = np.array(data)

    def set_np_data(self, data):
        assert isinstance(data, np.ndarray)
        self.data = data

    def calculate_statistics(self):
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        if isinstance(self.mean, np.ndarray) and len(self.mean) > 10:
            self.mean_fltrd = utils.smooth_exponential(self.mean, 0.005)
            self.std_fltrd = utils.smooth_exponential(self.std, 0.005)
        else: self.mean_fltrd = self.mean


class Approach:
    def __init__(self, approach_name, project_name=None, run_name=None, metrics_names=[]):
        self.name = approach_name
        self.project_name = project_name
        self.train_duration_mio = 16 if 'pd' in approach_name else 8
        self.run_name = run_name
        self.path = utils.get_absolute_project_path() + f'graphs/{self.name}/'
        self.metrics_names = metrics_names
        data_on_disc = self._get_metrics_data()
        self._calculate_statistics()
        if data_on_disc:
            self.get_table_metrics()

    def _get_metrics_data(self):
        # first try to load from disc
        metrics_path = self.path + 'metrics.npz'
        data_is_on_disc = os.path.exists(metrics_path)
        if data_is_on_disc:
            from scripts.plots.compare import MET_SUM_SCORE, MET_STEPS_TO_CONV
            from scripts.common.callback import EVAL_INTERVAL_RARE
            self.metrics = []
            npz = np.load(metrics_path)
            for metric_label in npz.keys():
                self.metrics_names.append(metric_label)
                metric = Metric(metric_label, self, self.train_duration_mio)
                metric_data = npz[metric_label]
                # normalize summary score
                if metric_label == MET_SUM_SCORE:
                    max_score = self.train_duration_mio*1e6/EVAL_INTERVAL_RARE
                    metric_data /= 0.5*max_score
                    # normalize training duration to range [0,1]
                    metric_data *= 16/self.train_duration_mio
                    metric_data *= 100 # show in percent
                elif metric_label == MET_STEPS_TO_CONV:
                    self.steps_to_conv = metric_data
                    self.steps_to_conv_mean = np.mean(metric_data)
                    self.steps_to_conv_std = np.std(metric_data)
                metric.set_np_data(metric_data)
                self.metrics.append(metric)
        # fetch from wandb if not on disc
        else:
            self._api = Api(self.project_name)
            self.metrics = [Metric(name, self, self.train_duration_mio) for name in self.metrics_names]
            self._api.get_metrics(self)
            self._metrics_to_np()
        return data_is_on_disc

    def _calculate_statistics(self):
        for metric in self.metrics:
            metric.calculate_statistics()

    def _metrics_to_np(self):
        # convert each metric individually
        for metric in self.metrics:
            metric.convert_data_to_np()

    def save(self):
        # prepare and create path if necessary
        path = utils.get_absolute_project_path() + 'graphs/'
        path += self.name + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        assert isinstance(self.metrics[0].data, np.ndarray)
        metrics = [metric.data for metric in self.metrics]
        keys = [metric.label for metric in self.metrics]
        np.savez(path+'metrics', **{key:metric for key,metric in zip(keys, metrics)})
        print('Successfully saved approach:', self.name)


    def get_table_metrics(self):
        from scripts.plots.compare import MET_SUM_SCORE, MET_STEP_REW
        for metric in self.metrics:
            if metric.label == MET_SUM_SCORE:
                self.final_sum_scores = metric.data[:, -1]
                self.final_sum_score_mean = np.mean(self.final_sum_scores)
                self.final_sum_score_std = np.std(self.final_sum_scores)
            elif metric.label == MET_STEP_REW:
                self.final_rews = metric.data[:, -1]
                # Rew at convergence
                # determine convergence timepoint as training time percentage
                # conv_steps_frac = self.steps_to_conv / (self.train_duration_mio*1e6)
                # conv_indices = np.array([len(rews) for rews in metric.data]) * conv_steps_frac
                # conv_indices = conv_indices.astype(int)
                # self.rews_at_conv = metric.data[range(len(conv_indices)),conv_indices]
                # query only mean and std curves as we either way need to show mean and std
                conv_steps_frac = np.mean(self.steps_to_conv) / (self.train_duration_mio*1e6)
                conv_index = int(metric.data.shape[1] * conv_steps_frac)
                self.rews_at_conv_mean = metric.mean_fltrd[conv_index]
                self.rews_at_conv_std = metric.std_fltrd[conv_index]

                # Rews at training end
                self.rews_at_end_mean = metric.mean_fltrd[-1]
                self.rews_at_end_std = metric.std_fltrd[-1]

                # Steps to 75% human-likeness
                # get indices of 75% rew and map these to training time
                rew75_indices = np.argmax(metric.data >= 0.75, axis=1)
                n_points = len(metric.mean)
                rew75_indices[rew75_indices==0] = n_points
                self.steps_to_75rew = rew75_indices / n_points * self.train_duration_mio
                self.steps_to_75rew_mean = np.mean(self.steps_to_75rew)
                self.steps_to_75rew_std = np.std(self.steps_to_75rew)

        # print('\nApproach: ', self.name)
        # print('Converged after percent: ', conv_steps_frac)
        # print('indices: ', conv_index)
        # print('Rews MEAN at CONV: ', self.rews_at_conv_mean)
        # print('Rews STD at CONV: ', self.rews_at_conv_std)
        # print('75 Rews reached after: ', self.steps_to_75rew)

        # round all metrics
        self.final_sum_score_mean = np.round(self.final_sum_score_mean, 1)
        self.final_sum_score_std = np.round(self.final_sum_score_std, 1)
        self.steps_to_conv_mean = int(self.steps_to_conv_mean)
        self.steps_to_conv_std = int(self.steps_to_conv_std)
        self.steps_to_75rew_mean = np.round(self.steps_to_75rew_mean, 1)
        self.steps_to_75rew_std = np.round(self.steps_to_75rew_std,1)
        self.rews_at_conv_mean = np.round(self.rews_at_conv_mean, 2)
        self.rews_at_conv_std = np.round(self.rews_at_conv_std, 2)
        self.rews_at_end_mean = np.round(self.rews_at_end_mean, 2)
        self.rews_at_end_std = np.round(self.rews_at_end_std, 2)
