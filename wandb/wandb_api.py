import wandb
import numpy as np

MET_STEPS_TO_CONV = 'log_steps_to_convergence'

class Api:

    def __init__(self, project_name=None):
        self.api = wandb.Api()
        if project_name is not None:
            self.set_project(project_name)

    def set_project(self, project_name):
        self.project_name = project_name
        self.runs = self.api.runs("rustamg/%s" % project_name)

    def get_metrics(self, approach):
        # get relevant runs, finished only
        runs = [run for run in self.runs if (run.name == approach.run_name)]
        i = 1
        for run in runs:
            print(f'Fething run {i} of {len(runs)} runs')
            i += 1
            history = run.history(samples=int(1e5))
            for metric in approach.metrics:
                sum = run.summary
                sum_keys = list(sum.keys())
                if metric.label == MET_STEPS_TO_CONV:
                    if MET_STEPS_TO_CONV in sum_keys:
                        metric.append_run(sum[MET_STEPS_TO_CONV])
                    else:
                        print('WARNING! One run has not converged!')
                        metric.append_run(10e6)
                    continue
                metric.append_run(history[metric.label].dropna().tolist())


if __name__ == '__main__':
    # Project is specified by <entity/project-name>
    PROJECT_NAME = "pd_approaches"
    run_names = ['BSLN, init std = 1', 'BSLN - normed target angles', 'normed deltas']

    api = wandb.Api()
    runs = api.runs("rustamg/%s" % PROJECT_NAME)
    summary_list = []
    config_list = []
    name_list = []
    history_list = []
    for run in runs:
        if not run.label in run_names:
            continue
        elif run.state != 'finished':
            print(f'Incomplete run:\n{run.label}: {run.state}\n')
            continue
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # keys = ['_det_eval/1. AUC stable walks count', '_det_eval/2. stable walks count']
        history_list.append(run.history(samples=int(1e6)))

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        config_list.append(config)

        # run.name is the name of the run.
        # name_list.append(run.name)

    import pandas as pd
    history_df = pd.DataFrame.from_records(history_list)
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})
    all_df = pd.concat([name_df, config_df,summary_df], axis=1)

    SAVE_DF = False
    if SAVE_DF: all_df.to_csv("project.csv")