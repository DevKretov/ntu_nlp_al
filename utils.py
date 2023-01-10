from pathlib import Path
import json
import yaml

LOCAL_RUNS_FOLDER = 'runs'
LOCAL_RUNS_FOLDER_PATH = Path(LOCAL_RUNS_FOLDER)

TRASK = True
CONFIGS_FOLDER_NAME = 'configs'
APP_CONFIG_FILE_NAME = 'config.yaml'
CONFIGS_FOLDER_PATH = Path(__file__).resolve().parent / CONFIGS_FOLDER_NAME
APP_CONFIG_FILE_NAME = CONFIGS_FOLDER_PATH / APP_CONFIG_FILE_NAME

config = yaml.safe_load(open(str(APP_CONFIG_FILE_NAME)))


import matplotlib.pyplot as plt

class TrainingVisualisation:

    def __init__(self):
        # plots to compare the performance of strategies and the full dataset
        self.num_cols = 2
        self.num_rows = 2
        plt.rcParams.update({'font.size': 25})
        plt.rcParams['figure.facecolor'] = 'white'
        self.fig, self.axs = plt.subplots(self.num_rows, self.num_cols, figsize=(30,20), dpi=50)

        self.max_iterations_range = 0

        self.model_type = config['run']['finetuned_model_type']
        self.metrics_to_visualise = []
        if self.model_type == config['app']['model_classification_name']:
            self.metrics_to_visualise = config['visualisation']['classification_metrics']
        elif self.model_type == config['app']['model_tagging_name']:
            self.metrics_to_visualise = config['visualisation']['tagging_metrics']
        else:
            raise NotImplementedError(f'No implemented metrics list for model type: {self.model_type}')

        self.metrics_visualise_names_dict = config['visualisation']['metrics_names']

    def visualise(self, save_fig_path=None):
        axs = self.axs
        fig = self.fig

        for ax in axs.flat:
            ax.set(xlabel='iterations', ylabel='score')  # name x and y axis
            ax.set_xticks(list(range(1, self.max_iterations_range + 1)))
            handles, labels = ax.get_legend_handles_labels()  # get all different labels and make one legend for all
            fig.legend(handles, labels, loc='center')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)  # make nicer
        if save_fig_path is not None:
            if Path(save_fig_path).parent.is_dir():
                plt.savefig(save_fig_path)
            else:
                Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_fig_path)

        plt.show()  # only show when all strategies are added

    def add_full_training_metrics(self, full_training_metrics):
        self.fig.suptitle(
            'Comparison of different AL strategies and the performance of the full dataset',
            fontsize=35
        )

        axs = self.axs

        metric_i = 0
        for row_i in range(self.num_rows):
            for col_i in range(self.num_cols):
                metric_name = self.metrics_to_visualise[metric_i]
                metric_name_to_visualise = self.metrics_visualise_names_dict[metric_name]

                axs[row_i, col_i].axhline(
                    y=full_training_metrics[metric_name],
                    color='r',
                    linestyle='-',
                    label='full'
                )  # plot the score from the full dataset for comparison

                self.max_iterations_range = max(
                    self.max_iterations_range,
                    full_training_metrics[metric_name]
                )

                axs[row_i, col_i].set_title(metric_name_to_visualise)
                metric_i += 1


    def add_al_strategy_metrics(self, al_iteration_metrics, strategy):
        self.fig.suptitle('Comparison of different AL strategies', fontsize=35)
        axs = self.axs

        if strategy == 'random':
            strategy = 'baseline'

        iterations_range = list(range(1, len(al_iteration_metrics) + 1))

        metrics_scores_dict_to_visualise = {_metric_name: [] for _metric_name in self.metrics_to_visualise}
        for iteration in al_iteration_metrics:
            for _metric_name in self.metrics_to_visualise:
                metrics_scores_dict_to_visualise[_metric_name].append(iteration[_metric_name])

        metric_i = 0
        for row_i in range(self.num_rows):
            for col_i in range(self.num_cols):
                metric_name = self.metrics_to_visualise[metric_i]
                metric_data_to_visualise = metrics_scores_dict_to_visualise[metric_name]

                axs[row_i, col_i].plot(iterations_range, metric_data_to_visualise, label=strategy)

                metric_i+=1

        self.fig.tight_layout()
        self.max_iterations_range = max(self.max_iterations_range, len(iterations_range))