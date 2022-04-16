import time
import tqdm #import tqdm
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

from pathlib import Path
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix

from dataset import Dataset

from strategies.entropy_strategy import EntropySampling
from strategies.badge_strategy import BadgeSampling
from strategies.kmeans_strategy import KMeansSampling
from strategies.least_confidence_strategy import LeastConfidence
from strategies.random_strategy import RandomStrategy
from strategies.tagging_least_confidence_strategy import TaggingLeastConfidence

import yaml

CONFIGS_FOLDER_NAME = 'configs'
APP_CONFIG_FILE_NAME = 'config.yaml'
CONFIGS_FOLDER_PATH = Path(__file__).resolve().parent / CONFIGS_FOLDER_NAME
APP_CONFIG_FILE_NAME = CONFIGS_FOLDER_PATH / APP_CONFIG_FILE_NAME
config = yaml.safe_load(open(str(APP_CONFIG_FILE_NAME)))


class ALTrainer:

    METRICS_TO_USE_AVEGARE_IN = config['al']['metrics_to_use_average_in']
    DEFAULT_AVERAGE_MODE = config['al']['default_average_mode']

    def __init__(
            self,
            wandb_on=False,
            imbalanced_training=False,
            model_type=config['app']['model_classification_name'],
            wandb_run=None,
            wandb_table=None,
            wandb_save_datasets_artifacts=False
    ):

        self.wandb_on = wandb_on
        self.model_type = model_type
        self.imbalanced_training = imbalanced_training
        self.rng = np.random.RandomState(2022)

        self.training_dict_keys = config['model']['training_dict_keys']

        self.lr_scheduler = None
        self.metrics = []

        self.wandb_run = wandb_run
        self.wandb_table = wandb_table
        self.wandb_save_datasets_artifacts = wandb_run and wandb_save_datasets_artifacts

        self.wandb_log_data_dict = dict()

    def set_model(self, model):
        self.model = model

    def set_strategy(self, strategy):
        self.strategy = strategy

    def set_dataset(self, dataset:Dataset):
        self.dataset = dataset

    # TODO: decide if needed considering transformers...
    def set_criterion(self, criterion):
        self.criterion = criterion
        pass

    def set_optimizer(self, optimizer:torch.optim.Optimizer):
        self.optimizer = optimizer

    # TODO: solve this issue and return LR scheduler back to training pipeline
    def set_lr_scheduler(self, scheduler, warmup_steps = 0):
        # if isinstance(scheduler, str):
        #     if scheduler.lower().strip() == 'linear':
        #         lr_scheduler = get_scheduler(
        #             name="linear",
        #             optimizer=self.optimizer,
        #             num_warmup_steps=warmup_steps,
        #             num_training_steps=num_training_steps
        #         )
        self.lr_scheduler = None

    def send_model_to_device(self):
        self.model.model = self.model.model.to(self.device)
        logging.debug(f'START: Is model on CUDA - {self.model.model.device}')

    def set_device(self, device):
        self.device = device

    def get_training_steps_num(self):
        return len(self.dataset.train_dataloader)

    def add_evaluation_metric(self, metric_obj):
        self.metrics.append(metric_obj)

    def full_train(
            self,
            train_epochs = 10,
            train_batch_size = 32,
            val_batch_size = 64,
            test_batch_size = 64,
            debug = False
    ):

        logging.info(f'Full training initialized!')

        self.train_batch_size = train_batch_size
        self.dataset.prepare_dataloaders(
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            imbalanced_training=self.imbalanced_training,
            al=False
        )

        logging.info(f'Training is run on {len(self.dataset.train_dataloader)} batches!')
        logging.info(f'Evaluation is run on {len(self.dataset.val_dataloader)} batches!')
        logging.info(f'Testing is run on {len(self.dataset.test_dataloader)} batches!')

        logging.info(f'\n=========================\n')

        steps_per_epoch = -1
        evaluation_steps = -1
        if debug:
            steps_per_epoch = 5
            evaluation_steps = 5

        self.train_model(
            epochs=train_epochs,
            steps_per_epoch=steps_per_epoch,
            evaluation_steps_num=evaluation_steps
        )

        logging.info(f'\nFull training finished')

    def determine_strategy(self, strategy_name):
        strategy = strategy_name
        if strategy.lower().strip() == 'random':
            strategy = RandomStrategy(
                self.model,
                self.dataset.unlabelled_dataloader,
                len(self.dataset.al_train_dataset['unlabelled']),
                self.device,
                model_type=self.model_type
            )
        elif strategy.lower().strip() == 'least_confidence':
            _class = LeastConfidence
            if self.model_type == 'tagging':
                _class = TaggingLeastConfidence

            strategy = _class(
                self.model,
                self.dataset.unlabelled_dataloader,
                len(self.dataset.al_train_dataset['unlabelled']),
                self.device,
                model_type=self.model_type
            )
        elif strategy.lower().strip() == 'least_confidence_thresh':
            strategy = LeastConfidence(
                self.model,
                self.dataset.unlabelled_dataloader,
                len(self.dataset.al_train_dataset['unlabelled']),
                self.device,
                model_type=self.model_type,
                threshold=0.6
            )
        elif strategy.lower().strip() == 'badge':

            strategy = BadgeSampling(
                self.model,
                self.dataset.unlabelled_dataloader,
                len(self.dataset.al_train_dataset['unlabelled']),
                self.device,
                model_type=self.model_type,
                num_labels=self.model.num_labels,
                embedding_dim=self.model.model.config.hidden_size,
                batch_size=self.val_batch_size
            )

        elif strategy.lower().strip() == 'entropy':
            strategy = EntropySampling(
                self.model,
                self.dataset.unlabelled_dataloader,
                len(self.dataset.al_train_dataset['unlabelled']),
                self.device,
                model_type=self.model_type
            )
        elif strategy.lower().strip() == 'kmeans':
            strategy = KMeansSampling(
                self.model,
                self.dataset.unlabelled_dataloader,
                len(self.dataset.al_train_dataset['unlabelled']),
                self.device,
                num_labels=self.model.num_labels,
                model_type=self.model_type,
                embedding_dim=self.model.model.config.hidden_size,
                batch_size=self.val_batch_size
            )

        else:
            raise NotImplementedError(f'There is no such strategy {strategy_name}')

        return strategy


    # TODO: if init_dataset_size is integer, take init_dataset_size samples to initial al dataset
    # TODO: if init_dataset_size is float [0.0, 1.0], take ratio
    # TODO: if add_dataset_size is integer, take add_dataset_size from unlabelled and "label" these data
    # TODO: if add_dataset_size is float [0.0, 1.0], take ratio from unlabelled dataset
    # TODO: enable train_epochs be a list (equal to al_iterations) to train with different epochs number
    # TODO: if strategy is string, just re-init the same strategy every new iteration
    # TODO" if strategy is list, take every new iteration new strategy
    def al_train(
            self,
            al_iterations = 5,
            init_dataset_size = 1000,
            add_dataset_size = 1000,
            train_epochs = 10,
            strategy = 'random',
            train_batch_size = 32,
            val_batch_size = 64,
            test_batch_size = 64,
            debug = False
    ):

        logging.info(f'Training initialized!')

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.dataset.prepare_al_datasets(init_dataset_size)
        self.dataset.prepare_dataloaders(
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            imbalanced_training=self.imbalanced_training,
            al=True
        )

        al_dataset_add_artifact = None
        train_dataset_artifact = None

        if self.wandb_save_datasets_artifacts:
            al_dataset_add_artifact = wandb.Artifact(
                config['reporting']['al_dataset_artifact_name'],
                type='dataset'
            )
            train_dataset_artifact = wandb.Artifact(
                config['reporting']['init_dataset_artifact_name'],
                type='dataset'
            )

        logging.info(f'\n===========================================================\n')

        if isinstance(strategy, str):
            strategy = self.determine_strategy(strategy)

        steps_per_epoch = -1
        evaluation_steps = -1
        if debug:
            steps_per_epoch = 5
            evaluation_steps = 5
            al_iterations = 5

        for al_iteration in range(al_iterations):
            self.wandb_log_data_dict = dict()

            logging.info(f'\n===========================================================')
            logging.info(f'\nAL iteration {(al_iteration + 1):3}/{al_iterations}')
            logging.info(f'Training is run on {len(self.dataset.train_dataloader)} batches!')
            logging.info(f'Evaluation is run on {len(self.dataset.val_dataloader)} batches!')
            logging.info(f'Testing is run on {len(self.dataset.test_dataloader)} batches!')

            self.train_model(
                epochs=train_epochs,
                steps_per_epoch=steps_per_epoch,
                evaluation_steps_num=evaluation_steps,
                al_iteration=al_iteration
            )

            logging.debug(f'Model training finished! Running AL strategy...')

            strategy.update_dataloader(self.dataset.unlabelled_dataloader)
            strategy.update_dataset_len(len(self.dataset.al_train_dataset['unlabelled']))

            indices = strategy.query(
                add_dataset_size
            )

            selected_dataset = self.dataset.al_train_dataset['unlabelled'].select(indices)

            # Save additional data entries from AL query to local storage and to W&B
            al_selected_dataset_save_path = str(strategy.strategy_log_folder_file / f'al_selected_dataset_{al_iteration}.csv')
            al_selected_dataset_df = pd.DataFrame.from_dict(
                selected_dataset.to_dict(32)
            )[[self.dataset.LABELS_STR_COLUMN_NAME, self.dataset.input_text_column_name]]#
            al_selected_dataset_df.to_csv(al_selected_dataset_save_path, index=False)

            # Save current training dataset to local storage and to W&B
            training_dataset_save_path = str(strategy.strategy_log_folder_file / f'train_dataset_{al_iteration}.csv')
            train_dataset_df = pd.DataFrame.from_dict(
                self.dataset.al_train_dataset['train'].to_dict(32)
            )[[self.dataset.LABELS_STR_COLUMN_NAME, self.dataset.input_text_column_name]]  #
            train_dataset_df.to_csv(training_dataset_save_path, index=False)

            if self.wandb_on:
                al_dataset_add_artifact.add_file(al_selected_dataset_save_path)
                self.wandb_run.log_artifact(al_dataset_add_artifact)

                train_dataset_artifact.add_file(training_dataset_save_path)
                self.wandb_run.log_artifact(train_dataset_artifact)

                al_dataset_add_artifact = wandb.Artifact(
                    config['reporting']['al_dataset_artifact_name'],
                    type='dataset'
                )
                train_dataset_artifact = wandb.Artifact(
                    config['reporting']['init_dataset_artifact_name'],
                    type='dataset'
                )

            logging.debug(f'Returned {len(indices)} indices from strategy')

            al_train_dataset_size_before = len(self.dataset.al_train_dataset['train'])
            al_unlabelled_dataset_size_before = len(self.dataset.al_train_dataset['unlabelled'])

            logging.debug(f'Before updating AL datasets: train size = {al_train_dataset_size_before}, unlabelled size = {al_unlabelled_dataset_size_before}, sum: {al_train_dataset_size_before + al_unlabelled_dataset_size_before} ')

            self.dataset.update_al_datasets_with_new_batch(
                indices_to_add=indices
            )

            al_train_dataset_size_after = len(self.dataset.al_train_dataset['train'])
            al_unlabelled_dataset_size_after = len(self.dataset.al_train_dataset['unlabelled'])

            logging.debug(f'\nUpdated AL datasets: train size = {al_train_dataset_size_after}, unlabelled size = {al_unlabelled_dataset_size_after}, sum: {al_train_dataset_size_after + al_unlabelled_dataset_size_after} ')

            assert (
                (al_train_dataset_size_before + al_unlabelled_dataset_size_before)
                ==
                (al_train_dataset_size_after + al_unlabelled_dataset_size_after)
            )

            self.dataset.prepare_dataloaders(
                train_batch_size=train_batch_size,
                val_batch_size=val_batch_size,
                test_batch_size=test_batch_size,
                imbalanced_training=self.imbalanced_training,
                al=True
            )

    def train_model(
            self,
            epochs,
            steps_per_epoch = -1,
            evaluation_steps_num = -1,
            al_iteration = -1
    ):
        '''

        Run model training with an evaluation afterwards.
        This method initialiazes model, sends model to GPU if exists,
        then sets up optimizer and runs training during specified number of epochs.

        In the end there is a testing evaluation with Weights&Biases logging.

        :param epochs: Number of epochs to train the model on training dataset
        :param steps_per_epoch: Number of steps per each epoch. -1 means that the training is run on full provided dataset
        :param evaluation_steps_num: Number of steps for evaluation. -1 means that the evaluation is run on full provided dataset
        :param al_iteration: The index of AL iteration in order to log some arbitrary run data like evaluation tables and metric for each run.
        :return:
        '''
        self.model.reinit_model()
        self.send_model_to_device()
        model = self.model.model

        if self.device != 'cpu':
            torch.cuda.empty_cache()

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        self.set_optimizer(optimizer)
        logging.info(f'Model initialized.')

        if steps_per_epoch == -1:
            steps_per_epoch = len(self.dataset.train_dataloader)

        for epoch_i in range(epochs):
            model.train()
            logging.info(f'\n\nEpoch {(epoch_i + 1):3}')

            losses_list =  []
            start_time = time.time()

            pbar = tqdm.trange(
                len(self.dataset.train_dataloader),
                desc="Iteration",
                smoothing=0.05,
                disable=False,
                position=0,
                leave=True,
                file=sys.stdout
            )

            step_i = 0
            for next_batch in self.dataset.train_dataloader:
                model.zero_grad()

                if self.model_type == 'tagging':
                    next_batch, next_batch_metadata = next_batch

                next_batch = {your_key: next_batch[your_key] for your_key in self.training_dict_keys}
                next_batch = {k: v.to(self.device) for k, v in next_batch.items()}

                outputs = model(**next_batch)
                loss = outputs.loss
                loss.backward()

                losses_list.append(loss.item())

                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

                pbar.set_description(f'Training mean loss: {np.mean(losses_list)}')
                pbar.update(1)

                step_i += 1
                if step_i == steps_per_epoch:
                    break
            logging.info(f'Epoch finished. Evaluation:')

            if evaluation_steps_num == -1:
                evaluation_steps_num = len(self.dataset.val_dataloader)

        if self.wandb_on:
            if al_iteration == -1:
                # if we are in the full train mode
                wandb.config.update(
                    {
                        config['reporting']['weights_and_biases_full_train_mean_loss_log_name']: np.mean(losses_list)
                    }
                )
            else:
                self.wandb_log_data_dict.update(
                    {
                        config['reporting']['weights_and_biases_train_mean_loss_log_name']: np.mean(losses_list)
                    }
                )

        self.evaluate(
            num_batches_to_eval=evaluation_steps_num,
            dataloader=self.dataset.val_dataloader,
            print_metrics=False,
            mode=config['app']['eval_mode'],
            al_iteration=al_iteration,
            epoch=epoch_i,
        )

        logging.info(f'Test set evaluation:')
        self.evaluate(
            num_batches_to_eval=-1,
            dataloader=self.dataset.test_dataloader,
            print_metrics=True,
            mode=config['app']['test_mode'],
            al_iteration=al_iteration
        )

        if self.wandb_on:
            wandb.log(self.wandb_log_data_dict, step_i=al_iteration, commit=True)



    def evaluate(
            self,
            num_batches_to_eval=-1,
            dataloader = None,
            print_metrics = False,
            mode = config['app']['eval_mode'],
            al_iteration = -1,
            epoch = 0
    ):

        model = self.model.model
        logging.debug(f'Is model on CUDA - {model.device}')

        eval_result = dict()
        #eval_result[al_iteration] = al_iteration


        if dataloader is None:
            dataloader = self.dataset.val_dataloader

        logging.info(f'Running {mode} mode...')
        if num_batches_to_eval == -1:
            num_batches_to_eval = len(dataloader)

        model.eval()
        pbar = tqdm.trange(
            num_batches_to_eval,
            #desc="Iteration",
            smoothing=0.05,
            disable=False,
            position=0,
            leave=True,
            file=sys.stdout
        )

        eval_loss = []

        batch_i = 0

        labels_all = None
        predictions_all = None

        wandb_table_predictions_data = []
        evaluation_table_data = []

        for next_batch_full in dataloader:
            if self.model_type == config['app']['model_tagging_name']:
                next_batch_full, next_batch_metadata = next_batch_full

            next_batch = {your_key: next_batch_full[your_key] for your_key in self.training_dict_keys}
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}
            with torch.no_grad():
                outputs = model(**next_batch)
            loss = outputs.loss

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            probs = F.softmax(logits, dim=-1)
            probs = probs.data.cpu().numpy()

            labels_seqeval = next_batch[config['dataset']['encoded_labels_column_name']].data.cpu().numpy()#.tolist()
            predictions_seqeval = predictions.data.cpu().numpy()#.tolist()

            labels_seqeval_new = []
            predictions_seqeval_new = []

            if self.model_type == config['app']['model_tagging_name']:
                for seq_i in range(len(labels_seqeval)):
                    _seq_labels = labels_seqeval[seq_i]
                    _seq_predictions = predictions_seqeval[seq_i]
                    mask = _seq_labels != -100

                    _probs = probs[seq_i, mask]#.tolist()
                    _seq_labels = _seq_labels[mask].tolist()
                    _seq_predictions = _seq_predictions[mask].tolist()

                    _seq_labels = [self.dataset.int_2_labels[_tag] for _tag in _seq_labels]
                    _seq_predictions = [self.dataset.int_2_labels[_tag] for _tag in _seq_predictions]
                    _seq_tokens = next_batch_metadata[seq_i][self.dataset.input_text_column_name]
                    _seq_tokens = self.dataset.tokenizer.tokenize(_seq_tokens, is_split_into_words=True)
                    _seq_id = np.full(len(_seq_tokens), seq_i)

                    labels_seqeval_new.append(_seq_labels)
                    predictions_seqeval_new.append(_seq_predictions)

                    wandb_table_data_batch = np.hstack(
                        (
                            _seq_id.reshape(-1, 1),
                            np.array(_seq_tokens).reshape(-1, 1),
                            np.array(_seq_labels).reshape(-1, 1),
                            np.array(_seq_predictions).reshape(-1, 1),
                            _probs
                        )
                    )

                    if self.wandb_table is not None:
                        for data_row in wandb_table_data_batch.tolist():
                            evaluation_table_data.append(data_row)
                            self.wandb_table.add_data(*data_row)



            elif self.model_type == config['app']['model_classification_name']:

                labels_str = [self.dataset.int_2_labels[_label] for _label in labels_seqeval]
                predictions_str = [self.dataset.int_2_labels[_label] for _label in predictions_seqeval]
                seq_i = np.arange(len(labels_str) * batch_i, len(labels_str) * (batch_i + 1))

                wandb_table_data_batch = np.hstack(
                    (
                        seq_i.reshape(-1, 1),
                        np.array(next_batch_full[self.dataset.input_text_column_name]).reshape(-1, 1),
                        np.array(labels_str).reshape(-1, 1),
                        np.array(predictions_str).reshape(-1, 1),
                        probs
                    )
                )

                if self.wandb_table is not None:
                    for data_row in wandb_table_data_batch.tolist():
                        evaluation_table_data.append(data_row)
                        self.wandb_table.add_data(*data_row)

            #if self.model_type != 'tagging':
            if labels_all is None:
                labels_all = next_batch[config['dataset']['encoded_labels_column_name']].data.cpu().numpy().reshape(-1)
                predictions_all = predictions.data.cpu().numpy().reshape(-1)
            else:

                labels_all = np.hstack((labels_all, next_batch[config['dataset']['encoded_labels_column_name']].data.cpu().numpy().reshape(-1)))
                predictions_all = np.hstack((predictions_all, predictions.data.cpu().numpy().reshape(-1)))

            for metric in self.metrics:
                if metric.name in [config['al']['seqeval_name']]:
                    metric.add_batch(predictions=predictions_seqeval_new, references=labels_seqeval_new)
                else:
                    metric.add_batch(predictions=predictions.view(-1), references=next_batch[config['dataset']['encoded_labels_column_name']].view(-1))

            eval_loss.append(loss.item())
            pbar.set_description(f'{mode} mean loss: {np.mean(eval_loss)}')
            pbar.update(1)

            batch_i += 1
            if batch_i == num_batches_to_eval:
                break

        if print_metrics:

            #if self.model_type != 'tagging':
            return_labels = lambda _int: self.dataset.int_2_labels[_int]
            return_labels = np.vectorize(return_labels)
            labels_all = labels_all.astype(np.int32)
            predictions_all = predictions_all.astype(np.int32)

            labels_all = np.where(labels_all == -100, 0, labels_all)
            labels_all_lst = return_labels(labels_all)
            predictions_all_lst = return_labels(predictions_all)

            logging.info(f'\nMetrics, confusion matrix')

            # Categories: {'alt': 0, 'comp': 1, 'misc': 2, 'rec': 3, 'sci': 4, 'soc': 5, 'talk': 6}
            if self.wandb_on:
                eval_result[config['reporting']['weights_and_biases_confusion_matrix_log_name']] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels_all,
                    preds=predictions_all,
                    class_names=self.dataset.int_2_labels
                )

            conf_matrix = confusion_matrix(labels_all_lst, predictions_all_lst, labels=self.dataset.int_2_labels)
            logging.info(f'\n{conf_matrix}')

            for metric in self.metrics:
                if metric.name in self.METRICS_TO_USE_AVEGARE_IN:
                    if set(labels_all_lst) - set(predictions_all_lst):
                        logging.warning(f'There is label not found in predictions: {set(labels_all_lst) - set(predictions_all_lst)}')
                        logging.warning(f'Printing metric without this label')

                        result = metric.compute(average=self.DEFAULT_AVERAGE_MODE, labels=np.unique(predictions_all))
                    else:
                        result = metric.compute(average=self.DEFAULT_AVERAGE_MODE)
                else:
                    result = metric.compute()

                if metric.name == config['al']['seqeval_name']:
                    for key in result.keys():
                        final_key = metric.name + '_' + key
                        if isinstance(result[key], dict):
                            for _key, value in result[key].items():
                                final_key = metric.name + '_' + key + '_' + _key
                                eval_result[final_key] = value
                        else:
                            eval_result[final_key] = result[key]

                        # if key.startswith('overall'):
                        #     eval_result[metric.name + '_' + key] = result[key]
                else:
                    eval_result[metric.name] = result[metric.name]

                logging.info(f'{eval_result}')

        if mode == config['app']['test_mode']:
            eval_result[
                config['reporting']['weights_and_biases_test_loss_log_name']
            ] = np.mean(eval_loss)
            if self.wandb_on:
                if self.wandb_table is not None:
                    artifact_name = config['reporting']['eval_table_artifact_name']
                    if al_iteration == -1:
                        artifact_name = config['reporting']['full_training_artifacts_name_prefix'] + artifact_name

                    eval_result[
                        config['reporting']['weights_and_biases_test_predictions_table_log_name']
                    ] = self.wandb_table

                    if self.wandb_save_datasets_artifacts:
                        logging.debug(f'Saving evaluation table artifact...')
                        evaluate_table_artifact = wandb.Artifact(artifact_name, type='evaluation')

                        evaluate_table_path = str(Path(self.wandb_run.dir) / f'evaluation_table_{al_iteration}.csv')
                        evaluation_result_df = pd.DataFrame.from_records(evaluation_table_data, columns=self.wandb_table.columns)
                        evaluation_result_df.to_csv(evaluate_table_path, index=False)
                        evaluate_table_artifact.add_file(evaluate_table_path)
                        self.wandb_run.log_artifact(evaluate_table_artifact)

                if al_iteration == -1:
                    new_eval_result = dict()
                    for key, value in eval_result.items():
                        new_eval_result[
                            config['reporting']['full_training_artifacts_name_prefix'] + key
                        ] = value

                    self.wandb_log_data_dict.update(new_eval_result)

                else:
                    self.wandb_log_data_dict.update(eval_result)

                if self.wandb_table is not None:
                    self.wandb_table = wandb.Table(columns=self.wandb_table.columns)

        print()
