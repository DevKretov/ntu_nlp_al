import time
import tqdm #import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import numpy as np
import pandas as pd
import datasets
from dataset import Dataset
from datasets import concatenate_datasets

import sys

from strategies import *




class ALTrainer:

    def __init__(self):

        self.lr_scheduler = None
        self.metrics = []
        pass

    def set_model(self, model):
        self.model = model

    def set_strategy(self, strategy):
        self.strategy = strategy

    def set_dataset(self, dataset:Dataset):
        self.dataset = dataset

    def prepare_dataloaders(
            self,
            train_batch_size = 32,
            val_batch_size = 64,
            test_batch_size = 64,
            al=False
    ):

        if al:
            if 'index' in self.al_train_dataset['train'].features.keys():
                self.al_train_dataset['train'] = self.al_train_dataset['train'].remove_columns(["index"])
            self.al_train_dataset['train'] = self.al_train_dataset['train'].remove_columns(["dataset_index"])

            if 'index' in self.al_train_dataset['unlabelled'].features.keys():
                self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].remove_columns(["index"])
            self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].remove_columns(["dataset_index"])
            self.train_dataloader = DataLoader(
                self.al_train_dataset['train'],
                shuffle=True,
                batch_size=train_batch_size
            )

            self.unlabelled_dataloader = DataLoader(
                self.al_train_dataset['unlabelled'],
                batch_size = train_batch_size
            )
        else:
            self.train_dataloader = DataLoader(
                self.dataset.dataset['train'],
                shuffle=True,
                batch_size=train_batch_size
            )

        self.val_dataloader = DataLoader(
            self.dataset.dataset['val'],
            batch_size=val_batch_size
        )

        self.test_dataloader = DataLoader(
            self.dataset.dataset['val'],
            batch_size=test_batch_size
        )


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

    def send_model_to_devide(self):
        self.determine_device()
       # self.model = self.model.to(self.device)

    def determine_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_training_steps_num(self):
        return len(self.train_dataloader)

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

        print(f'Full training initialized!')

        self.train_batch_size = train_batch_size

        self.prepare_dataloaders(
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            al=False
        )

        print(f'Training is run on {len(self.train_dataloader)} batches!')
        print(f'Evaluation is run on {len(self.val_dataloader)} batches!')
        print(f'Testing is run on {len(self.test_dataloader)} batches!')

        print(f'\n=========================\n')

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

        print(f'\nFull training finished')


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

        print(f'Training initialized!')

        self.train_batch_size = train_batch_size

        self.prepare_al_datasets(init_dataset_size)
        self.prepare_dataloaders(
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            al=True
        )

        print(f'Training is run on {len(self.train_dataloader)} batches!')
        print(f'Evaluation is run on {len(self.val_dataloader)} batches!')
        print(f'Testing is run on {len(self.test_dataloader)} batches!')

        print(f'\n=========================\n')

        if isinstance(strategy, str):

            if strategy.lower().strip() == 'random':
                strategy = RandomStrategy(
                    self.model,
                    self.unlabelled_dataloader,
                    len(self.al_train_dataset['unlabelled']),
                    self.device
                )
            elif strategy.lower().strip() == 'least_confidence':
                strategy = LeastConfidence(
                    self.model,
                    self.unlabelled_dataloader,
                    len(self.al_train_dataset['unlabelled']),
                    self.device
                )
            elif strategy.lower().strip() == 'badge':

                strategy = BadgeSampling(
                    self.model,
                    self.unlabelled_dataloader,
                    len(self.al_train_dataset['unlabelled']),
                    self.device,
                    num_labels=self.model.num_labels,
                    embedding_dim=self.model.model.config.hidden_size,
                    batch_size=train_batch_size
                )

            elif strategy.lower().strip() == 'entropy':
                strategy = EntropySampling(
                    self.model,
                    self.unlabelled_dataloader,
                    len(self.al_train_dataset['unlabelled']),
                    self.device
                )
            elif strategy.lower().strip() == 'kmeans':
                strategy = KMeansSampling(
                    self.model,
                    self.unlabelled_dataloader,
                    len(self.al_train_dataset['unlabelled']),
                    self.device,
                    num_labels=self.model.num_labels,
                    embedding_dim=self.model.model.config.hidden_size,
                    batch_size=train_batch_size
                )

            else:
                pass



        for al_iteration in range(al_iterations):
            print(f'\n===========================================================')
            print(f'\nAL iteration {(al_iteration + 1):3}/{al_iterations}')


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

            print(f'Model trained! Running AL strategy...')

            strategy.update_dataloader(self.unlabelled_dataloader)
            strategy.update_dataset_len(len(self.al_train_dataset['unlabelled']))

            indices = strategy.query(
                add_dataset_size
            )

            print(f'Returned {len(indices)} indices from strategy')

            al_train_dataset_size_before = len(self.al_train_dataset['train'])
            al_unlabelled_dataset_size_before = len(self.al_train_dataset['unlabelled'])

            print(f'Before updating AL datasets: train size = {al_train_dataset_size_before}, unlabelled size = {al_unlabelled_dataset_size_before}, sum: {al_train_dataset_size_before + al_unlabelled_dataset_size_before} ')

            self.update_al_datasets_with_new_batch(
                indices_to_add=indices
            )

            al_train_dataset_size_after = len(self.al_train_dataset['train'])
            al_unlabelled_dataset_size_after = len(self.al_train_dataset['unlabelled'])

            print(f'\nUpdated AL datasets: train size = {al_train_dataset_size_after}, unlabelled size = {al_unlabelled_dataset_size_after}, sum: {al_train_dataset_size_after + al_unlabelled_dataset_size_after} ')

            assert (
                (al_train_dataset_size_before + al_unlabelled_dataset_size_before)
                ==
                (al_train_dataset_size_after + al_unlabelled_dataset_size_after)
            )

            self.prepare_dataloaders(
                train_batch_size=train_batch_size,
                val_batch_size=val_batch_size,
                test_batch_size=test_batch_size,
                al=True
            )

    def train_model(
            self,
            epochs,
            steps_per_epoch = -1,
            evaluation_steps_num = -1,
    ):

        self.model.reinit_model()

        model = self.model.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        self.set_optimizer(optimizer)
        print(f'Model initialized.')

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        model.to(self.device)

        if steps_per_epoch == -1:
            steps_per_epoch = len(self.train_dataloader)

        for epoch_i in range(epochs):
            model.train()
            print(f'\n\nEpoch {(epoch_i + 1):3}')

            losses_list =  []
            start_time = time.time()

            pbar = tqdm.trange(
                len(self.train_dataloader),
                desc="Iteration",
                smoothing=0.05,
                disable=False,
                position=0,
                leave=True,
                file=sys.stdout
            )

            step_i = 0

            for next_batch in self.train_dataloader:
                model.zero_grad()

                next_batch = {k: v.to(self.device) for k, v in next_batch.items()}

                # TODO: Check model output
                # TODO: Add accuracy and all other scores here
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
            print(f'\n\nEpoch finished. Evaluation:')

            if evaluation_steps_num == -1:
                evaluation_steps_num = len(self.val_dataloader)

            self.evaluate(
                num_batches_to_eval=evaluation_steps_num,
                dataloader=self.val_dataloader,
                print_metrics=False,
                mode='Eval'
            )

        print(f'Test set evaluation:')
        self.evaluate(
            num_batches_to_eval=-1,
            dataloader=self.test_dataloader,
            print_metrics=True,
            mode='Test'
        )


    def evaluate(self, num_batches_to_eval=-1, dataloader = None, print_metrics = False, mode = 'Eval'):

        model = self.model.model

        if dataloader is None:
            dataloader = self.val_dataloader

        print(f'Running {mode} mode...')
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

        for next_batch in dataloader:
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}
            with torch.no_grad():
                outputs = model(**next_batch)
            loss = outputs.loss

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)


            for metric in self.metrics:
                metric.add_batch(predictions=predictions, references=next_batch["labels"])

            eval_loss.append(loss.item())
            pbar.set_description(f'{mode} mean loss: {np.mean(eval_loss)}')
            pbar.update(1)

            batch_i += 1
            if batch_i == num_batches_to_eval:
                break

        if print_metrics:
            print(f'\nMetrics')
            for metric in self.metrics:
                result = metric.compute()
                print(f'{result}')
        print()

    def prepare_al_datasets(
            self,
            al_init_dataset_size,
    ):


        dataset = self.dataset.dataset

        train_dataset_length = len(dataset['train'])

        if 'index' not in dataset['train'].features.keys():
            dataset['train'] = dataset['train'].add_column(
                'index',
                list(range(0, train_dataset_length))
            )

        selected_indices = np.random.choice(
            range(0, train_dataset_length),
            al_init_dataset_size,
            replace=False
        ).tolist()

        self.al_train_dataset_indices = selected_indices#.tolist()

        al_train_dataset = dataset['train'].filter(lambda example: example['index'] in selected_indices)
        al_train_dataset = al_train_dataset.map(lambda ex, ind: {'dataset_index': ind}, with_indices=True)#['index_dataset']
        rest_dataset = dataset['train'].filter(lambda example: example['index'] not in selected_indices)
        rest_dataset = rest_dataset.map(lambda ex, ind: {'dataset_index': ind}, with_indices=True)#['index_dataset']

        self.al_train_dataset = {
            'train': al_train_dataset,
            'unlabelled': rest_dataset
        }

        print(f'AL train dataset length: {len(al_train_dataset)}, rest dataset length: {len(rest_dataset)}')
        assert len(al_train_dataset) + len(rest_dataset) == len(dataset['train'])



    def update_al_datasets_with_new_batch(self, indices_to_add):
        #dataset = self.dataset.dataset

        data_to_add = self.al_train_dataset['unlabelled'].select(indices_to_add)
       # data_to_add.set_format(type='torch')
       # print('')
        self.al_train_dataset['train'] = concatenate_datasets(
            [
                self.al_train_dataset['train'],
                data_to_add
            ]
        )
        self.al_train_dataset['train'].set_format(type='torch')
        self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].filter(
            lambda example, indice: indice not in indices_to_add,
            with_indices=True
        )

        self.al_train_dataset['unlabelled'] =  self.al_train_dataset['unlabelled'].map(lambda ex, ind: {'dataset_index': ind}, with_indices=True)  # ['index_dataset']
        self.al_train_dataset['train'] = self.al_train_dataset['train'].map(
            lambda ex, ind: {'dataset_index': ind}, with_indices=True)  # ['index_dataset']

        self.al_train_dataset_indices.append(
            indices_to_add
        )

#def al_train(self):
