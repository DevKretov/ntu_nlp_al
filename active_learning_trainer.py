import time
import tqdm #import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import numpy as np
import pandas as pd
import datasets
from dataset import Dataset


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
            test_batch_size = 64
    ):

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



    def prepare_al_datasets(self):
        pass


    # TODO: decide if needed considering transformers...
    def set_criterion(self, criterion):
        self.criterion = criterion
        pass

    def set_optimizer(self, optimizer:torch.optim.Optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, scheduler:torch.optim.lr_scheduler._LRScheduler):
        self.lr_scheduler = scheduler

    def send_model_to_devide(self):
        self.determine_device()
       # self.model = self.model.to(self.device)

    def determine_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_training_steps_num(self):
        return len(self.train_dataloader)

    def add_evaluation_metric(self, metric_obj):
        self.metrics.append(metric_obj)


    def train_model(
            self,
            epochs,
            steps_per_epoch = -1
    ):

        model = self.model.model

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        model.to(self.device)

        if steps_per_epoch == -1:
            steps_per_epoch = len(self.train_dataloader)

        for epoch_i in range(epochs):
            model.train()
            print(f'\nEpoch {epoch_i:3}')

            losses_list =  []
            start_time = time.time()

            pbar = tqdm.trange(
                steps_per_epoch,
                desc="Iteration",
                smoothing=0.05,
                disable=False,
                position=0,
                leave=True
            )

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



                # if batch_i % print_train_stats_each == 0 and batch_i > 0:
                #     print(f'Train Epoch {epoch_i:3d}, batch {(batch_i + 1):4}/{num_batches_per_epoch} complete. '
                #           f'Mean loss: {(total_loss / (batch_i + 1)):3.5f} '
                #           f'Mean acc: {(total_acc / (batch_i + 1)):3.5f} '
                #           f'Mean perplexity: {(total_perplexity / (batch_i + 1)):3.5f} '
                #           f'Time: {(time.time() - start_time):3.5f}.')
                #
                #     #  print('| epoch {:3d} | {:5d}/{:5d} batches  | '
                #     #        'loss {:5.2f} | acc {:0.2f} | ppl {:8.2f}'.format(
                #     #      epoch_i, batch_i, num_batches_per_epoch, loss.item(), batch_accuracy, math.exp(loss.item())))
                #     # # total_loss = 0
                #     start_time = time.time()
                pbar.set_description(f'Mean loss: {np.mean(losses_list)}')
                pbar.update(1)
        print(f'Epoch finished. Evaluation:')

        self.evaluate()
            # mean_loss = round(total_loss / num_batches_per_epoch, 2)
            # mean_acc = round(total_acc / num_batches_per_epoch, 2)
          #  print(f'Training results: mean loss: {mean_loss}, mean_acc: {mean_acc}')

        #     if epoch_i % eval_each == 0:
        #
        #         val_loss, val_perplexity = self.evaluate(num_batches_to_eval=num_batches_to_eval_on)
        #
        #         # Save the model if the validation loss is the best we've seen so far.
        #         if not best_val_perplexity or val_perplexity < best_val_perplexity:
        #             print(
        #                 f'Saving the best model! Perplexity: {val_perplexity:3.5f} vs best perplexity {best_val_perplexity}')
        #             with open(self.save_model_dir_path / self.best_model_name, 'wb') as f:
        #                 torch.save(self.model, f)
        #             best_val_perplexity = val_perplexity
        #             no_improvement_epoch = 0
        #         else:
        #             no_improvement_epoch += 1
        #
        #             if early_stopping_tolerance != -1 and no_improvement_epoch > early_stopping_tolerance:
        #                 print(
        #                     f'Stopping training! {no_improvement_epoch} epochs no improvement, tolerance was {early_stopping_tolerance}')
        #                 break
        # best_model_path = str(self.save_model_dir_path / self.best_model_name)
        #return best_model_path

    def evaluate(self, num_batches_to_eval=-1, print_every=100):

        model = self.model.model

        print(f'Running evaluation...')
        if num_batches_to_eval == -1:
            num_batches_to_eval = len(self.val_dataloader)


        model.eval()

        start_time = time.time()

        print(f'Evaluation is run on {num_batches_to_eval} batches!')

        pbar = tqdm.trange(
            num_batches_to_eval,
            desc="Iteration",
            smoothing=0.05,
            disable=False,
            position=0,
            leave=True
        )

        eval_loss = []


        for next_batch in self.train_dataloader:
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}
            with torch.no_grad():
                outputs = model(**next_batch)
            loss = outputs.loss

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)


            for metric in self.metrics:
                metric.add_batch(predictions=predictions, references=next_batch["labels"])

            eval_loss.append(loss.item())
            pbar.set_description(f'Mean loss: {np.mean(eval_loss)}')
            pbar.update(1)

        for metric in self.metrics:
            metric.compute()
            print(f'Metric: {metric}')

#def al_train(self):
