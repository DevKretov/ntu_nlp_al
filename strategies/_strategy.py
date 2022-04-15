import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

import torch
import torch.nn.functional as F
import tqdm
import sys

from pathlib import Path


from pathlib import Path
import yaml

CONFIGS_FOLDER_NAME = 'configs'
APP_CONFIG_FILE_NAME = 'config.yaml'
CONFIGS_FOLDER_PATH = Path(__file__).resolve().parent.parent / CONFIGS_FOLDER_NAME
APP_CONFIG_FILE_NAME = CONFIGS_FOLDER_PATH / APP_CONFIG_FILE_NAME
config = yaml.safe_load(open(str(APP_CONFIG_FILE_NAME)))

class _Strategy:

    def __init__(
            self,
            model,
            unlabelled_dataset_dataloader,
            unlabelled_dataset_length,
            device,
            model_type = config['app']['model_classification_name']
    ):

        self.model = model
        self.unlabelled_dataset_dataloader = unlabelled_dataset_dataloader
        self.unlabelled_dataset_length = unlabelled_dataset_length
        self.device = device

        self.model_type = model_type
        self.log_output_folder = config['strategies']['log_output_folder']
        self.log_output_folder_path = Path(self.log_output_folder)

        self.log_output_folder_path.mkdir(exist_ok=True)

        self.training_dict_keys = config['model']['training_dict_keys']

    def update_dataloader(self, new_dataloader):
        self.unlabelled_dataset_dataloader = new_dataloader

    def update_dataset_len(self, new_dataset_len):
        self.unlabelled_dataset_length = new_dataset_len

    def create_logits(self):
        model = self.model.model
        logging.debug(f'Is model on CUDA - {model.device}')
        model.eval()

        num_batches = len(self.unlabelled_dataset_dataloader)

        logging.debug('AL evaluation iteration')
        pbar = tqdm.trange(
            num_batches,
            desc="AL evaluation iteration",
            smoothing=0.05,
            disable=False,
            position=0,
            leave=True,
            file=sys.stdout
        )

        batch_i = 0
        logits_all = None
        all_labels = None
        all_labels_tagging = []
        all_logits_tagging = []
        all_masks = []
        batches_data = []

        for next_batch in self.unlabelled_dataset_dataloader:

            batches_data.append(next_batch)
            if self.model_type == config['app']['model_tagging_name']:
                next_batch, next_batch_metadata = next_batch

            next_batch = {your_key: next_batch[your_key] for your_key in self.training_dict_keys}
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}
            if self.model_type == config['app']['model_classification_name']:
                if all_labels is None:
                    all_labels = next_batch[config['dataset']['encoded_labels_column_name']]
                else:
                    all_labels = torch.cat(
                        (
                            all_labels,
                            next_batch[config['dataset']['encoded_labels_column_name']]
                        ),
                        dim=0
                    )
            elif self.model_type == config['app']['model_tagging_name']:
                all_labels_tagging.append(next_batch[config['dataset']['encoded_labels_column_name']])
                all_masks.append(next_batch[config['dataset']['encoded_labels_column_name']] != -100)
            else:
                pass

            with torch.no_grad():
                outputs = model(**next_batch)
            loss = outputs.loss

            logits = outputs.logits
            if self.model_type == config['app']['model_classification_name']:
                if logits_all is None:
                    logits_all = logits.data.cpu()
                else:
                    logits_all = torch.cat((logits_all, logits.data.cpu()))
            elif self.model_type == config['app']['model_tagging_name']:
                all_logits_tagging.append(logits)
            else:
                pass

            batch_i += 1
            pbar.update(1)
            pbar.set_description(f'AL evaluation iteration. Batch {batch_i:5}/{num_batches}')

        self.all_logits_tagging = all_logits_tagging
        self.all_labels_tagging = all_labels_tagging
        self.all_masks = all_masks

        self.batches_data = batches_data
        if self.model_type == config['app']['model_classification_name']:
            self.logits = logits_all
            self.probs = F.softmax(self.logits, dim=-1)
        elif self.model_type == config['app']['model_tagging_name']:
            self.probs = [F.softmax(_logits, dim=-1) for _logits in self.all_logits_tagging]
        pass

    def query(self, n):
        '''
        Query the indices of unlabelled data for adding to AL dataset
        :param n:
        :return:
        '''

        raise NotImplementedError



