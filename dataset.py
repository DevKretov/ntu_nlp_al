from transformers import DataCollatorForTokenClassification

from datasets import concatenate_datasets
from datasets import load_dataset

from numpy.random import choice
import numpy as np

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

from pathlib import Path
import yaml

CONFIGS_FOLDER_NAME = 'configs'
APP_CONFIG_FILE_NAME = 'config.yaml'
CONFIGS_FOLDER_PATH = Path(__file__).resolve().parent / CONFIGS_FOLDER_NAME
APP_CONFIG_FILE_NAME = CONFIGS_FOLDER_PATH / APP_CONFIG_FILE_NAME
config = yaml.safe_load(open(str(APP_CONFIG_FILE_NAME)))


class Dataset:

    UNIFIED_LABELS_COLUMN_NAME = config['dataset']['encoded_labels_column_name']
    UNIFIED_LABELS_TXT_COLUMN_NAME = config['dataset']['txt_labels_column_name']
    LABELS_STR_COLUMN_NAME = config['dataset']['str_labels_column_name']

    def __init__(self, tokenizer):
        self.rng = np.random.RandomState(2022)
        self.tokenizer = tokenizer

    def load_hosted_dataset(self, dataset_name, revision=None):
       # TODO: maybe we do not need to put default revision name if it does not exist
       # if revision is None:
       #     revision = config['dataset']['hosted_dataset_default_revision_name']

        self.dataset = load_dataset(
            dataset_name,
            revision=revision,
        )

    def set_splits_names(self, train_split_name, val_split_name, test_split_name):
        self.train_split_name = train_split_name
        self.val_split_name = val_split_name
        self.test_split_name = test_split_name

    def load_csv_dataset(self, data_files_dict, delimiter = '|'):
        self.dataset = load_dataset(
            'csv',
            data_files=data_files_dict,
            delimiter=delimiter,
        )

    def truncate_dataset(self, dataset_key, take_max_n = 1000, shuffle = False):
        if dataset_key not in self.dataset.keys():
            logging.error(f'Dataset key: {dataset_key} not in {self.dataset.keys()}!')
            # TODO: rewrite it in normal way, now it's garbage
            non_present_key = set(self.dataset.keys()) - {'train', 'test'}
            non_present_key = list(non_present_key)[0]
            logging.debug(f'Non present key: {non_present_key}')

            self.dataset[dataset_key] = self.dataset[non_present_key]

        assert dataset_key in self.dataset.keys()
        take_max_n = min(take_max_n, len(self.dataset[dataset_key]))
        if not shuffle:
            indices = list(range(0, take_max_n))
        else:
            indices = choice(range(0, len(self.dataset[dataset_key])), take_max_n, replace=False)

        self.dataset[dataset_key] = self.dataset[dataset_key].select(indices)

class ClassificationDataset(Dataset):

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.num_labels = None
        self.data_collator = DataCollatorForTokenClassification(tokenizer)

    def prepare_dataset(self, labels_column_name, input_text_column_name, max_length=256, truncation=True):
        self.prepare_labels(labels_column_name)
        input_text_column_name_new = self.encode_dataset(input_text_column_name, max_length, truncation)

        self.labels_column_name = labels_column_name
        self.input_text_column_name = input_text_column_name_new

    def prepare_labels(self, labels_column_name):


        self.dataset = self.dataset.rename_column(
            labels_column_name,
            self.UNIFIED_LABELS_COLUMN_NAME
        )

        # the next line plays a huge role when the label is already an integer: it casts it back to str so that the whole routine works fine
        # TODO: do nothing if it's already an integer
        self.dataset = self.dataset.map(lambda _entry: {self.UNIFIED_LABELS_COLUMN_NAME: str(_entry[self.UNIFIED_LABELS_COLUMN_NAME])})
        self.dataset = self.dataset.class_encode_column(self.UNIFIED_LABELS_COLUMN_NAME)
        self.int_2_labels = self.dataset['train'].features[self.UNIFIED_LABELS_COLUMN_NAME].__dict__['_int2str']

        self.dataset = self.dataset.map(
            lambda _entry: {self.LABELS_STR_COLUMN_NAME: self.int_2_labels[_entry['labels']]}
        )

    def encode_dataset(self, input_text_column_name, max_length=256, truncation=True):

        if isinstance(input_text_column_name, list):
            columns_to_analyse = input_text_column_name.copy()
            input_text_column_name = 'input_text_prepared'
            #input_text_column_name = input_text_column_name
            # the dataset is a list of columns. merge them
           # splits = self.dataset.keys()#['train', 'val', 'test']
            #for split in splits:
             #   self.dataset[split][input_text_column_name] = [' '.join(_one_tuple) for _one_tuple in zip(*[self.dataset[split][_label_col] for _label_col in columns_to_analyse])]
            self.dataset = self.dataset.map(lambda _entry: {input_text_column_name: ' [SEP] '.join([_entry[_label_col] for _label_col in columns_to_analyse]).strip()})



        self.dataset = self.dataset.map(
            lambda examples: self.tokenizer(
                examples[input_text_column_name],
                max_length=max_length,
                truncation=truncation,
                padding='max_length'),
            batched=True
        )

        # Check the correctness of this operation
        self.dataset.set_format(
            type='torch',
            columns=config['model']['training_dict_keys'],
            output_all_columns=True
        )

        return input_text_column_name

    def get_all_categories(self):
        return self.dataset['train'].features[self.UNIFIED_LABELS_COLUMN_NAME].__dict__['_str2int']

    def get_num_categories(self):
        return len(self.dataset['train'].features[self.UNIFIED_LABELS_COLUMN_NAME].__dict__['_str2int'])

    def update_al_datasets_with_new_batch(self, indices_to_add):
        data_to_add = self.al_train_dataset['unlabelled'].select(indices_to_add)

        self.al_train_dataset['train'] = concatenate_datasets(
            [
                self.al_train_dataset['train'],
                data_to_add
            ]
        )

        self.al_train_dataset['train'].set_format(
            type='torch',
            columns=config['model']['training_dict_keys'],
            output_all_columns=True
        )

        logging.debug(f'Filtering...')
        self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].filter(
            lambda example, indice: indice not in indices_to_add,
            with_indices=True
        )

        self.al_train_dataset['unlabelled'].set_format(
            type='torch',
            columns=config['model']['training_dict_keys'],
            output_all_columns=True
        )

    def prepare_al_datasets(
            self,
            al_init_dataset_size,
    ):
        dataset = self.dataset
        train_dataset_length = len(dataset['train'])

        if config['dataset']['index_column_name'] not in dataset['train'].features.keys():
            dataset['train'] = dataset['train'].add_column(
                config['dataset']['index_column_name'],
                list(range(0, train_dataset_length))
            )

        selected_indices = self.rng.choice(
            range(0, train_dataset_length),
            al_init_dataset_size,
            replace=False
        ).tolist()

        self.al_train_dataset_indices = selected_indices

        al_train_dataset = dataset['train'].filter(
            lambda example: example[config['dataset']['index_column_name']] in selected_indices
        )
        rest_dataset = dataset['train'].filter(
            lambda example: example[config['dataset']['index_column_name']] not in selected_indices
        )

        self.al_train_dataset = {
            'train': al_train_dataset,
            'unlabelled': rest_dataset
        }

        logging.debug(f'AL train dataset length: {len(al_train_dataset)}, rest dataset length: {len(rest_dataset)}')
        assert len(al_train_dataset) + len(rest_dataset) == len(dataset['train'])

    def prepare_dataloaders(
            self,
            train_batch_size=32,
            val_batch_size=64,
            test_batch_size=64,
            imbalanced_training=False,
            al=False
    ):

        if al:
            if config['dataset']['index_column_name'] in self.al_train_dataset['train'].features.keys():
                self.al_train_dataset['train'] = self.al_train_dataset['train'].remove_columns([config['dataset']['index_column_name']])

            if config['dataset']['index_column_name'] in self.al_train_dataset['unlabelled'].features.keys():
                self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].remove_columns([config['dataset']['index_column_name']])

            sampler = None
            shuffle = True

            if imbalanced_training:
                labels = self.dataset['train'][config['dataset']['encoded_labels_column_name']].numpy()
                labels_unique = np.unique(labels)
                class_weights = compute_class_weight(class_weight='balanced', classes=labels_unique, y=labels)

                labels_al = self.al_train_dataset['train'][config['dataset']['encoded_labels_column_name']]  # .tolist()
                samples_weights = [class_weights[_label] for _label in labels_al]
                num_samples = len(self.al_train_dataset['train'][config['dataset']['encoded_labels_column_name']])
                sampler = WeightedRandomSampler(samples_weights, num_samples)
                shuffle = False

            self.train_dataloader = DataLoader(
                self.al_train_dataset['train'],
                shuffle=shuffle,
                batch_size=train_batch_size,
                sampler=sampler
            )

            self.unlabelled_dataloader = DataLoader(
                self.al_train_dataset['unlabelled'],
                batch_size=val_batch_size
            )
        else:
            self.train_dataloader = DataLoader(
                self.dataset['train'],
                shuffle=True,
                batch_size=train_batch_size
            )

        self.val_dataloader = DataLoader(
            self.dataset[self.val_split_name],
            batch_size=val_batch_size
        )

        self.test_dataloader = DataLoader(
            self.dataset['test'],
            batch_size=test_batch_size
        )


class TokenClassificationDataset(Dataset):

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.num_labels = None
        self.data_collator = DataCollatorForTokenClassification(tokenizer)

    def prepare_dataset(self, labels_column_name, tokens_column_name, **kwargs):
        self.input_text_column_name = tokens_column_name
        self.labels_column_name = labels_column_name

        self.dataset = self.dataset.map(
            lambda _entry: {self.LABELS_STR_COLUMN_NAME: _entry[self.labels_column_name]}
        )

        self.dataset = self.dataset.map(self.tokenize_adjust_labels, batched=True)

        self.int_2_labels = self.dataset['train'].features[self.labels_column_name].feature.names

    def get_all_categories(self):

        tags = self.dataset['train'].features[self.labels_column_name].feature.names
        tags_dict = {tag: i for i, tag in enumerate(tags)}

        self.num_labels = len(tags)
        return tags_dict

    def get_num_categories(self):
        if self.num_labels is None:
            self.get_all_categories()
        return self.num_labels

    def prepare_al_datasets(
            self,
            al_init_dataset_size,
    ):

        dataset = self.dataset
        train_dataset_length = len(dataset['train'])

        if config['dataset']['index_column_name'] not in dataset['train'].features.keys():
            dataset['train'] = dataset['train'].add_column(
                config['dataset']['index_column_name'],
                list(range(0, train_dataset_length))
            )

        selected_indices = self.rng.choice(
            range(0, train_dataset_length),
            al_init_dataset_size,
            replace=False
        ).tolist()

        self.al_train_dataset_indices = selected_indices#.tolist()

        al_train_dataset = dataset['train'].filter(
            lambda example: example[config['dataset']['index_column_name']] in selected_indices
        )
        rest_dataset = dataset['train'].filter(
            lambda example: example[config['dataset']['index_column_name']] not in selected_indices
        )

        self.al_train_dataset = {
            'train': al_train_dataset,
            'unlabelled': rest_dataset
        }

        logging.debug(f'AL train dataset length: {len(al_train_dataset)}, rest dataset length: {len(rest_dataset)}')
        assert len(al_train_dataset) + len(rest_dataset) == len(dataset['train'])

    def prepare_dataloaders(
            self,
            train_batch_size=32,
            val_batch_size=64,
            test_batch_size=64,
            imbalanced_training=False,
            al=False
    ):

        if al:
            if config['dataset']['index_column_name'] in self.al_train_dataset['train'].features.keys():
                self.al_train_dataset['train'] = self.al_train_dataset['train'].remove_columns([config['dataset']['index_column_name']])

            if config['dataset']['index_column_name'] in self.al_train_dataset['unlabelled'].features.keys():
                self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].remove_columns([config['dataset']['index_column_name']])

            # imbalanced training resampling

            sampler = None
            shuffle = True

            # if imbalanced_training:
            #     labels = self.dataset['train']['labels'].numpy()
            #     labels_unique = np.unique(labels)
            #     class_weights = compute_class_weight(class_weight='balanced', classes=labels_unique, y=labels)
            #
            #     labels_al = self.al_train_dataset['train']['labels'].tolist()
            #     samples_weights = [class_weights[_label] for _label in labels_al]
            #     num_samples = len(self.al_train_dataset['train']['labels'])
            #     sampler = WeightedRandomSampler(samples_weights, num_samples)
            #     shuffle = False

            self.train_dataloader = DataLoader(
                self.al_train_dataset['train'],
                shuffle=shuffle,
                batch_size=train_batch_size,
                sampler=None,
                collate_fn=self.processing_function
            )

            self.unlabelled_dataloader = DataLoader(
                self.al_train_dataset['unlabelled'],
                batch_size=val_batch_size,
                collate_fn=self.processing_function
            )
        else:
            self.train_dataloader = DataLoader(
                self.dataset['train'],
                shuffle=True,
                batch_size=train_batch_size,
                collate_fn=self.processing_function
            )

        self.val_dataloader = DataLoader(
            self.dataset[self.val_split_name],
            batch_size=val_batch_size,
            collate_fn=self.processing_function
        )

        self.test_dataloader = DataLoader(
            self.dataset['test'],
            batch_size=test_batch_size,
            collate_fn=self.processing_function
        )


    def processing_function(self, batch):

        list_of_keys_to_collate = config['model']['training_dict_keys']
        list_of_keys_not_to_collate = set(batch[0].keys()) - set(list_of_keys_to_collate)
        batch_to_collate = [
            {_key: _one_entry[_key] for _key in list_of_keys_to_collate} for _one_entry in batch
        ]
        batch_not_to_collate = [
            {_key: _one_entry[_key] for _key in list_of_keys_not_to_collate} for _one_entry in batch
        ]

        batch = self.data_collator(batch_to_collate)
        return batch, batch_not_to_collate


    def tokenize_adjust_labels(self, all_samples_per_split):
        tokenized_samples = self.tokenizer.batch_encode_plus(
            all_samples_per_split[self.input_text_column_name],
            is_split_into_words=True,

        )
        # tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used
        # so the new keys [input_ids, labels (after adjustment)]
        # can be added to the datasets dict for each train test validation split
        total_adjusted_labels = []

        for k in range(0, len(tokenized_samples[config['model']['encoded_input_ids_column_name']])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split[self.labels_column_name][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if (wid is None):
                    adjusted_label_ids.append(-100)
                elif (wid != prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    #label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(existing_label_ids[i])

            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples[self.UNIFIED_LABELS_COLUMN_NAME] = total_adjusted_labels
        return tokenized_samples

    def update_al_datasets_with_new_batch(self, indices_to_add):
        data_to_add = self.al_train_dataset['unlabelled'].select(indices_to_add)
        self.al_train_dataset['train'] = concatenate_datasets(
            [
                self.al_train_dataset['train'],
                data_to_add
            ]
        )

        logging.debug(f'Filtering...')
        self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].filter(
            lambda example, indice: indice not in indices_to_add,
            with_indices=True
        )

