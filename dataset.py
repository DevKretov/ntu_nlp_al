from transformers import DataCollatorWithPadding

from datasets import concatenate_datasets
from datasets import load_dataset
from scipy.stats import entropy

from numpy.random import choice
import tensorflow as tf
import numpy as np

class Dataset:

    UNIFIED_LABELS_COLUMN_NAME = 'labels'
    UNIFIED_LABELS_TXT_COLUMN_NAME = 'labels_txt'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # self.data_collator = DataCollatorWithPadding(
        #     tokenizer=self.tokenizer,
        #     return_tensors="pt"
        # )
        pass

    def load_csv_dataset(self, data_files_dict, delimiter = '|'):
        self.dataset = load_dataset(
            'csv',
            data_files=data_files_dict,
            delimiter=delimiter,
        )

    def truncate_dataset(self, dataset_key, take_max_n = 1000, shuffle = False):

        assert dataset_key in self.dataset.keys()

        take_max_n = min(take_max_n, len(self.dataset[dataset_key]))
        if not shuffle:

            indices = list(range(0, take_max_n))
        else:
            indices = choice(range(0, len(self.dataset[dataset_key])), take_max_n, replace=False)

        self.dataset[dataset_key] = self.dataset[dataset_key].select(indices)


    def prepare_labels(self, labels_column_name):
        # convert labels to ids

        # self.dataset = self.dataset.map(
        #     lambda examples: {labels_column_name: examples[]},
        #     batched=True
        # )

        self.dataset = self.dataset.rename_column(
            labels_column_name,
            self.UNIFIED_LABELS_COLUMN_NAME
        )
        self.dataset = self.dataset.class_encode_column(self.UNIFIED_LABELS_COLUMN_NAME)
        self.int_2_labels = self.dataset['train'].features[self.UNIFIED_LABELS_COLUMN_NAME].__dict__['_int2str']

        self.dataset = self.dataset.map(
            lambda _entry: {'label_str': self.int_2_labels[_entry['labels']]})

        

    def encode_dataset(self, input_text_column_name, max_length=256, truncation=True):

        self.dataset = self.dataset.map(
            lambda examples: self.tokenizer(
                examples[input_text_column_name],
                max_length=max_length,
                truncation=truncation,
                padding='max_length'),
            batched=True
        )

        #self.dataset = self.dataset.remove_columns([input_text_column_name])
        #self.dataset = self.dataset.rename_column("label", "labels")
        # Check the correctness of this operation
        self.dataset.set_format(
            type='torch',
            columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
            output_all_columns=True
        )

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
        self.al_train_dataset['unlabelled'] = self.al_train_dataset['unlabelled'].filter(
            lambda example, indice: indice not in indices_to_add,
            with_indices=True
        )

        self.al_train_dataset_indices.append(
            data_to_add['index']
        )

       # print(f'Train dataset: before: {len(al_train_dataset)}, after: {len(new_training_dataset)}')
        #print(f'Rest dataset: before: {len(rest_dataset)}, after: {len(new_rest_dataset)}')


    def prepare_al_datasets(self, al_init_dataset_size):
        train_dataset_length = len(self.dataset['train'])

        if 'index' not in self.dataset['train'].features.keys():
            self.dataset['train'] = self.dataset['train'].add_column(
                'index',
                list(range(0, train_dataset_length))
            )

        selected_indices = choice(
            range(0, train_dataset_length),
            al_init_dataset_size,
            replace=False
        )

        self.al_train_dataset_indices = selected_indices.tolist()

        al_train_dataset = self.dataset['train'].filter(lambda example: example['index'] in selected_indices)
        al_train_dataset = al_train_dataset.map(lambda ex, ind: {'dataset_index': ind}, with_indices=True)#['index_dataset']
        rest_dataset = self.dataset['train'].filter(lambda example: example['index'] not in selected_indices)
        rest_dataset = rest_dataset.map(lambda ex, ind: {'dataset_index': ind}, with_indices=True)#['index_dataset']

        self.al_train_dataset = {
            'train': al_train_dataset,
            'unlabelled': rest_dataset
        }

        print(f'AL train dataset length: {len(al_train_dataset)}, rest dataset length: {len(rest_dataset)}')
        assert len(al_train_dataset) + len(rest_dataset) == len(self.dataset['train'])


    def prepare_al_datasets_tf(self, train_batch_size):

        self.al_train_dataset_tf = {
            'train': None,
            'unlabelled': None
        }


        self.al_train_dataset_tf['train'] = self.al_train_dataset['train'].to_tf_dataset(
            columns=['dataset_index', 'input_ids', 'token_type_ids', 'attention_mask'],
            label_cols=["labels"],
            shuffle=True,
            batch_size=train_batch_size,
            collate_fn=self.data_collator,
        )

        self.al_train_dataset_tf['unlabelled'] = self.al_train_dataset['unlabelled'].to_tf_dataset(
            columns=['dataset_index', 'input_ids', 'token_type_ids', 'attention_mask'],
            label_cols=["labels"],
            shuffle=True,
            batch_size=128,
            collate_fn=self.data_collator,
        )

        print(f'AL TF train dataset length: {len(self.al_train_dataset["train"])}, rest dataset length: {len(self.al_train_dataset["unlabelled"])}')

    def prepare_tf_datasets(self, val_batch_size = 64, test_batch_size = 64):

        self.tf_datasets = {
            'train': None,
            'test': None,
            'val': None
        }

        if 'val' in self.dataset.keys():
            self.tf_datasets['val'] = self.dataset["val"].to_tf_dataset(
                  columns=['input_ids', 'token_type_ids', 'attention_mask'],
                  label_cols=["labels"],
                  shuffle=True,
                  batch_size=val_batch_size,
                  collate_fn=self.data_collator,
            )

        if 'test' in self.dataset.keys():
            self.tf_datasets['test'] = self.dataset["test"].to_tf_dataset(
                  columns=['input_ids', 'token_type_ids', 'attention_mask'],
                  label_cols=["labels"],
                  shuffle=True,
                  batch_size=test_batch_size,
                  collate_fn=self.data_collator,
            )