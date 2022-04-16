import logging

import torch
from transformers import get_scheduler
from datasets import list_metrics, load_metric

from active_learning_trainer import ALTrainer
from transformers import AutoTokenizer
from dataset import ClassificationDataset, TokenClassificationDataset
from model import Model
import pandas as pd

import datetime
import wandb
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


if __name__ == '__main__':

    config = yaml.safe_load(open(str(APP_CONFIG_FILE_NAME)))

    dataset_config = yaml.safe_load(
        open(
            CONFIGS_FOLDER_PATH / (config['app']['selected_dataset'] + '.yaml')
        )
    )

    # parameters = dict()
    # parameters['use_gpu'] = True
    #
    # parameters['weights_and_biases_on'] = False
    # parameters['weights_and_biases_key'] = '6dcdbcb537587c3adf5e7d6f8ece3d9d5223f3ad'

    current_timestamp = str(datetime.datetime.now()).split('.')[0]
    #
    # parameters['weights_and_biases_save_predictions'] = True
    # parameters['weights_and_biases_save_dataset_artifacts'] = True
    #
    # parameters['pretrained_model_name'] = 'prajjwal1/bert-tiny' #'distilbert-base-uncased'


    # parameters['train_dataset_file_path'] = 'data/imdb/train_IMDB.csv'
    # parameters['val_dataset_file_path'] = 'data/imdb/test_IMDB.csv'
    # parameters['test_dataset_file_path'] = 'data/imdb/test_IMDB.csv'

    # parameters['dataset_from_datasets_hub'] = True
    # parameters['dataset_from_datasets_hub_name'] = 'conll2003'
    # parameters['train_dataset_file_path'] = 'data/news/train.csv'
    # parameters['val_dataset_file_path'] = 'data/news/val.csv'
    # parameters['test_dataset_file_path'] = 'data/news/test.csv'
    # parameters['dataset_file_delimiter'] = ','
    #
    # parameters['dataset_text_column_name'] =  'tokens' #'text'
    # parameters['dataset_label_column_name'] = 'ner_tags'#'airline_sentiment'

    # parameters['train_dataset_file_path'] = 'data/csob/train.csv'
    # parameters['val_dataset_file_path'] = 'data/csob/val.csv'
    # parameters['test_dataset_file_path'] = 'data/csob/test.csv'
    # parameters['dataset_file_delimiter'] = ','
    #
    # ### FIRST IS FOR TAGGING
    # parameters['dataset_text_column_name'] = 'text'  # 'text'
    # parameters['dataset_label_column_name'] = 'category'  # 'airline_sentiment'
    #
    #
    # parameters['dataset_text_column_name'] = 'text_cleaned'  # 'text'
    # parameters['dataset_label_column_name'] = 'label_reduced'  # 'airline_sentiment'

    # # TODO: implement different loss functions in training
    # parameters['loss'] = 'cross_entropy'
    # parameters['loss_weighted'] = False
    #
    # parameters['class_imbalance_reweight'] = True
    # parameters['train_batch_size'] = 32
    # parameters['val_batch_size'] = 64
    # parameters['test_batch_size'] = 64
    # parameters['epochs'] = 5
    # parameters['finetuned_model_type'] = 'classification'
    # parameters['finetuned_model_type'] = 'tagging'
    # model_type = parameters['finetuned_model_type']
    #
    #
    # parameters['al_iterations'] = 10
    # parameters['init_dataset_size'] = 32
    # parameters['add_dataset_size'] = 32
    # #parameters['al_strategy'] = 'random' #'least_confidence'
    # parameters['al_strategy'] = 'least_confidence' #'least_confidence'
    # parameters['full_train'] = False
    #
    # parameters['debug'] = True



    device = 'cpu'
    if config['model']['use_gpu']:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    weights_and_biases_run_name = f'''{current_timestamp}_run_{config['model']['finetuned_model_type']}_{device}'''
    if config['app']['debug_mode']:
        weights_and_biases_run_name = 'DEBUG_' + weights_and_biases_run_name

    logging.info(f'Device set to {device}!')


    ### Preparing data
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['pretrained_model_name']
    )

    dataset_obj = None
    if config['model']['finetuned_model_type'] == config['app']['model_classification_name']:
        dataset_obj = ClassificationDataset(tokenizer)
    elif config['model']['finetuned_model_type'] == config['app']['model_tagging_name']:
        dataset_obj = TokenClassificationDataset(tokenizer)
    else:
        raise NotImplementedError(f'''Type {config['model']['finetuned_model_type']} not supported yet!''')

    if dataset_config['load_from_hub']:
        dataset_name = dataset_config['hub_dataset_name']
        #dataset_name = parameters['dataset_from_datasets_hub_name']
        dataset_obj.load_hosted_dataset(dataset_name)
    else:
        data_files = {
            config['app']['dataset_train_key']: [dataset_config['train_file_path']],
            config['app']['dataset_val_key']: [dataset_config['val_file_path']],
            config['app']['dataset_test_key']: [dataset_config['test_file_path']]
        }

        dataset_obj.load_csv_dataset(
            data_files,
            delimiter=dataset_config['delimiter']
        )

    dataset_obj.truncate_dataset(
        config['app']['dataset_train_key'],
        config['dataset']['train_dataset_size'],
        shuffle=config['dataset']['shuffle_datasets']
    )
    dataset_obj.truncate_dataset(
        config['app']['dataset_val_key'],
        config['dataset']['val_dataset_size'],
        shuffle=config['dataset']['shuffle_datasets']
    )
    dataset_obj.truncate_dataset(
        config['app']['dataset_test_key'],
        config['dataset']['val_dataset_size'],
        shuffle=config['dataset']['shuffle_datasets']
    )

    dataset_obj.prepare_dataset(
        dataset_config['label_column_name'],
        dataset_config['text_column_name'],
    )

    ### Prepare W&B structure for saving predictions
    wandb_table = None
    if config['reporting']['weights_and_biases_on'] and \
        config['reporting']['weights_and_biases_save_predictions']:

        categories = sorted(list(dataset_obj.get_all_categories().items()), key=lambda key: key[1])
        categories_names = [_tuple[0] for _tuple in categories]
        wandb_table_columns = config['reporting']['weights_and_biases_eval_table_columns'] + categories_names
        wandb_table = wandb.Table(columns=wandb_table_columns)

    logging.info(f'Categories: {dataset_obj.get_all_categories()}')
    num_labels = dataset_obj.get_num_categories()

    model = Model(
        config['model']['pretrained_model_name'],
        model_type=config['model']['finetuned_model_type'],
        num_labels=num_labels
    )

    run = None

    this_run_folder_path = LOCAL_RUNS_FOLDER_PATH / weights_and_biases_run_name
    this_run_folder_path.mkdir(exist_ok=True, parents=True)

    if config['reporting']['weights_and_biases_on']:
        wandb.login(key=config['reporting']['weights_and_biases_key'])
        run = wandb.init(
            name=weights_and_biases_run_name,
            project=config['reporting']['project_name'],
         #   reinit=True
        )

        wandb.config.update(config)
        wandb.config.update(dataset_config)

        wandb.watch(model.model)
        artifact = wandb.Artifact(
            config['reporting']['init_dataset_artifact_name'],
            type='dataset'
        )

        for _dataset in dataset_obj.dataset.keys():
            save_path = str(this_run_folder_path / (_dataset + '_dataset.csv'))
            dataset_df = pd.DataFrame.from_dict(
                dataset_obj.dataset[_dataset].to_dict(32)
            ).to_csv(save_path, index=False)

            artifact.add_file(save_path)

        run.log_artifact(artifact)

    trainer = ALTrainer(
        wandb_on=config['reporting']['weights_and_biases_on'],
        imbalanced_training=config['model']['class_imbalance_reweight'],
        model_type=config['model']['finetuned_model_type'],
        wandb_run=run,
        wandb_table=wandb_table,
        wandb_save_datasets_artifacts=config['reporting']['weights_and_biases_save_dataset_artifacts']
    )
    trainer.set_model(model)

    # TODO: add strategy
    trainer.set_strategy(None)
    trainer.set_dataset(dataset_obj)
    dataset_obj.prepare_dataloaders(
        train_batch_size=config['model']['train_batch_size'],
        val_batch_size=config['model']['val_batch_size'],
        test_batch_size=config['model']['test_batch_size'],
    )

    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=config['model']['learning_rate']
    )
    trainer.set_optimizer(optimizer)

    num_training_steps = config['model']['train_epochs'] * trainer.get_training_steps_num()

    # TODO: Implement LR scheduler or decide if it is needed or not (basing on the way how BERT models are fine-tuned (max 5 epochs)
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )
    # trainer.set_lr_scheduler(lr_scheduler)
    trainer.set_device(device)

    trainer.add_evaluation_metric(load_metric('accuracy'))

    metrics_list = []
    if config['model']['finetuned_model_type'] == config['app']['model_tagging_name']:
        metrics_list = config['model']['metrics']['tagging']
    elif config['model']['finetuned_model_type'] == config['app']['model_classification_name']:
        metrics_list = config['model']['metrics']['classification']
    else:
        raise NotImplementedError(f'''There is no such model type implemented like {config['app']['finetuned_model_type']}''')

    for metric in metrics_list:
        trainer.add_evaluation_metric(load_metric(metric))

    if config['al']['full_train']:
        trainer.full_train(
            train_epochs=config['model']['train_epochs'],
            train_batch_size=config['model']['train_batch_size'],
            val_batch_size=config['model']['val_batch_size'],
            test_batch_size=config['model']['test_batch_size'],
            debug=config['app']['debug_mode']
        )

    trainer.al_train(
        al_iterations=config['al']['num_iterations'],
        init_dataset_size=config['al']['init_dataset_size'],
        add_dataset_size=config['al']['add_dataset_size'],
        train_epochs=config['model']['train_epochs'],
        strategy=config['app']['strategy'],
        train_batch_size=config['model']['train_batch_size'],
        val_batch_size=config['model']['val_batch_size'],
        test_batch_size=config['model']['test_batch_size'],
        debug=config['app']['debug_mode']
    )

