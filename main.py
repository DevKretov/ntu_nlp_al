import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.INFO)

import torch
from transformers import get_scheduler
from datasets import list_metrics, load_metric

from active_learning_trainer import ALTrainer
from transformers import AutoTokenizer
from dataset import ClassificationDataset, TokenClassificationDataset
from utils import TrainingVisualisation
from model import Model
import pandas as pd

import datetime
import wandb
from pathlib import Path
import json
import yaml
import argparse

LOCAL_RUNS_FOLDER = 'runs'
LOCAL_RUNS_FOLDER_PATH = Path(LOCAL_RUNS_FOLDER)

TRASK = True
CONFIGS_FOLDER_NAME = 'configs'
APP_CONFIG_FILE_NAME = 'config.yaml'
CONFIGS_FOLDER_PATH = Path(__file__).resolve().parent / CONFIGS_FOLDER_NAME
APP_CONFIG_FILE_NAME = CONFIGS_FOLDER_PATH / APP_CONFIG_FILE_NAME

# TODO: Init wandb run for each strategy

if __name__ == '__main__':

    config = yaml.safe_load(open(str(APP_CONFIG_FILE_NAME)))

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset_path",
        required=False,
        default=config['run']['selected_dataset'],
        help = "Location of dataset config file "
    )
    parser.add_argument(
        "--run_name_suffix",
        required=False,
        help="Suffix for run name"
    )
    parser.add_argument(
        "--al_strategy",
        required=False,
        help="Specify one strategy to override those in config. But only one"
    )

    parser.add_argument(
        "--pretrained_model_name",
        required=False,
        help="The name of the pretrained model to use as starting point in training"
    )

    parser.add_argument(
        "--add_dataset_size",
        required=False,
        default=32,
        type=int,
        help="The name of the pretrained model to use as starting point in training"
    )

    # init_dataset_size
    parser.add_argument(
        "--init_dataset_size",
        required=False,
        default=32,
        type=int,
        help="Initial size of the training dataset"
    )

    #finetuned_model_type
    parser.add_argument(
        "--finetuned_model_type",
        required=False,
        help="The type of the model to tune: either classification or tagging"
    )

    # learning_rate
    parser.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        default=5e-5,
        help="Learning rate of the model"
    )

    # train_epochs
    parser.add_argument(
        "--train_epochs",
        required=False,
        type=int,
        default=5,
        help="Number of training epochs for each al step"
    )

    args = parser.parse_args()

    learning_rate = args.learning_rate if args.learning_rate is not None else config['model']['learning_rate']
    print(f'LR = {learning_rate}')

    train_epochs = args.train_epochs if args.train_epochs is not None else config['model']['train_epochs']

    # add_dataset_size
    add_dataset_size = args.add_dataset_size if args.add_dataset_size is not None else config['al']['add_dataset_size']
    init_dataset_size = args.init_dataset_size if args.init_dataset_size is not None else config['al']['init_dataset_size']
    logging.info(f'Init dataset size: {init_dataset_size}')
    logging.info(f'Add dataset size: {add_dataset_size}')

    dataset_path = str(CONFIGS_FOLDER_PATH / (args.dataset_path + '.yaml'))
    dataset_config = yaml.safe_load(open(dataset_path))
    pretrained_model_name = args.pretrained_model_name if args.pretrained_model_name is not None else config['run']['pretrained_model_name']

    current_timestamp = str(datetime.datetime.now()).split('.')[0]

    device = 'cpu'
    if config['model']['use_gpu']:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    weights_and_biases_run_name = f'''{current_timestamp}_run_{config['run']['finetuned_model_type']}_{pretrained_model_name}_{device}_{args.run_name_suffix}'''
    if config['app']['debug_mode']:
        weights_and_biases_run_name = 'DEBUG_' + weights_and_biases_run_name

    logging.info(f'Device set to {device}!')

    ### Preparing data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
       # strip_accents=False
    )

    dataset_obj = None

    finetuned_model_type = args.finetuned_model_type if args.finetuned_model_type is not None else config['run']['finetuned_model_type']

    if finetuned_model_type == config['app']['model_classification_name']:
        dataset_obj = ClassificationDataset(tokenizer)
    elif finetuned_model_type == config['app']['model_tagging_name']:
        dataset_obj = TokenClassificationDataset(tokenizer)
    else:
        raise NotImplementedError(f'''Type {finetuned_model_type} not supported yet!''')

    dataset_train_split_key = config['app']['dataset_train_key'] if dataset_config.get('dataset_train_key', None) is None else dataset_config.get('dataset_train_key')
    dataset_val_split_key = config['app']['dataset_val_key'] if dataset_config.get('dataset_val_key', None) is None else dataset_config.get('dataset_val_key')
    dataset_test_split_key = config['app']['dataset_test_key'] if dataset_config.get('dataset_test_key', None) is None else dataset_config.get('dataset_test_key')

    dataset_obj.set_splits_names(dataset_train_split_key, dataset_val_split_key, dataset_test_split_key)

    if dataset_config['load_from_hub']:
        dataset_name = dataset_config['hub_dataset_name']
        #dataset_name = parameters['dataset_from_datasets_hub_name']

        dataset_obj.load_hosted_dataset(dataset_name, revision=dataset_config['revision'])
    else:
        data_files = {
            dataset_train_split_key: [dataset_config['train_file_path']],
            dataset_val_split_key: [dataset_config['val_file_path']],
            dataset_test_split_key: [dataset_config['test_file_path']]
        }

        dataset_obj.load_csv_dataset(
            data_files,
            delimiter=dataset_config['delimiter']
        )

    dataset_obj.truncate_dataset(
        dataset_train_split_key,
        config['dataset']['train_dataset_size'],
        shuffle=config['dataset']['shuffle_datasets']
    )
    dataset_obj.truncate_dataset(
        dataset_val_split_key,
        config['dataset']['val_dataset_size'],
        shuffle=config['dataset']['shuffle_datasets']
    )
    dataset_obj.truncate_dataset(
        dataset_test_split_key,
        config['dataset']['val_dataset_size'],
        shuffle=config['dataset']['shuffle_datasets']
    )

    dataset_obj.prepare_dataset(
        dataset_config['label_column_name'],
        dataset_config['text_column_name'],
    )




    ### Prepare W&B structure for saving predictions
    wandb_table = None
    if config['run']['weights_and_biases_on'] and \
        config['reporting']['weights_and_biases_save_predictions']:

        categories = sorted(list(dataset_obj.get_all_categories().items()), key=lambda key: key[1])
        categories_names = [_tuple[0] for _tuple in categories]
        wandb_table_columns = config['reporting']['weights_and_biases_eval_table_columns'] + categories_names
        wandb_table = wandb.Table(columns=wandb_table_columns)

    logging.info(f'Categories: {dataset_obj.get_all_categories()}')
    num_labels = dataset_obj.get_num_categories()

    model = Model(
        pretrained_model_name,
        model_type=finetuned_model_type,
        num_labels=num_labels
    )

    run = None

    this_run_folder_path = LOCAL_RUNS_FOLDER_PATH / weights_and_biases_run_name
    this_run_folder_path.mkdir(exist_ok=True, parents=True)

    if config['run']['weights_and_biases_on']:
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
        wandb_on=config['run']['weights_and_biases_on'],
        imbalanced_training=config['run']['class_imbalance_reweight'],
        model_type=finetuned_model_type,
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
        lr=learning_rate
    )
    trainer.set_optimizer(optimizer, init_lr = learning_rate)

    num_training_steps = train_epochs * trainer.get_training_steps_num()

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
    if finetuned_model_type == config['app']['model_tagging_name']:
        metrics_list = config['model']['metrics']['tagging']
    elif finetuned_model_type == config['app']['model_classification_name']:
        metrics_list = config['model']['metrics']['classification']
    else:
        raise NotImplementedError(f'''There is no such model type implemented like {finetuned_model_type}''')

    for metric in metrics_list:
        trainer.add_evaluation_metric(load_metric(metric))

    if config['run']['visualise_locally']:
        visualisation = TrainingVisualisation()

    if config['run']['full_train']:
        trainer.full_train(
            train_epochs=train_epochs,
            train_batch_size=config['model']['train_batch_size'],
            val_batch_size=config['model']['val_batch_size'],
            test_batch_size=config['model']['test_batch_size'],
            debug=config['app']['debug_mode'],
            save_model_path=str(this_run_folder_path / 'dev_models')
        )

        full_training_metrics = trainer.full_training_metrics
        if config['run']['visualise_locally']:
            visualisation.add_full_training_metrics(full_training_metrics)

    strategies = config['run']['strategies']
    if args.al_strategy is not None:
        strategies = [args.al_strategy]

    implemented_strategies_list = list(config['app']['strategies'].keys())
    for strategy in strategies:
        if strategy not in implemented_strategies_list:
            logging.warning(f'Strategy {strategy} not found in implemented strategies list: {implemented_strategies_list}. Skipping.')
            continue

        trainer.al_train(
            al_iterations=config['al']['num_iterations'],
            init_dataset_size=init_dataset_size,
            add_dataset_size=add_dataset_size,#config['al']['add_dataset_size'],
            train_epochs=train_epochs,
            strategy=strategy,
            train_batch_size=config['model']['train_batch_size'],
            val_batch_size=config['model']['val_batch_size'],
            test_batch_size=config['model']['test_batch_size'],
            debug=config['app']['debug_mode'],
            save_model_path=str(this_run_folder_path / 'dev_models'),
        )

        al_strategy_metrics = trainer.al_strategy_metrics
        # Letadlo2022+2022

        if config['run']['visualise_locally']:
            visualisation.add_al_strategy_metrics(al_strategy_metrics, strategy)

    if config['run']['visualise_locally']:
        visualisation.visualise(save_fig_path=config['run']['visualisation_save_path'])

