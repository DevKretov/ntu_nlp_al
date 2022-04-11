import torch
from transformers import get_scheduler
from datasets import list_metrics, load_metric

from active_learning_trainer import ALTrainer
from transformers import AutoTokenizer
from dataset import Dataset, TokenClassificationDataset
from model import Model


import datetime
import wandb


if __name__ == '__main__':

    parameters = dict()
    parameters['use_gpu'] = True

    parameters['weights_and_biases_on'] = True
    parameters['weights_and_biases_key'] = '5e5e00356042a33b5cb271399b8d05c9c9d6ded8'
    # TODO: run name based on timestamp
    current_timestamp = str(datetime.datetime.now()).split('.')[0]

    # TODO: implement it
    parameters['weights_and_biases_save_predictions'] = False

    parameters['pretrained_model_name'] = 'prajjwal1/bert-small' #'distilbert-base-uncased'


    # parameters['train_dataset_file_path'] = 'data/imdb/train_IMDB.csv'
    # parameters['val_dataset_file_path'] = 'data/imdb/test_IMDB.csv'
    # parameters['test_dataset_file_path'] = 'data/imdb/test_IMDB.csv'

    parameters['dataset_from_datasets_hub'] = False
    parameters['dataset_from_datasets_hub_name'] = 'conll2003'
    parameters['train_dataset_file_path'] = 'data/news/train.csv'
    parameters['val_dataset_file_path'] = 'data/news/val.csv'
    parameters['test_dataset_file_path'] = 'data/news/test.csv'
    parameters['dataset_file_delimiter'] = ','

    parameters['dataset_text_column_name'] =  'tokens' #'text'
    parameters['dataset_label_column_name'] = 'ner_tags'#'airline_sentiment'

    parameters['dataset_text_column_name'] = 'text_cleaned'  # 'text'
    parameters['dataset_label_column_name'] = 'label_reduced'  # 'airline_sentiment'

    # TODO: implement this with CrossEntropyLoss
    parameters['loss'] = 'cross_entropy'
    parameters['loss_weighted'] = False

    parameters['class_imbalance_reweight'] = True
    parameters['train_batch_size'] = 32
    parameters['val_batch_size'] = 64
    parameters['test_batch_size'] = 64
    parameters['epochs'] = 5
   # parameters['finetuned_model_type'] = 'classification'
    parameters['finetuned_model_type'] = 'classification'
    model_type = parameters['finetuned_model_type']


    parameters['al_iterations'] = 100
    parameters['init_dataset_size'] = 32
    parameters['add_dataset_size'] = 32
    parameters['al_strategy'] = 'random' #'least_confidence'
    parameters['full_train'] = False

    parameters['debug'] = False



    device = 'cpu'
    if parameters['use_gpu']:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parameters['weights_and_biases_run_name'] = f'{current_timestamp}_run_{model_type}_{device}'
    print(f'Device set to {device}!')



    tokenizer = AutoTokenizer.from_pretrained(parameters['pretrained_model_name'])

    dataset_obj = None
    if parameters['finetuned_model_type'] == 'classification':
        dataset_obj = Dataset(tokenizer)
    elif parameters['finetuned_model_type'] == 'tagging':
        dataset_obj = TokenClassificationDataset(tokenizer)
    else:
        model_type = parameters['finetuned_model_type']
        raise NotImplementedError(f'Type {model_type} not supported yet!')



    if parameters['dataset_from_datasets_hub']:
        dataset_name = parameters['dataset_from_datasets_hub_name']
        dataset_obj.load_hosted_dataset(dataset_name)
    else:
        data_files = {
            'train': [parameters['train_dataset_file_path']],
            'val': [parameters['val_dataset_file_path']],
            'test': [parameters['test_dataset_file_path']]
        }

        dataset_obj.load_csv_dataset(
            data_files,
            delimiter=parameters['dataset_file_delimiter']
        )

    dataset_obj.truncate_dataset('train', 10000)
    dataset_obj.truncate_dataset('val', 1000)
    dataset_obj.truncate_dataset('test', 10000)

    dataset_obj.prepare_dataset(parameters['dataset_label_column_name'], parameters['dataset_text_column_name'], )
    # dataset_obj.prepare_labels(parameters['dataset_label_column_name'])
    # dataset_obj.encode_dataset(parameters['dataset_text_column_name'])

    print(f'Categories: {dataset_obj.get_all_categories()}')
    num_labels = dataset_obj.get_num_categories()

    model = Model(
        parameters['pretrained_model_name'],
        model_type=parameters['finetuned_model_type'],
        num_labels=num_labels
    )


    if parameters['weights_and_biases_on']:
        wandb.login(key='5e5e00356042a33b5cb271399b8d05c9c9d6ded8')
        wandb.init(
            name=parameters['weights_and_biases_run_name'],
            project='ntu_al',
            reinit=True
        )

        wandb.config.update(parameters)
        wandb.watch(model.model)

    trainer = ALTrainer(
        wandb_on=parameters['weights_and_biases_on'],
        imbalanced_training=parameters['class_imbalance_reweight'],
        model_type=parameters['finetuned_model_type']
    )
    trainer.set_model(model)

    # TODO: add strategy
    trainer.set_strategy(None)
    trainer.set_dataset(dataset_obj)
    dataset_obj.prepare_dataloaders(
        train_batch_size=parameters['train_batch_size'],
        val_batch_size=parameters['val_batch_size'],
        test_batch_size=parameters['test_batch_size'],
    )

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=5e-5)
    trainer.set_optimizer(optimizer)

    num_training_steps = parameters['epochs'] * trainer.get_training_steps_num()

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    trainer.set_lr_scheduler(lr_scheduler)
    trainer.set_device(device)

    trainer.add_evaluation_metric(load_metric('accuracy'))

    if parameters['finetuned_model_type'] == 'tagging':
        trainer.add_evaluation_metric(load_metric('seqeval'))
    else:
        trainer.add_evaluation_metric(load_metric('f1'))
        trainer.add_evaluation_metric(load_metric('precision'))
        trainer.add_evaluation_metric(load_metric('recall'))

    if parameters['full_train']:
        trainer.full_train(
            train_epochs=parameters['epochs'],
            train_batch_size=parameters['train_batch_size'],
            val_batch_size=parameters['val_batch_size'],
            test_batch_size=parameters['test_batch_size'],
            debug=parameters['debug']
        )

    trainer.al_train(
        al_iterations=parameters['al_iterations'],
        init_dataset_size=parameters['init_dataset_size'],
        add_dataset_size=parameters['add_dataset_size'],
        train_epochs=parameters['epochs'],
        strategy=parameters['al_strategy'],
        train_batch_size=parameters['train_batch_size'],
        val_batch_size=parameters['val_batch_size'],
        test_batch_size=parameters['test_batch_size'],
        debug=parameters['debug']
    )
   # trainer.train_model(parameters['epochs'])

    #al_strategy = RandomStrategy(model, )
