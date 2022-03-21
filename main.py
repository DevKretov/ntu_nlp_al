import torch
from transformers import get_scheduler
from datasets import list_metrics, load_metric

from active_learning_trainer import ALTrainer
from transformers import AutoTokenizer
from dataset import Dataset
from model import Model
from strategies import RandomStrategy

if __name__ == '__main__':

    parameters = dict()
    parameters['pretrained_model_name'] = 'prajjwal1/bert-tiny' #'distilbert-base-uncased'
    parameters['dataset_label_column_name'] = 'sentiment'
    parameters['dataset_text_column_name'] = 'review'

    parameters['train_dataset_file_path'] = 'data/imdb/train_IMDB.csv'
    parameters['val_dataset_file_path'] = 'data/imdb/test_IMDB.csv'
    parameters['test_dataset_file_path'] = 'data/imdb/test_IMDB.csv'

    parameters['train_batch_size'] = 4
    parameters['val_batch_size'] = 4
    parameters['test_batch_size'] = 4
    parameters['epochs'] = 5
    parameters['finetuned_model_type'] = 'classification'

    parameters['al_iterations'] = 5
    parameters['init_dataset_size'] = 100
    parameters['add_dataset_size'] = 100
    parameters['al_strategy'] = 'random'
    parameters['debug'] = True





    tokenizer = AutoTokenizer.from_pretrained(parameters['pretrained_model_name'])

    dataset_obj = Dataset(tokenizer)

    data_files = {
        'train': [parameters['train_dataset_file_path']],
        'val': [parameters['val_dataset_file_path']],
        'test': [parameters['test_dataset_file_path']]
    }

    dataset_obj.load_csv_dataset(
        data_files,
        delimiter=','
    )

    dataset_obj.truncate_dataset('train', 1000)
    dataset_obj.truncate_dataset('val', 1000)
    dataset_obj.truncate_dataset('test', 1000)

    dataset_obj.prepare_labels(parameters['dataset_label_column_name'])
    dataset_obj.encode_dataset(parameters['dataset_text_column_name'])

    print(f'Categories: {dataset_obj.get_all_categories()}')
    num_labels = dataset_obj.get_num_categories()

    model = Model(
        parameters['pretrained_model_name'],
        model_type=parameters['finetuned_model_type'],
        num_labels=num_labels
    )

    trainer = ALTrainer()
    trainer.set_model(model)

    # TODO: add strategy
    trainer.set_strategy(None)
    trainer.set_dataset(dataset_obj)
    trainer.prepare_dataloaders(
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
    trainer.determine_device()


    trainer.add_evaluation_metric(load_metric('accuracy'))
    trainer.add_evaluation_metric(load_metric('f1'))
    trainer.add_evaluation_metric(load_metric('precision'))
    trainer.add_evaluation_metric(load_metric('recall'))

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
