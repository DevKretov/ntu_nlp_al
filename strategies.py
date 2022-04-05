from transformers import DataCollatorWithPadding

import tensorflow as tf
import numpy as np
from scipy.stats import entropy
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb
import torch
import tqdm


from datasets import concatenate_datasets
from datasets import load_dataset

from copy import deepcopy

from transformers import BertTokenizerFast
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import BertConfig
from dataset import Dataset
from model import Model

from transformers import BertConfig



config = BertConfig.from_pretrained('bert-base-multilingual-uncased', num_labels=12)

class _Strategy:

    def __init__(
            self,
            model,
            unlabelled_dataset_dataloader,
            unlabelled_dataset_length,
            device,

    ):

        self.model = model
        self.unlabelled_dataset_dataloader = unlabelled_dataset_dataloader
        self.unlabelled_dataset_length = unlabelled_dataset_length
        self.device = device


    def create_logits(self):
        model = self.model.model
        model.eval()

        num_batches = len(self.unlabelled_dataset_dataloader)
        logits_all = []
        pbar = tqdm.trange(
            num_batches,
            desc="AL evaluation iteration",
            smoothing=0.05,
            disable=False,
            position=0,
            leave=True
        )

        batch_i = 0
        for next_batch in self.unlabelled_dataset_dataloader:
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}
            with torch.no_grad():
                outputs = model(**next_batch)
            loss = outputs.loss

            logits = outputs.logits
            logits_all = torch.cat(logits_all, logits)

            #predictions = torch.argmax(logits, dim=-1)
            batch_i += 1
            pbar.update(1)
            pbar.set_description(f'Batch {batch_i:5}/{num_batches}')

        self.logits = logits_all

    def query(self, n):
        '''
        Query the indices of unlabelled data for adding to AL dataset
        :param n:
        :return:
        '''

        raise NotImplementedError
        # ## TODO: max range by logits
        # random_choice = np.random.choice(
        #     range(0, len(self.dataset_obj.al_train_dataset['unlabelled'])),
        #     n,
        #     replace=False
        # ).tolist()
        #
        # return random_choice


#class MaximumEntropyStrategy(_Strategy):


class RandomStrategy(_Strategy):

    def __init__(self, model, dataloader, dataset_len, device):
        super().__init__(model, dataloader, dataset_len, device)

        print('AL Random strategy applied!')

    def update_dataloader(self, new_dataloader):
        self.unlabelled_dataset_dataloader = new_dataloader

    def update_dataset_len(self, new_dataset_len):
        self.unlabelled_dataset_length = new_dataset_len

    def query(self, n):

        #self.create_logits()
        print(f'Selecting {n} indices from {self.unlabelled_dataset_length} ')
        ## TODO: max range by logits
        random_choice = np.random.choice(
            self.unlabelled_dataset_length,
            n,
            replace=False
        ).tolist()

        return random_choice

class Strategy:

    def __init__(self, model, dataset_obj):
        assert isinstance(dataset_obj, Dataset)
        assert isinstance(model, Model)

        self.model = model
        self.dataset_obj = dataset_obj


    def select_new_unlabelled_batch(self, num_examples_to_select = 20):
        logits_all_tf = self.model.model.predict(
            self.dataset_obj.al_train_dataset_tf['unlabelled'],
            verbose=1
        )['logits']

        softmax_layer = tf.keras.layers.Softmax()

        def _calculate_entropy(logits):
            return entropy(softmax_layer(logits).numpy(), axis=1)

        entropy_all = _calculate_entropy(logits_all_tf)
        top_n_indices = np.argpartition(
            entropy_all,
            -num_examples_to_select
        )[-num_examples_to_select:]


        return top_n_indices

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



    def run_strategy(
            self,
            num_runs = 5,
            initial_al_train_dataset_size = 100,
            val_dataset_batch_size = 16,
            test_dataset_batch_size = 16,
            al_dataset_train_tf_batch_size = 16,
            num_examples_to_add_to_batch = 32
    ):

        # All model handling procedures: in Model class
        # All strategy-related things - here
        # ALL DATASETS OPERATIONS: here

        # All visualisations: in utils

        self.dataset_obj.prepare_al_datasets(initial_al_train_dataset_size)
        self.dataset_obj.prepare_tf_datasets(test_dataset_batch_size, val_dataset_batch_size)

        for al_iteration_i in range(num_runs):
            self.model.reinit_model()
            self.model.compile_model()

            self.dataset_obj.prepare_al_datasets_tf(al_dataset_train_tf_batch_size)

            print(f'Model training...')

            self.model.model.fit(
                self.dataset_obj.al_train_dataset_tf['train'],
                steps_per_epoch=2,
                validation_data=self.dataset_obj.tf_datasets['val'],
                validation_steps=10,
                epochs=1,
                verbose=1
            )

            print(f'Strategy...')
            selected_indices = self.select_new_unlabelled_batch(num_examples_to_add_to_batch)
           # selected_indices = self.query_badge(num_examples_to_add_to_batch)

            #selected_indices = self.query_random(num_examples_to_add_to_batch)
            self.update_al_datasets_with_new_batch(selected_indices)
            print(f'Datasets updated...')

        # TODO:
        # evaluate each run and see how the train and val accuracy develops over time in each experiment
        # grid of 2 graphs: train and val datasets accuracy,

        # TODO:
        # 



    def update_al_datasets_with_new_batch(self, indices_to_add):
        data_to_add = self.dataset_obj.al_train_dataset['unlabelled'].select(indices_to_add)

        self.dataset_obj.al_train_dataset['train'] = concatenate_datasets(
            [
                self.dataset_obj.al_train_dataset['train'],
                data_to_add
            ]
        )
        self.dataset_obj.al_train_dataset['unlabelled'] = self.dataset_obj.al_train_dataset['unlabelled'].filter(
            lambda example, indice: indice not in indices_to_add,
            with_indices=True
        )

        self.dataset_obj.al_train_dataset['unlabelled'] =  self.dataset_obj.al_train_dataset['unlabelled'].map(lambda ex, ind: {'dataset_index': ind}, with_indices=True)  # ['index_dataset']
        self.dataset_obj.al_train_dataset['train'] = self.dataset_obj.al_train_dataset['train'].map(
            lambda ex, ind: {'dataset_index': ind}, with_indices=True)  # ['index_dataset']

        self.dataset_obj.al_train_dataset_indices.append(
            data_to_add['index']
        )

    def get_grad_embedding(self, X, dataset_size):
        model = self.model
        embDim = self.model.config.hidden_size

        num_labels = self.dataset_obj.get_num_categories()
        embedding = np.zeros(
            [
                dataset_size,
                embDim * num_labels
            ]
        )

        softmax_layer = tf.keras.layers.Softmax()

        for _batch in X:
            out = self.model.model(_batch, output_hidden_states=True)#['logits']
            # Pooling last hidden state output.
            # TODO: take a look at last_hidden_state parameter of BERT model output.
            out_np = np.mean(out[1][-1].numpy(), axis=1)
            out = out[0]

            idxs = _batch[0]['dataset_index'].numpy()

            batchProbs = softmax_layer(out).numpy()
            maxInds = np.argmax(batchProbs, 1)

            # TODO: check batch size
            for j in range(len(maxInds)):
                for c in range(num_labels):
                    if c == maxInds[j]:
                        # TODO Add indices of entries of dataset into collator
                        embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out_np[j] * (1 - batchProbs[j][c]))
                    else:
                        embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out_np[j] * (-1 * batchProbs[j][c]))

        return embedding

    def init_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query_badge(self, n):
       # idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(
            self.dataset_obj.al_train_dataset_tf['unlabelled'],
            len(self.dataset_obj.al_train_dataset['unlabelled'])
        )
        chosen = self.init_centers(gradEmbedding, n),
        return chosen[0]

    # print(f'Train dataset: before: {len(al_train_dataset)}, after: {len(new_training_dataset)}')
    # print(f'Rest dataset: before: {len(rest_dataset)}, after: {len(new_rest_dataset)}')



    def query_random(self, n):

        random_choice = np.random.choice(
            range(0, len(self.dataset_obj.al_train_dataset['unlabelled'])),
            n,
            replace=False
        ).tolist()

        return random_choice

