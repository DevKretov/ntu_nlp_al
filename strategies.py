from transformers import DataCollatorWithPadding

import tensorflow as tf
import numpy as np
from scipy.stats import entropy
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb
import torch
import torch.nn.functional as F
import tqdm
import sys
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from transformers import BertForSequenceClassification

from datasets import concatenate_datasets
from datasets import load_dataset

from pathlib import Path
import json

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
            num_labels = 0

    ):

        self.model = model
        self.unlabelled_dataset_dataloader = unlabelled_dataset_dataloader
        self.unlabelled_dataset_length = unlabelled_dataset_length
        self.device = device

        self.log_output_folder = 'al_strategies_log'
        self.log_output_folder_path = Path(self.log_output_folder)

        self.log_output_folder_path.mkdir(exist_ok=True)

        self.training_dict_keys = ['attention_mask', 'input_ids', 'labels', 'token_type_ids']

    def create_logits(self):
        model = self.model.model
        print(f'Is model on CUDA - {model.is_cuda()}')
        model.eval()

        num_batches = len(self.unlabelled_dataset_dataloader)
        logits_all = None
        print('AL evaluation iteration')
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
        all_labels = None
        for next_batch in self.unlabelled_dataset_dataloader:
            next_batch = {your_key: next_batch[your_key] for your_key in self.training_dict_keys}
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}
            if all_labels is None:
                all_labels = next_batch['labels']
            else:
                all_labels = torch.cat((all_labels, next_batch['labels']), dim=0)
            with torch.no_grad():
                outputs = model(**next_batch)
            loss = outputs.loss

            logits = outputs.logits
            if logits_all is None:
                logits_all = logits.cpu().data
            else:
                logits_all = torch.cat((logits_all, logits.cpu().data))

            #predictions = torch.argmax(logits, dim=-1)
            batch_i += 1
            pbar.update(1)
            pbar.set_description(f'AL evaluation iteration. Batch {batch_i:5}/{num_batches}')



        self.logits = logits_all
        self.probs = F.softmax(self.logits, dim=1)
        pass

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




class BadgeSampling(_Strategy):
    def __init__(self, model, dataloader, dataset_len, device, num_labels, embedding_dim, batch_size):
        super().__init__(model, dataloader, dataset_len, device)


        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.query_i = 0
        self.name = 'badge'
        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)

        print('AL BADGE strategy applied!')

    def update_dataloader(self, new_dataloader):
        self.unlabelled_dataset_dataloader = new_dataloader

    def update_dataset_len(self, new_dataset_len):
        self.unlabelled_dataset_length = new_dataset_len

    def init_centers(self, X, K):
        distances = []
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        distances.append(0)
        centInds = [0.] * len(X)
        cent = 0
       # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
           # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            distances.append(D2[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll, distances

    def query(self, n):
       # idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding().numpy()
        chosen, distances = self.init_centers(gradEmbedding, n)

        self.query_i += 1
        self.query_i_name = f'{self.query_i}.json'
        json.dump(
            distances,
            open(f'{str(self.strategy_log_folder_file / self.query_i_name)}', 'w'),
            sort_keys=True,
            indent=4
        )

        return chosen

    def get_grad_embedding(self):
        embDim = self.embedding_dim
        nLab = self.num_labels

        num_batches = len(self.unlabelled_dataset_dataloader)

        embedding = np.zeros(
            [
                self.unlabelled_dataset_length,
                embDim * nLab
            ]
        )

        model = self.model.model
        #model = model.to(self.device)
        print(f'Is model on CUDA - {model.is_cuda()}')

        model.eval()

        logits_all = None
        print('AL evaluation iteration')
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
        all_labels = None


        for next_batch in self.unlabelled_dataset_dataloader:
            next_batch = {your_key: next_batch[your_key] for your_key in self.training_dict_keys}
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}

            y = next_batch['labels']
            #idxs = np.arange(batch_i * self.batch_size, (batch_i + 1) * self.batch_size)
            idxs = np.arange(
                batch_i * self.batch_size,
                min(
                    (batch_i + 1) * self.batch_size,
                    self.unlabelled_dataset_length
                )
            )

            with torch.no_grad():
                outputs = model(**next_batch, output_hidden_states=True)
              #  outputs = model(**next_batch)
                # TODO: change pooler and try to take unpooled last hidden state (take the first vector or MEAN of those with attention mask equal 1)
                #hidden_states_pooled = model.bert.pooler(outputs.hidden_states[-1])
                hidden_states_pooled = outputs.hidden_states[-1][:, 0].data.cpu().numpy()

            loss = outputs.loss
            logits = outputs.logits
            #out_np = np.mean(outputs[2][-1].data.cpu().numpy(), axis=1)
            out_np = hidden_states_pooled
            out = outputs[0]

           # hidden_states = outputs.hidden_states[-1]#.shape

            batchProbs = F.softmax(logits, dim=1).data.cpu().numpy()
            maxInds = np.argmax(batchProbs, 1)

            for j in range(len(y)):
                for c in range(nLab):
                    if c == maxInds[j]:
                        embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out_np[j]) * (1 - batchProbs[j][c])
                    else:
                        embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out_np[j]) * (-1 * batchProbs[j][c])

            # predictions = torch.argmax(logits, dim=-1)
            batch_i += 1
            pbar.update(1)
            pbar.set_description(f'AL evaluation iteration. Batch {batch_i:5}/{num_batches}')

        return torch.Tensor(embedding)


class KMeansSampling(_Strategy):

    def __init__(self, model, dataloader, dataset_len, device, num_labels, embedding_dim, batch_size):
        super().__init__(model, dataloader, dataset_len, device)

        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.query_i = 0
        self.name = 'k_means'

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)
        print('AL K Means strategy applied!')

    def update_dataloader(self, new_dataloader):
        self.unlabelled_dataset_dataloader = new_dataloader

    def update_dataset_len(self, new_dataset_len):
        self.unlabelled_dataset_length = new_dataset_len

    def query(self, n):

        embedding = self.get_embedding()
        embedding = embedding#.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embedding)

        cluster_idxs = cluster_learner.predict(embedding)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embedding - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = np.array(
            [np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])

        return q_idxs

    def get_embedding(self):

        embedding = np.zeros([self.unlabelled_dataset_length, self.embedding_dim])

        batch_i = 0

        model = self.model.model
        model.eval()

        num_batches = len(self.unlabelled_dataset_dataloader)
        print('AL evaluation iteration')
        pbar = tqdm.trange(
            num_batches,
            desc="AL evaluation iteration",
            smoothing=0.05,
            disable=False,
            position=0,
            leave=True,
            file=sys.stdout
        )

        for next_batch in self.unlabelled_dataset_dataloader:
            next_batch = {your_key: next_batch[your_key] for your_key in self.training_dict_keys}
            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}

            y = next_batch['labels']
            idxs = np.arange(
                batch_i * self.batch_size,
                min(
                    (batch_i + 1) * self.batch_size,
                    self.unlabelled_dataset_length
                )
            )

            with torch.no_grad():
                outputs = model(**next_batch, output_hidden_states=True)
              #  outputs = model(**next_batch)
            loss = outputs.loss
            logits = outputs.logits

            #out_np = np.mean(outputs[2][-1].data.cpu().numpy(), axis=1)
            out_np = hidden_states_pooled = outputs.hidden_states[-1][:, 0].data.cpu().numpy()

            embedding[idxs] = out_np

            batch_i += 1
            pbar.update(1)
            pbar.set_description(f'AL evaluation iteration. Batch {batch_i:5}/{num_batches}')

        return embedding

    pass

class EntropySampling(_Strategy):
    def __init__(self, model, dataloader, dataset_len, device):
        super().__init__(model, dataloader, dataset_len, device)

        self.name = 'entropy_sampling'

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)

        print('AL Entropy sampling strategy applied!')

    def update_dataloader(self, new_dataloader):
        self.unlabelled_dataset_dataloader = new_dataloader

    def update_dataset_len(self, new_dataset_len):
        self.unlabelled_dataset_length = new_dataset_len

    def query(self, n):

        print(f'Selecting {n} indices from {self.unlabelled_dataset_length} ')
        self.create_logits()

        assert self.logits is not None and self.probs is not None, 'Cannot process until these variables are initialized'

        #max_probs = self.probs.max(dim=1)[0]
        log_probs = torch.log(self.probs)
        U = (self.probs * log_probs).sum(1)

        indices = U.sort()[1][:n]

        self.query_i += 1
        self.query_i_name = f'{self.query_i}.json'
        json.dump(
            U.sort()[0][:n].data.cpu().numpy().tolist(),
            open(f'{str(self.strategy_log_folder_file / self.query_i_name)}', 'w'),
            sort_keys=True,
            indent=4
        )

        return indices


class LeastConfidence(_Strategy):
    def __init__(self, model, dataloader, dataset_len, device, threshold = 1.0):
        super().__init__(model, dataloader, dataset_len, device)

        self.name = 'least_confidence'
        self.query_i = 0
        self.threshold = threshold

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)
        print('AL Least confidence strategy applied!')

    def update_dataloader(self, new_dataloader):
        self.unlabelled_dataset_dataloader = new_dataloader

    def update_dataset_len(self, new_dataset_len):
        self.unlabelled_dataset_length = new_dataset_len

    def query(self, n):

        print(f'Selecting {n} indices from {self.unlabelled_dataset_length} ')
        self.create_logits()

        assert self.logits is not None and self.probs is not None, 'Cannot process until these variables are initialized'

        max_probs = self.probs.max(dim=1)[0]
        max_probs = max_probs[max_probs < self.threshold]
        n = min(len(max_probs), n)

        indices = max_probs.sort()[1][:n]

        self.query_i += 1
        self.query_i_name = f'{self.query_i}.json'
        json.dump(
            max_probs[indices].data.cpu().numpy().tolist(),
            open(f'{str(self.strategy_log_folder_file / self.query_i_name)}', 'w'),
            sort_keys=True,
            indent=4
        )

        return indices



class RandomStrategy(_Strategy):

    def __init__(self, model, dataloader, dataset_len, device):
        super().__init__(model, dataloader, dataset_len, device)

        self.name = 'random'

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)

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
