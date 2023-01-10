import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

import numpy as np

from scipy import stats

import pdb
import torch
import torch.nn.functional as F
import tqdm
import sys
from sklearn.metrics import pairwise_distances
import json

from copy import deepcopy
from ._strategy import _Strategy

class BadgeSampling(_Strategy):
    def __init__(self, model, dataloader, dataset_len, device, num_labels, embedding_dim, batch_size, model_type):
        super().__init__(model, dataloader, dataset_len, device, model_type)


        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.query_i = 0
        self.name = 'badge'
        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)

        logging.info('AL BADGE strategy applied!')

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
        logging.debug(f'Is model on CUDA - {model.device}')

        model.eval()

        logits_all = None
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
        for next_batch in self.unlabelled_dataset_dataloader:
            if self.model_type == 'tagging':
                next_batch, next_batch_metadata = next_batch

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
                hidden_states_pooled = outputs.hidden_states[-1][:, 0].data.cpu().numpy()

            logits = outputs.logits
            out_np = hidden_states_pooled

            batchProbs = F.softmax(logits, dim=1).data.cpu().numpy()
            maxInds = np.argmax(batchProbs, 1)

            for j in range(len(y)):
                for c in range(nLab):
                    if c == maxInds[j]:
                        embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out_np[j]) * (1 - batchProbs[j][c])
                    else:
                        embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out_np[j]) * (-1 * batchProbs[j][c])

            batch_i += 1
            pbar.update(1)
            pbar.set_description(f'AL evaluation iteration. Batch {batch_i:5}/{num_batches}')

        return torch.Tensor(embedding)
