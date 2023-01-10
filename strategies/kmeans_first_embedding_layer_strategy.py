import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

import numpy as np
import torch
import tqdm
import sys
from sklearn.cluster import KMeans

from ._strategy import _Strategy

class KMeansFirstLayerSampling(_Strategy):

    def __init__(self, model, dataloader, dataset_len, device, num_labels, embedding_dim, batch_size, model_type):
        super().__init__(model, dataloader, dataset_len, device, model_type)

        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.query_i = 0
        self.name = 'k_means'

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)
        logging.info('AL KMeansFirstLayerSampling strategy applied!')

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

            out_np = outputs.hidden_states[1][:, 0].data.cpu().numpy() # the only difference here

            embedding[idxs] = out_np

            batch_i += 1
            pbar.update(1)
            pbar.set_description(f'AL evaluation iteration. Batch {batch_i:5}/{num_batches}')

        return embedding

    pass
