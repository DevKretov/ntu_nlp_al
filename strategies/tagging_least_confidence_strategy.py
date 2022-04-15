import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

import numpy as np
import torch
import json

from _strategy import _Strategy

class TaggingLeastConfidence(_Strategy):
    def __init__(self, model, dataloader, dataset_len, device, model_type, threshold = 1.0):
        super().__init__(model, dataloader, dataset_len, device, model_type)

        self.name = 'least_confidence'
        self.query_i = 0
        self.threshold = threshold

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)
        logging.info('AL Least confidence strategy applied!')

    def query(self, n):

        logging.info(f'Selecting {n} indices from {self.unlabelled_dataset_length} ')
        self.create_logits()

       # assert self.logits is not None and self.probs is not None, 'Cannot process until these variables are initialized'
        if self.model_type == 'tagging':
            log_seqs_probs = []
            max_probs = [_probs.max(dim=-1) for _probs in self.probs]

            for batch_i in range(len(max_probs)):
                batch = max_probs[batch_i].values
                for seq_i in range(len(batch)):
                    max_values = batch[seq_i][self.all_masks[batch_i][seq_i]].data.cpu().numpy()
                    log_max_values = np.log(max_values)
                    seq_len = len(log_max_values)
                    log_seq_prob = np.sum(log_max_values) / seq_len

                    log_seqs_probs.append(log_seq_prob)

            max_probs = torch.tensor(log_seqs_probs)
        else:
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



