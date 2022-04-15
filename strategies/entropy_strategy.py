import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

import torch
import json

from _strategy import _Strategy

class EntropySampling(_Strategy):
    def __init__(self, model, dataloader, dataset_len, device, model_type):
        super().__init__(model, dataloader, dataset_len, device, model_type)

        self.name = 'entropy_sampling'
        self.query_i = 0

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)

        logging.info('AL Entropy sampling strategy applied!')

    def query(self, n):

        logging.info(f'Selecting {n} indices from {self.unlabelled_dataset_length} ')
        self.create_logits()

        assert self.logits is not None and self.probs is not None, 'Cannot process until these variables are initialized'

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
