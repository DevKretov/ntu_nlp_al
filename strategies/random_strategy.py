import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s', level=logging.DEBUG)

import numpy as np
from ._strategy import _Strategy

class RandomStrategy(_Strategy):

    def __init__(self, model, dataloader, dataset_len, device, model_type):
        super().__init__(model, dataloader, dataset_len, device, model_type)

        self.name = 'random'

        self.strategy_log_folder_file = self.log_output_folder_path / self.name
        self.strategy_log_folder_file.mkdir(exist_ok=True)

        logging.info('AL Random strategy applied!')

    def query(self, n):

        logging.info(f'Selecting {n} indices from {self.unlabelled_dataset_length} ')

        random_choice = np.random.choice(
            self.unlabelled_dataset_length,
            n,
            replace=False
        ).tolist()

        return random_choice
