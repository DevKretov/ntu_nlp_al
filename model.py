import os
import numpy as np
import pandas as pd
import argparse
import pickle
# import seqeval
import time
import csv
import torch
import torch.nn.functional as F
from scipy.special import softmax

# from seqeval.metrics import accuracy_score
# from seqeval.metrics import classification_report
# from seqeval.metrics import f1_score
from pathlib import Path
from sklearn.model_selection import train_test_split

import json

# from electra.model import tokenization

import subprocess
import tensorflow as tf
import transformers
from transformers import ElectraTokenizerFast, ElectraConfig, ElectraModel, TFElectraForSequenceClassification, ElectraForSequenceClassification


from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


from transformers import BertConfig, PretrainedConfig
from transformers import BertTokenizerFast

from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

class Model:

    CLASSIFICATION_MODEL_TYPE = 'classification'
    TAGGING_MODEL_TYPE = 'tagging'

    # Model type can be classification or tagging
    def __init__(self, model_name, model_type, config=None, num_labels = None, from_tf=False):
        if not isinstance(config, PretrainedConfig):
            config = None

        #self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        self.model_name = model_name
        self.config = config
        self.model_type = model_type
        self.num_labels = num_labels

        self.reinit_model(model_name, config, num_labels, from_tf)

    def reinit_model(self, model_name = None, config = None, num_labels = None, from_tf=False):
        if model_name is None:
            model_name = self.model_name

        if config is None:
            config = self.config
        elif not isinstance(config, PretrainedConfig):
            config = None

        if num_labels is None:
            num_labels = self.num_labels

        if self.model_type == self.CLASSIFICATION_MODEL_TYPE:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                from_tf=from_tf
             #   num_labels=num_labels
            )
        elif self.model_type == self.TAGGING_MODEL_TYPE:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                config=config,
                from_tf=from_tf
             #   num_labels=num_labels
            )


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def classify_text(self, input_text, max_length = 128, truncation=True):

        self.model.eval()
        next_batch = self.tokenizer(
            input_text,
            max_length=max_length,
            truncation=truncation,
            padding='max_length',
            return_tensors='pt'
        )

      #  next_batch = {k: v.to(self.device) for k, v in next_batch.items()}
        with torch.no_grad():
            outputs = self.model(**next_batch)
        loss = outputs.loss

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        probs = F.softmax(logits, dim=-1)
        probs = probs.data.cpu().numpy()

        pass

    def save_model(self, save_model_dir_path = ''):
        save_path = Path(save_model_dir_path)
        if not save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_model_dir_path)

    def compile_model(self, optimizer = None, loss = None, metrics = None):
        # TODO assert their classes to corresponding packages

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

        if loss is None:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        if metrics is None:
            metrics = tf.metrics.SparseCategoricalAccuracy()

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )


    def clean_gpu_memory(self, ):
        tf.keras.backend.clear_session()
