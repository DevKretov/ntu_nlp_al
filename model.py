import os
import numpy as np
import pandas as pd
import argparse
import pickle
# import seqeval
import time
import csv
from scipy.special import softmax

# from seqeval.metrics import accuracy_score
# from seqeval.metrics import classification_report
# from seqeval.metrics import f1_score

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
    def __init__(self, model_name, model_type, config=None, num_labels = None):
        if not isinstance(config, PretrainedConfig):
            config = None

        #self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        self.model_name = model_name
        self.config = config
        self.model_type = model_type

        self.reinit_model(model_name, config, num_labels)

    def reinit_model(self, model_name = None, config = None, num_labels = None):
        if model_name is None:
            model_name = self.model_name

        if config is None:
            config = self.config
        elif not isinstance(config, PretrainedConfig):
            config = None

        if self.model_type == self.CLASSIFICATION_MODEL_TYPE:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                num_labels=num_labels
            )
        elif self.model_type == self.TAGGING_MODEL_TYPE:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                config=config,
                num_labels=num_labels
            )


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
