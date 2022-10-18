# -*- coding: utf-8 -*-
import logging
from datasets import load_dataset 
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def load(dataset_name: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    logger.info("Loading Emotion dataset.")
    emotions = load_dataset(dataset_name)
    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    logger.info("Formatting dataset.")
    emotions_encoded["train"].features
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    emotions_encoded["train"].features

    return emotions_encoded
