# -*- coding: utf-8 -*-
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from urllib.parse import urlparse
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

## NOT CURRENTLY WORKING 
with mlflow.start_run():

    preds_output = trainer.predict(emotions_encoded["validation"])
    preds_output.metrics

    y_valid = np.array(emotions_encoded["validation"]["label"])
    y_preds = np.argmax(preds_output.predictions, axis=1)

    metrics = mlflow_metrics(y_valid, y_preds)

    mlflow.log_metric("accuracy_score",  metrics['accuracy'])
    mlflow.log_metric("f1_score",  metrics['f1'])
    mlflow.log_metric("recall", metrics['recall'])