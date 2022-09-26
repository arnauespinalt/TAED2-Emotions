# -*- coding: utf-8 -*-
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from urllib.parse import urlparse
from datasets import load_dataset 
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import plot_confusion_matrix

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#Loading dataset emotion
emotions = load_dataset("emotion")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    rec = recall_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "recall": rec}

def mlflow_metrics(y_val, y_pred):
    f1 = f1_score(y_val, y_pred, average="weighted")
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred, average="weighted")
    return {"accuracy": acc, "f1": f1, "recall": rec}

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))

emotions_encoded["train"].features
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_encoded["train"].features

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
training_args = TrainingArguments(output_dir="results",
                                  num_train_epochs=8,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  load_best_model_at_end=False,
                                  metric_for_best_model="f1",
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  save_strategy="no",
                                  disable_tqdm=False)


trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"])

with mlflow.start_run():

    #trainer.train()

    #results = trainer.evaluate()

    preds_output = trainer.predict(emotions_encoded["validation"])
    preds_output.metrics

    y_valid = np.array(emotions_encoded["validation"]["label"])
    y_preds = np.argmax(preds_output.predictions, axis=1)

    metrics = mlflow_metrics(y_valid, y_preds)

    mlflow.log_metric("accuracy_score",  metrics['accuracy'])
    mlflow.log_metric("f1_score",  metrics['f1'])
    mlflow.log_metric("recall", metrics['recall'])

    #labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    #plot_confusion_matrix(y_preds, y_valid, labels)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="Distil-Bert-Uncased-Emotions")
    else:
            mlflow.sklearn.log_model(model, "model")



