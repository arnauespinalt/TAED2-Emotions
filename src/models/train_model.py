'''
-*- coding: utf-8 -*-
'''
import logging
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import torch
from transformers import Trainer, TrainingArguments, \
    AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def compute_metrics(pred):
    '''Computes the following metrics: f1-score, accuracy and recall. '''
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f_score = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    rec = recall_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f_score, "recall": rec}


def train_model(dataset):
    '''
    Trains the model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "bert-base-uncased"

    logger.info(f"Loading {model_name} model.")

    model = (AutoModelForSequenceClassification.from_pretrained(model_name,
        num_labels=6).to(device))

    batch_size = 64
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
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["validation"])

    with mlflow.start_run():
        logger.info(f"Training {model_name} model with arguments: {training_args}.")
        #trainer.train()

        results = trainer.evaluate()

        return trainer, results

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model",
                registered_model_name="Distil-Bert-Uncased-Emotions")
        else:
            mlflow.sklearn.log_model(model, "model")
