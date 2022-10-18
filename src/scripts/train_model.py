# -*- coding: utf-8 -*-
import mlflow
import mlflow.sklearn
import logging
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    rec = recall_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "recall": rec}


def train_model(dataset, train = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "bert-base-uncased"

    num_labels = 6

    logger.info(f"Loading {model_name} model.")
    model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))

    batch_size = 64
    logging_steps = len(dataset["train"]) // batch_size
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
        
        if train:
            trainer.train()
            results = trainer.evaluate()
        else:
            results = None

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

       # if tracking_url_type_store != "file":

        #         mlflow.sklearn.log_model(model, "model", registered_model_name="Distil-Bert-Uncased-Emotions")
        # else:
        #         mlflow.sklearn.log_model(model, "model")

        trainer.save_model('models/model')
      
        return trainer, results





