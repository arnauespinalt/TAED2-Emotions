# -*- coding: utf-8 -*-
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score
import sklearn.metrics as metrics



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def mlflow_metrics(y_val, y_pred):
    f1 = f1_score(y_val, y_pred, average="weighted")
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred, average="weighted")

<<<<<<< HEAD
    return {"accuracy": acc, 
            "f1": f1, 
            "recall": rec}
=======
    fpr, tpr, threshold = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    return {"accuracy": acc, 
            "f1": f1, 
            "recall": rec, 
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr}
>>>>>>> 4bb5b3c41e84230055b293d6eb6557b4f90d97b5

def predict_model(model, dataset):

    with mlflow.start_run():
        logger.info("Executing predictions.")
<<<<<<< HEAD
        print(dataset['validation'][0])
=======

>>>>>>> 4bb5b3c41e84230055b293d6eb6557b4f90d97b5
        preds_output = model.predict(dataset["validation"])
        preds_output.metrics

        y_valid = np.array(dataset["validation"]["label"])
        y_preds = np.argmax(preds_output.predictions, axis=1)

        metrics = mlflow_metrics(y_valid, y_preds)

        mlflow.log_metric("accuracy_score",  metrics['accuracy'])
        mlflow.log_metric("f1_score",  metrics['f1'])
        mlflow.log_metric("recall", metrics['recall'])

    return metrics





