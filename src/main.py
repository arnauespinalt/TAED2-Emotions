
from src.data.make_dataset import load
from src.models.train_model import train_model
from src.models.predict_model import predict_model
from src.visualization.visualize import plots
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def ETL():
    # Load and transform the dataset.
    dataset = load("emotion")

    # Load the pre-trained model and fine-tune it with the dataset to do a classification task.
    model, results = train_model(dataset)

    # Execute some predictions with the new fine-tuned model.
    metrics = predict_model(model, dataset)
    logger.info(f"The accuracy of the model is {metrics['accuracy']}")
    print(metrics)

    # Do some visualization for the metrics.
    plots(metrics)