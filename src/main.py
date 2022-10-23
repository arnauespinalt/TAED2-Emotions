
from src.data.make_dataset import load
from src.scripts.train_model import train_model
from src.scripts.predict_model import predict_model
from src.visualization.visualize import predict_sentence
import logging
from codecarbon import EmissionsTracker
import json

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def ETL():
    # We will track CO2 emissions for each step of our pipeline
    
    tracker1 = EmissionsTracker(project_name = "Loading dataset", measure_power_secs=30) #, outuput_dir = Users/onaclara/Desktop/TAED2/LAB/TAED2-Emotions, emissions_endpoint = https://github.com/DaniGmzGnz/TAED2-Emotions/blob/develop/README.md)
    tracker1.start()

    # Load and transform the dataset.
    dataset = load("emotion")
    emissions1: float = tracker1.stop()

    tracker2 = EmissionsTracker(project_name = "Training model", measure_power_secs=200)
    tracker2.start()

    # Load the pre-trained model and fine-tune it with the dataset to do a classification task.
    model, results = train_model(dataset, train = True)
    emissions2: float = tracker2.stop()

    tracker3 = EmissionsTracker(project_name = "Prediction", measure_power_secs=200)
    tracker3.start()

    # Execute some predictions with the new fine-tuned model.
    metrics = predict_model(model, dataset)
    logger.info(f"The accuracy of the model is {metrics['accuracy']}")
    print(metrics)  
    emissions3: float = tracker3.stop()

    tracker4 = EmissionsTracker(project_name = "Visualizing metrics", measure_power_secs=30)
    tracker4.start()

    # Do some visualization for the metrics.
    result = predict_sentence(model, "Lately i'm feeling so tired.")
    emissions4: float = tracker4.stop()

    logger.info(f"Load Dataset Emissions: {emissions1}")
    logger.info(f"Fine-tuning Emissions: {emissions1}")
    logger.info(f"Predicting Emissions: {emissions1}")
    logger.info(f"Visualization Emissions: {emissions1}")

    print("EMISSION RESULTS OF EACH STEP:")
    print("1. Loading dataset emissions:", emissions1)
    print("2. Training model emissions:", emissions2)
    print("3. Prediction emissions:", emissions3)
    print("4. Visualizing metrics emissions:", emissions4)

    return result

def predict(sentence):
    config = {}
    dataset = load("emotion")
    model, results = train_model(dataset, train = False)

    with open('models/model/bert-base-uncased-emotion/config.json') as f:
        data = json.load(f)
        result = predict_sentence(model, data['model'], sentence)

    print(result)
    return result


