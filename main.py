import sys
from src.main import ETL, predict
from fastapi import FastAPI
import warnings
from codecarbon import EmissionsTracker


app = FastAPI()

REQUIRED_PYTHON = "python3"

@app.get("/state")
async def read_root():
    return {"Currently": "Running"}

@app.get("/{sentence}")
def main(task = 'predict', sentence = None):
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError(f"Unrecognized python interpreter: {REQUIRED_PYTHON}")

    if system_major != required_major:
        raise TypeError(f"This project requires Python {required_major}. Found: Python {sys.version}")

    else:
        print(">>> Development environment passes all tests!")


    if task=='train':
        # Execute the ETL task for our code.
        ETL()

    elif task=='predict': 
        # Execute the prediction of a given sentence.
        # sentence = input('Enter your sentence to predict: ')
        if sentence is None:
            warnings.warn("If you want to predict, you must give a sentence with the argument --sentence='Your sentence'")
        else: 
            tracker2 = EmissionsTracker(project_name = "Predict a sentence", measure_power_secs=200)
            tracker2.start()
            res = predict(sentence)
            emissions: float = tracker2.stop()
            print(f"Emissions on predicting a sentence: {emissions}")
            return {sentence: res}
    else: 
        warnings.warn("Not a valid task.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Select the task you want to perform.')

    parser.add_argument('--task', metavar='path', required=False,
                        help='The task you want to execute, a complete training or a prediction of a given sentence.')

    args = parser.parse_args()
    main(task=args.task)
