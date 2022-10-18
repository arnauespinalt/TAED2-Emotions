import seaborn as sns
from transformers import DistilBertForTokenClassification
import torch
from transformers import AutoTokenizer
from transformers import pipeline

import matplotlib.pyplot as plt

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def predict_sentence(model, config, sentence):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = DistilBertForTokenClassification.from_pretrained('models/model').to(device)

    # encoded = sentence.map(tokenize, batched=True, batch_size=None)

    # encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    classifier = pipeline("text-classification", model = config, config='models/model/config.json')
    res = classifier(sentence)

    print(res)
    return res
