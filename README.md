TAED2-Emotions
==============================
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

We are the team that develops a model to classify some text input into a variety of emotions.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# Model Card

## Model Details

### Developers
The model has been developed by @bhadresh-savani and downloaded from github: https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion 

The model has been versioned and adapted by the Emotion team, formed by the following Data Science students at the UPC: Ona Clarà Ventosa, Sandra Millet Salvador, Arnau Espinalt Xixons and Daniel Gómez González. 

### Model date

Created September 2021. 
Used by the Emotion team September 2022. 

### Model description

[Distilbert](https://arxiv.org/abs/1910.01108) is created with knowledge distillation during the pre-training phase which reduces the size of a BERT model by 40%, while retaining 97% of its language understanding. It's smaller, faster than Bert and any other Bert-based model.

[Distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) finetuned on the emotion dataset using HuggingFace Trainer with below Hyperparameters
```
 learning rate 2e-5, 
 batch size 64,
 num_train_epochs=8,
```
### Model type
The model is a Natural Language Processing model. 

### Training procedure
[Colab Notebook](https://github.com/bhadreshpsavani/ExploringSentimentalAnalysis/blob/main/SentimentalAnalysisWithDistilbert.ipynb)

### Resource information
[None]

### Citation details

* [Natural Language Processing with Transformer By Lewis Tunstall, Leandro von Werra, Thomas Wolf](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/)

### License
[None]

### Where to send questions or comments about the model
[None]

## Intended Use

The model is intended to be used for sentiment analysis, which given a dataset of messages or phrases in English, classifies them into 5 types of emotions.  

## Metrics

### Model performance measures

```json
{
'test_accuracy': 0.938,
 'test_f1': 0.937932884041714,
 'test_loss': 0.1472451239824295,
 'test_mem_cpu_alloc_delta': 0,
 'test_mem_cpu_peaked_delta': 0,
 'test_mem_gpu_alloc_delta': 0,
 'test_mem_gpu_peaked_delta': 163454464,
 'test_runtime': 5.0164,
 'test_samples_per_second': 398.69
 }
```

### Model Performance Comparision on Emotion Dataset from Twitter:

| Model | Accuracy | F1 Score |  Test Sample per Second |
| --- | --- | --- | --- |
| [Distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) | 93.8 | 93.79 | 398.69 |
| [Bert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion) | 94.05 | 94.06 | 190.152 |
| [Roberta-base-emotion](https://huggingface.co/bhadresh-savani/roberta-base-emotion) | 93.95 | 93.97| 195.639 |
| [Albert-base-v2-emotion](https://huggingface.co/bhadresh-savani/albert-base-v2-emotion) | 93.6 | 93.65 | 182.794 |


### Decision thresholds
[None]

### Variation approaches
The model is a variation itself, and it has been explained in the basic information.

## Evaluation Data

### Datasets 

The model has been finetuned with the dataset emotion, which contains text messages of twitter. 

https://huggingface.co/nlp/viewer/?dataset=emotion

### Motivation

This datasets allow us to use the model for a Sentiment Analysis task in which we take as an input text, which is a complex input to process. 

### Preprocessing
[None]

## Ethical consideration
Co2_eq_emissions:
- Emissions: 43,24 kg Co2
    -- Loading dataset: 4,3e-4 kg Co2
    -- Training: 42,17 kg Co2
    -- Evaluation: 1,07 kg Co2
- Source: code carbon
- Training_type: fine-tuning
- Geographical_location: Barcelona, Catalonia
- Hardware_used: 8 CPUs and 0 GPUs, 16Gb ram/cpu, Cpu model: Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz

## Caveats and recommendations
[None]





# Dataset Card for Emotion


## Dataset Description

- *Homepage: *
[Twitter-Sentiment-Analysis](https://huggingface.co/nlp/viewer/?dataset=emotion).

### Dataset Summary

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. For more detailed information please refer to the paper.

### Supported Tasks and Leaderboards

[None]

### Languages

English (en)

## Dataset Structure

### Data Instances

*training data: default*

Size of downloaded dataset files: 1.97 MB
Size of the generated dataset: 2.07 MB
Total amount of disk used: 4.05 MB

*evaluation data: emotion*

Size of downloaded dataset files: 1.97 MB
Size of the generated dataset: 2.09 MB
Total amount of disk used: 4.06 MB

### Data Fields

The data fields are the same among all splits:

*default:*
- `text`: a `string` feature.
- `label`: a classification label, with possible values including `sadness` (0), `joy` (1), `love` (2), `anger` (3), `fear` (4), `surprise` (5).
*emotion:*
- `text`: a `string` feature.
- `label`: a `string` feature.


### Data Splits

| name    | train | validation | test |
| ------- | ----: | ---------: | ---: |
| default | 16000 |       2000 | 2000 |
| emotion | 16000 |       2000 | 2000 |

## Dataset Creation

### Curation Rationale
[None]

### Source Data
#### Initial Data Collection and Normalization

[None]

#### Who are the source language producers?
Twitter. 


## Considerations for Using the Data

### Social Impact of Dataset
[None]

### Discussion of Biases
[None]

### Other Known Limitations
[None]

## Additional Information

### Dataset Curators
[None]

### Licensing Information
[None]

### Citation Information
```
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}
```

### Contributions

Thanks to [@lhoestq](https://github.com/lhoestq), [@thomwolf](https://github.com/thomwolf), [@lewtun](https://github.com/lewtun) for adding this dataset.


