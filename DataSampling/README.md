# Text2SQL Multi-Agent Data Sampling

## Overview
This is a part of the main Multi-Agent LLMs in Text2SQL project that focuses on data sampling and analysis. The main goal of this part of the project is to analyze the Text2SQL datasets and create a new dataset that is more balanced and representative of the original dataset. The new dataset will be used to train the Multi-Agent LLMs in Text2SQL project.

## Data Sampling Project Structure
```
text2sql-multi-agent/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/
│   │   ├── spider/
│   │   ├── spider_v2/
│   │   └── bird/
│   ├── processed/
│   │   ├── statistics/
│   │   └── samples/
│   └── final/
├── notebooks/
│   ├── exploratory/
│   └── analysis/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_processor.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── global_statistics.py
│   │   └── sample_statistics.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_evaluator.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
└── tests/
    ├── __init__.py
    └── test_data_processor.py
```