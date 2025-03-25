# Text2SQL Multi-Agent

## Overview
This project aims to build a multi-agent system for text-to-SQL generation. The system will be trained on the Spider dataset and evaluated on the Spider and Bird datasets.

## Project Structure
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