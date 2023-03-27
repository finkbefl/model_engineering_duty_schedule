Model Engineering: Duty Schedule
==============================================

Case study on the development of a model to predict the duty schedule of an rescue service with python.

- [Prerequisites](#prerequisites)
- [Project Description](#project-description)
- [Running the Program](#running-the-program)
  - [Data Preparation](#data-preparation)
  - [Feature Selection and Data Split](#feature-selection-and-data-split)
  - [Baseline Model](#baseline-model)
  - [Model Benchmark](#model-benchmark)
  - [Model Evaluation](#model-evaluation)
- [Restrictions](#restrictions)
- [Project Structure](#project-structure)
- [License](#license)

## Prerequisites

Use conda to create a virtual environment for this python project and install all dependencies:

```bash
conda env create --file program_requirements.yml
```

The virtual environment must be activated:

```bash
conda activate dutySchedule_env 
```

## Project Description

The project contains corresponding Python scripts for data preparation, feature selection including the data-split in test and training data, the creation of a baseline model and the execution of a model benchmark. Subsequently, the model with the best performance will be evaluated.

## Running the Program

Note: The data is not pushed into this repository, so a clone of the project is not runnable out of the box!

### Data Preparation

```bash
python src/data/data_prep.py
```

### Feature Selection and Data Split

```bash
python src/data/featurization.py
```

### Baseline Model

```bash
python src/model/baseline_model.py
```

### Model Benchmark

```bash
python src/model/model_benchmark.py
```

### Model Evaluation

```bash
python src/evaluation/model_evaluation.py
```

## Restrictions

This case study is about performing a first loop of model engineering. The results of the models are of course not yet satisfactory and must be further optimized.

## Project Structure

```
.
├── data
│   ├── modeling: Generated data for modelling
│   │   ├── test_input.csv
│   │   ├── test_target.csv
│   │   ├── train_input.csv
│   │   ├── train_target.csv
│   │   └── y_pred.csv
│   ├── processed: Generated processed data
│   │   ├── sickness_table_prepared.csv
│   │   └── sickness_table_prepared_stationary.csv
│   └── raw: Raw input data
│       └── sickness_table.csv
├── docs
│   ├── data
│   │   └── eda: The exploratory data analysis as jupyter notebook
│   │       ├── exploratory_data_analysis.ipynb
├── .gitignore
├── LICENSE
├── logs: Generated logs
│   ├── baseline_model.log
│   ├── data_prep.log
│   ├── featurization.log
│   ├── model_benchmark.log
│   ├── model_evaluation.log
│   └── utils.plot_data.log
├── output: Generated plots
│   ├── baseline_model.html
│   ├── data_prep_high_corr_col.html
│   ├── data_prep_pot_outl.html
│   ├── model_benchmark.html
│   ├── model_evaluation.html
│   ├── prepared_input_data.html
│   ├── prepared_input_data_stationary.html
│   ├── preprocessed_input_data.html
│   ├── test_data.html
│   ├── train_data.html
│   ├── unittestAddBox.html
│   ├── unittestMultipleFigures.html
│   ├── unittestMultipleLayersCircle.html
│   └── unittestMultipleLayersVBar.html
├── program_requirements.yml
├── README.md
├── src: Python source code
│   ├── data
│   │   ├── data_prep.py
│   │   └── featurization.py
│   ├── evaluation
│   │   └── model_evaluation.py
│   ├── model
│   │   ├── baseline_model.py
│   │   └── model_benchmark.py
│   └── utils
│       ├── check_parameter.py
│       ├── csv_operations.py
│       ├── own_exceptions.py
│       ├── own_logging.py
│       ├── plot_data.py
└── tests: A small set of first python unit tests
    ├── __init__.py
    └── utils
        ├── own_logging_test.py
        ├── plot_data_test.py
```

## License
[MIT](LICENSE)