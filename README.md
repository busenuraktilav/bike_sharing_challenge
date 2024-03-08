# bike_sharing_challenge

This project makes explorative analysis on [bike sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and builds a prediction modelfor the hourly utilization "cnt".

## Getting Started

Create the conda environment

```bash
conda create -n "bike-sharing" python=3.9
```

If activate the environment

```bash
conda activate bike-sharing
```

install the requirements to the conda environment

```bash
pip install -r requirements.txt
```

Change the cfg.yaml file as you like to make experiments and try the code with different models and preprocessing steps. The default cfg.yaml runs the random forest algorithm and uses mean absolute error as a metric. The code outputs the selected metric

Run the code by

```bash
python -m src.main
```

### Prerequisites

You need to have a conda environment installed to your machine. You can install [here](https://docs.anaconda.com/free/miniconda/index.html)

## How to Contribute

Run the following command to install the pre-commit hooks based on the configuration file. This ensures that the black and unit tests are run before making a commit to the repo.

```bash
pre-commit install
```

