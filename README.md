# Pixel Ratioing Tookit for CRISM

## Introduction

This tool utilizes the Random Forest model to predict bland pixels and applies pixel ratioing to reduce spectral noise, with the aim of enhancing spectral features.

## Usage

You can install an environment with all the dependencies with:

```bash
conda env create -f environment.yaml
```

Use the following to activate the environment:

```bash
conda activate crism
```

## Note

"functions.py" includes the necessary functions.

"Training.ipynb" includes methods for model training.

"Example_Code.ipynb" includes instructions on how to use the program.

To retrain the model, see "Training.ipynb". Otherwise, please refer to "Example_Code.ipynb", the notebook will automatically download the pre-trained model and a sample image, and display the result figure.
