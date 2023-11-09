
# CS5228 Final Project: Predicting HDB Rental Prices
This is the project repository for the CS5228 final project for AY23/24 Semester 1 for Group 3 of Benjamin Ho, Bernard Cheng, Pham Quang Minh, and Thong Nguyen.

# Setup
## Jupyter Notebook
Follow this [guide](https://jupyter.org/install) to set up Jupyter Notebook

## Data Science Packages
```
pip install scikit-learn matplotlib pandas numpy
```

## Clone the Repository
```
git clone https://github.com/benhgm/cs5228_flat_rental.git
```

# Repository Structure
## Models
All trained models are stored in the `models` folder.
## Data
All data files are stored in the `data` folder.
- `data/train.csv` contains the raw training data
- `data/test.csv` contains test data used for making predictions for Kaggle submission
- `data/auxiliary-data` contains all the auxiliary data provided for the project

## Processing Scripts
All scripts containing methods for processing the data are stored in the `src` folder.
- `src/cleaning.py` contains methods for performing data cleaning
- `src/preprocessing.py` contains methods for preprocessing the data
- `src/feature_eng.py` contains methods for performing feature engineering and adding additional features to the dataset

# Notebooks for Data Analysis
We created various notebooks to support us in the data analysis process. This was helpful in doing Exploratory Analysis.
- `clean_data.ipynb` implements the data cleaning steps and saves a copy of cleaned data in `data/train_cleaned.csv`
- `preprocess_data.ipynb` implements the preprocessing steps and saves a copy of the preprocessed data in `data/train_preprocessed.csv`
- `analyse_data.ipynb` imeplements additional analysis and feature engineering steps and saves a copy of the extended dataset in `data/train_feat_eng.csv`

# Modelling
## Generate Train / Test Data
Before training any prediction model, we first generate the training and testing data. Use the `generate_data.ipynb` notebook to generate 4 different datasets for modelling.

## Train the Models and Make Predictions
We provide 7 notebooks in the `experiments` folder, one for each model type that we experimented with.
- `linear_regression.ipynb` Implements a Linear Regression Model
- `kneighbours.ipynb` Implements a K-Neighbours Regressor
- `bagging_regressor.ipynb` Implements a Bagging Regression Model
- `random_forest.ipynb` Implements a Random Forest Regression Model
- `gradient_boost.ipynb` Implements a Gradient Boosting Regressor
- `lightgbm.ipynb` Implements a LightGBM Gradient Boosting Model
- `adaboost.ipynb` Implements a AdaBoost Regressor
