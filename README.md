# Wildfire Ignition Point Classification

## Overview

This project aims to build a classification model to predict whether a given geographical point has a high probability of being an ignition point for a wildfire.

The dataset includes various features related to vegetation, weather conditions, and geographical information for each point. The target variable (`ignition`) indicates whether the point is a known ignition point (`1`) or not (`0`).

## Dataset

The dataset contains various features, including:

- **Geographical Information**: `elevation`, `slope`, `aspect`
- **Vegetation Class**: `cropland`, `forest`, `wetland`, etc.
- **Weather Data**: `max_temp`, `avg_temp`, `max_wind_vel`, `avg_rel_hum`, `sum_prec`
- **Population**: `pop_dens`
- **Target**: `ignition` (1 if the point is an ignition point, 0 otherwise)

Each row in the dataset represents either an ignition point or a non-ignition point, along with the associated environmental and weather conditions.

## Project Structure
This project is organized as follows:

```bash
├── data                    # Data folder (dataset not included)
├── notebooks                # Jupyter notebooks for exploration and analysis
├── src                      # Source code for the project
│   ├── data                 # Data loading and preprocessing scripts
│   │   ├── data_cleaning.py
│   │   ├── data_processing.py
│   │   └── load_data.py
│   ├── features             # Feature engineering scripts
│   │   └── feature_engineering.py
│   ├── models               # Model training and evaluation scripts
│   │   ├── evaluate_model.py
│   │   └── model_training.py
│   └── visualization        # Scripts for visualizing results
│       └── visualize.py
├── README.md                # Project overview
└── requirements.txt         # Dependencies
```
