# Predicting Solar Flares Using CNN and LSTM on Two Solar Cycles of Active Region Data

[![DOI](https://zenodo.org/badge/358039770.svg)](https://zenodo.org/badge/latestdoi/358039770)

This repository contains codes for data processing, model training, and performance evaluation used in paper *Predicting Solar Flares Using CNN and LSTM on Two Solar Cycles of Active Region Data* (Sun et al. 2022)

## Quick start
1. Download data: Change the email and data directory in `download.py` and run `python download.py`.
2. Preprocess data
  1. Change the data directory in `preprocess.py`.
  2. Install Redis. (Alternatively, change the default value of `redis` to False in function `query` in `data.py`)
  3. Run `python preprocess.py`.
3. Exploratory data analysis (`eda.py`)
4. Fit and evaluate machine learning methods:
  1. Scikit-learn models
  2. Pytorch-lightning models
    1. MLP
    2. LSTM
    3. 2D CNN
    4. 3D CNN
5. Present results (notebooks/mlflow_results.ipynb)

## Citation
```plain
@article{sun2022predicting,
  title={Predicting solar flares using cnn and lstm on two solar cycles of active region data},
  author={Sun, Zeyu and Bobra, Monica G and Wang, Xiantong and Wang, Yu and Sun, Hu and Gombosi, Tamas and Chen, Yang and Hero, Alfred},
  journal={The Astrophysical Journal},
  volume={931},
  number={2},
  pages={163},
  year={2022},
  publisher={IOP Publishing}
}
```
