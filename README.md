# flare-prediction-smarp
Flare prediction with SMARP and SHARP

## Quick start
1. Download data: Change the email and data directory in `download.py` and run `python download.py`.
2. Preprocess data
  1. Change the data directory in `preprocess.py`.
  2. Install Redis. (Alternatively, change the default value of `redis` to False in function `query` in `data.py`)
  3. Run `python preprocess.py`.
3. Exploratory data analysis
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
@article{10.1002/essoar.10508256.3,
  author = {Sun, Zeyu and Bobra, Monica and Wang, Xiantong and Wang, Yu and Sun, Hu and Gombosi, Tamas and Chen, Yang and Hero, Alfred},
  title = {Predicting Solar Flares using CNN and LSTM on Two Solar Cycles of Active Region Data},
  journal = {Earth and Space Science Open Archive},
  pages = {31},
  year = {2021},
  DOI = {10.1002/essoar.10508256.3},
  url = {https://doi.org/10.1002/essoar.10508256.3},
}
```
