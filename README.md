# flare-prediction-smarp
Flare prediction with SMARP and SHARP

## Main results
<tables>

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
5. Generate a leaderboard

## Citation
