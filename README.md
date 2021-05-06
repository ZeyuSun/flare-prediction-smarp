# flare-prediction-smarp
Flare prediction with SMARP and SHARP

## Download data
Change the email and data directory in `download.py` and run `python download.py`.

## Preprocess data
1. Change the data directory in `preprocess.py`.
2. Install Redis. (Alternatively, change the default value of `redis` to False in function `query` in `data.py`)
3. Run `python preprocess.py`.
