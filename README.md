# Squiggles: MSMC Time Series Curve Visualizer and Clustering Tool
![App homepage](images/main.png)
Python Dash app/dashboard using tslearn for clustering time series data.

# Installation
1. Clone repo an MSMC_clustering submodule with: `git clone <repo> --recursive`
2. Create environment from `requirements.txt`
3. Activate environment
4. run `app.py` with python with: `python app.py`
5. Open app on any browser using default Dash url: `localhost:8050` 

# The dealio
![Pretty much the whole app](images/catch_my_drift.jpeg)


# TODO:
- Now with common names, refer to curves by common name and move file name to a new field called "file name" or something
- Get loading a model and summary stats like intended cluster size or points working
- Get creating a model on the fly working