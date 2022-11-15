## Setup

1) Install the latest version of Python 3 from the python website.

    `https://www.python.org/downloads/`

1) Create a virtual environment

    `python3 -m venv .venv.nosync`

1) Activate the virtual environment

    `source .venv.nosync/bin/activate # if you want to deactivate, then run "deactivate"`

1) On the command line, install the dependencies necessary for this project.

    `python3 -m pip install tsfresh`

1) Download the following datasets from `https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft`

    - `OD.csv`
    - `OE.csv`
    - `1D.csv`
    - `1E.csv`
    - `2D.csv`
    - `2E.csv`
    - `3D.csv`
    - `3E.csv`
    - `4D.csv`
    - `4E.csv`

1) Run the `preprocess.py` script to obtain data containing the 7 features outlined on page 6 of this paper by Mey et al. `https://arxiv.org/abs/2005.12742`

	`python3 preprocess.py`

1) Download the TS-fresh features from this repo: `https://github.com/deepinsights-analytica/ieee-etfa2020-paper/tree/master/data`
