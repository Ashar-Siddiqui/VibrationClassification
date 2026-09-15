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

---

# Predictive Maintenance — Vibration Classification

Research and Python code exploring whether a Random Forest can distinguish a balanced rotating shaft from one with strong unbalance using vibration-derived features.

**Author:** Ashar Siddiqui

**Stack:** Python · NumPy · SciPy · scikit-learn · tsfresh feature data

**[Read the research paper (PDF)](docs/predictive-maintenance-research-paper.pdf)** · [Research and reproducibility notes](docs/research-notes.md) · [Citation](CITATION.cff)

## Research paper

**How Can a Machine Learning Classification Model Assist in Determining the Magnitude of a Rotating Shaft’s Unbalance?**

Ashar Siddiqui · Santa Clara High School, Santa Clara, CA, USA *(affiliation printed in the manuscript)*

The six-page manuscript examines Random Forest classification for predictive maintenance using rotating-shaft vibration data collected by Mey et al. It covers the research motivation, related work, methodology, results, and limitations. The PDF is the author's supplied manuscript, preserved without modification.

### Reported findings

| Evaluation in the manuscript | Reported accuracy |
| :--- | ---: |
| Training set: unbalance levels 0 and 4 | 100% |
| Testing set: unbalance levels 0 and 4 | 100% |

Source: Section VII, Table I, page 5 of the [paper](docs/predictive-maintenance-research-paper.pdf). These are **paper-reported results for a binary extreme-level comparison**, not a new reproduction run. Level 0 represents no unbalance; level 4 is the strongest unbalance in the dataset. Performance on intermediate levels 1–3, remaining useful life prediction, and industrial deployment are not established by this experiment.

## Implementation

- [`main.py`](main.py) loads precomputed tsfresh feature arrays for levels 0 and 4, uses the D files for training and E files for testing, and trains a 300-tree Random Forest (`max_depth=20`, `min_samples_leaf=1`, `random_state=0`). It prints training/test accuracy and feature importances.
- [`preprocess.py`](preprocess.py) is a separate raw-CSV feature preparation script. It skips the first 50,000 data rows, groups the remaining rows into 4,096-row batches, and writes `.npy` arrays under `data/`.

The scripts are retained as the research implementation. The CSV preprocessing output is **not** the precomputed `data_tsfresh/` input consumed by `main.py`. The manuscript and code also differ in whether normalization is enabled. See the [reproducibility notes](docs/research-notes.md) before interpreting or extending the experiment.

## Run the classifier

Create a Python environment and install the packages imported by the scripts:

```sh
git clone https://github.com/Ashar-Siddiqui/VibrationClassification.git
cd VibrationClassification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy scikit-learn
```

The original experiment does not include a dependency lockfile; this installs a new environment rather than reconstructing the exact historical environment.

Download the upstream feature files from the [Mey et al. repository](https://github.com/deepinsights-analytica/ieee-etfa2020-paper/tree/master/data), then place them under the local names expected by `main.py`:

| Upstream file | Local path |
| :--- | :--- |
| [`data/0D/a1_tsfresh.npy`](https://github.com/deepinsights-analytica/ieee-etfa2020-paper/blob/master/data/0D/a1_tsfresh.npy) | `data_tsfresh/0Dtsfresh.npy` |
| [`data/0E/a1_tsfresh.npy`](https://github.com/deepinsights-analytica/ieee-etfa2020-paper/blob/master/data/0E/a1_tsfresh.npy) | `data_tsfresh/0Etsfresh.npy` |
| [`data/4D/a1_tsfresh.npy`](https://github.com/deepinsights-analytica/ieee-etfa2020-paper/blob/master/data/4D/a1_tsfresh.npy) | `data_tsfresh/4Dtsfresh.npy` |
| [`data/4E/a1_tsfresh.npy`](https://github.com/deepinsights-analytica/ieee-etfa2020-paper/blob/master/data/4E/a1_tsfresh.npy) | `data_tsfresh/4Etsfresh.npy` |

Run the classifier from the repository root:

```sh
python main.py
```

For the separate CSV preprocessing experiment, obtain `0D.csv`, `0E.csv`, through `4D.csv`, `4E.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft), place them in the repository root, and run `python preprocess.py`. The first character of `0D` and `0E` is the digit zero. Review the preprocessing caveat in the research notes before relying on the generated features.

## Citation and attribution

To cite this manuscript, use [CITATION.cff](CITATION.cff) or the [BibTeX entry](docs/paper.bib). No publication date, DOI, venue, or peer-review status is assigned here because the supplied manuscript does not identify them.

The experimental dataset and upstream feature files originate from the work of Oliver Mey, Willi Neudeck, Andre Schneider, and Olaf Enge-Rosenblatt, **Machine Learning-Based Unbalance Detection of a Rotating Shaft Using Vibration Data** ([paper](https://arxiv.org/abs/2005.12742), [code and data](https://github.com/deepinsights-analytica/ieee-etfa2020-paper)). Their work is cited in the manuscript; the data and feature files are not redistributed in this repository.
