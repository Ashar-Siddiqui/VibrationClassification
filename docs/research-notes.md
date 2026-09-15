# Research artifact and reproducibility

## Manuscript

- **Title:** How Can a Machine Learning Classification Model Assist in Determining the Magnitude of a Rotating Shaft’s Unbalance?
- **Author:** Ashar Siddiqui
- **Affiliation printed in the paper:** Santa Clara High School, Santa Clara, CA, USA
- **Artifact:** [Six-page PDF](predictive-maintenance-research-paper.pdf)
- **Original supplied filename:** `tempRP_feb27.pdf`
- **SHA-256:** `87a1fe2c851b8afcf0161d4176df6f153210d5cb2898825a986d932967c0c789`

The PDF is preserved byte-for-byte. Its file timestamps are not used as a publication date. The repository citation does not assert a publication venue or peer-review status.

## Scope of the result

Section VII, Table I (page 5) reports 100% training and testing accuracy. The model compares only two classes: unbalance level 0 and level 4, using development (D) measurements for training and evaluation (E) measurements for testing. Section VIII (page 6) explicitly identifies intermediate levels 1–3 and further industrial validation as future work.

The repository publishes the paper's reported results. Adding this artifact did not rerun training or independently reproduce those numbers. The repository does not include the original dependency versions, dataset checksums, saved model, or a complete machine-readable experiment report.

## Relationship between the paper and the code

| Aspect | Supplied manuscript | Repository implementation |
| :--- | :--- | :--- |
| Model | Random Forest, 300 trees, depth 20, minimum leaf size 1, seed 0 | Same parameters in `main.py` |
| Train/test partition | Levels 0 and 4; D for training, E for evaluation | Loads `0Dtsfresh.npy`, `4Dtsfresh.npy`, `0Etsfresh.npy`, and `4Etsfresh.npy` |
| Feature sources | Discusses CSV statistical features and loads tsfresh arrays in its model listing | `preprocess.py` writes `data/`; `main.py` instead reads separately supplied `data_tsfresh/` arrays |
| Normalization | Page 5 shows separate `MinMaxScaler().fit_transform(...)` calls on training and test data | Scaling code exists inside a disabled triple-quoted block |

These differences must be reconciled before claiming an exact reproduction of the manuscript. They are documented here without changing the original research code or paper.

### Preprocessing caveat

After reshaping to `(batches, batchsize, columns)`, `preprocess.py` indexes `d[:, 1, :]`, `d[:, 2, :]`, and so on. These expressions select sample positions within a batch, rather than selecting the intended sensor columns across all samples. The same indexing appears in the manuscript listing. Consequently, the seven-feature output should be reviewed before it is used as a validated implementation of the described statistical preprocessing. This script is not invoked by `main.py`.

### Follow-up evaluation

An exact reproduction should identify the feature files and library versions, resolve the preprocessing and normalization differences, and save the evaluated configuration with its predictions. If normalization is used, fit it on training data and apply that same fitted transform to evaluation data. A broader predictive-maintenance study should evaluate intermediate unbalance levels and report per-class metrics and a confusion matrix under a documented split.
