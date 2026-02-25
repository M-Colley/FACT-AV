# Understanding the Effects of Different Reliabilities of Automated Vehicle Functionality on the Calibration of Trust

This repository contains code and analysis artifacts for studying how the mean Intersection over Union (mIoU) of AV perception outputs relates to human trust calibration. It includes feature-importance analysis, symbolic regression, and a multilayer perceptron (MLP) classifier trained on both reliability and demographic/contextual variables.

## Repository Layout
- `data/`: Preprocessed Excel datasets used across experiments.
- `MLP/`: PyTorch dataset, network, training, and evaluation scripts.
- `results/`: Generated figures and outputs for each modeling approach.
- `outputs/`: Intermediate results produced by the scripts.
- `ML-approaches.py`: Traditional ML baselines and feature-importance workflows.
- `main_*pysr_trust_calibration*.py`: PySR symbolic regression pipelines.

## Data
The `data/` directory contains preprocessed datasets used for model training and analysis:
- `all_combined_prepared.xlsx`
- `all_combined_prepared_removed_REI.xlsx`
- `all_combined_prepared_with_demographics.xlsx`
- `all_combined_prepared_with_demographics_with_baseline.xlsx`

Key columns include:
- `mIoU`: Reliability measure of AV perception output.
- `Trust1`–`Trust5`: Trust responses (5-class scale).
- `SCENARIO`, `INTRODUCTION`: Experiment context.
- Demographic features such as `Age`, `Gender`, `Education`, `Job`, `DrivingFrequency`, and `License`.

## Setup
1. Create a Python 3.11 (or newer) virtual environment.
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install --upgrade -r requirements.txt
   pip install --upgrade -r requirements-dev.txt
   ```

## Running the Analyses
### Feature Importance (Random Forest)
```bash
python ML-approaches.py
```
This script trains a `RandomForestRegressor` and reports feature-importance scores for trust prediction.

![Random Forest Feature Importance](results/ML-Approaches/feature_importance_random_classifier.png)

### Symbolic Regression (PySR)
```bash
python main_pysr_trust_calibration.py
```
Symbolic regression is used to derive equations that relate mIoU and trust, with variants that include personalized and group-based analyses.

![Symbolic Regression Example](results/PySR/relationship_pysr_other_rows_df_all_combined_prepared_legend.png)

### Multilayer Perceptron (MLP)
```bash
python MLP/train.py
python MLP/eval.py
```
The MLP is trained on ten features, including mIoU, demographics, and scenario metadata. It uses a 4-layer architecture with dimensions `[128, 512, 1024, 1024]`, dropout of 0.5, and the AdamW optimizer.

![MLP Training Snapshot](MLP/epochs/epoch1990.jpg)

## Testing
Run the automated test suite with:
```bash
pytest
```

## Results Summary
- **Feature Importance:** mIoU is a significant predictor of trust, with ~23% feature importance in Random Forest analysis.
- **Symbolic Regression:** Initial correlation is weak (R²=0.01) without filtering; filtered subsets show stronger trends.
- **MLP Classifier:** Achieves 74.2% accuracy and F1 score on a 5-class trust estimation task.

## Tools and Libraries
- **Python:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Symbolic Regression:** `pysr`
- **Explainability:** `shap`
- **Gradient Boosting:** `xgboost`, `lightgbm`, `catboost`
- **Deep Learning:** `torchmetrics` (PyTorch ecosystem)

## Qualitative Feedback Highlights
Participants provided open-ended feedback about reliability visualization and trust calibration, including:
- “The more videos I watched, the more I felt comfortable with the system.”
- “Some of the videos had distracting artifacts which impacted my trust level.”

## Citation
If you use this repository in academic work, please cite the associated publication or dataset (add citation details as appropriate).
