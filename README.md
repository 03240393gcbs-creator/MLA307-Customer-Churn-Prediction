<<<<<<< HEAD
# Customer Churn Prediction Using Sequence Models

## MLA307 — Sequence Models
**Year 2, Semester I | Mini Project**

## 1. Project Overview

This project investigates customer churn prediction using sequential user activity data. The objective is to identify users who are likely to cancel their service based on their historical interaction patterns.

The project uses the Mini Sparkify dataset and compares recurrent neural network models with a traditional machine learning baseline. It explores whether sequence models can learn useful temporal patterns from customer activity to support early churn identification.

## 2. Project Objectives

The main objectives are to:

- Prepare customer activity data for sequence modelling.
- Construct fixed-length sequences representing daily customer behaviour.
- Implement and compare SimpleRNN, GRU, and LSTM models.
- Establish logistic regression as a baseline.
- Evaluate model performance using appropriate classification metrics.
- Investigate the potential usefulness of churn predictions for customer retention.

## 3. Dataset

The project uses the Mini Sparkify event dataset, `mini_sparkify_event_data.json`, which contains user interaction events from a music-streaming service.

The data includes information about user activity, such as listening to songs, visiting pages, interacting with features, and cancellation events.

**Dataset source:** Add the verified source URL for the exact dataset version used in this project. Ensure the dataset is obtained and used according to its applicable terms.

The dataset is processed to create customer-level activity sequences and churn labels for supervised learning.

## 4. Data Preparation and Feature Engineering

The `prepare_sequences.py` script prepares the dataset for sequence modelling.

The implementation uses **3 November 2018** as the cutoff date. Users who had already cancelled on or before the cutoff are excluded from the eligible prediction cohort. Churn labels are determined using cancellation events occurring after the cutoff within the available observation period.

For each eligible customer, daily activity is represented using a fixed window of **30 days** and **13 activity features**:

1. Event count
2. Number of sessions
3. Songs played
4. Minutes listened
5. Thumbs-up interactions
6. Thumbs-down interactions
7. Add-to-playlist interactions
8. Add-friend interactions
9. Advertisement-related activity
10. Downgrade events
11. Upgrade events
12. Home-page visits
13. Error events

These features form a three-dimensional input structure for the recurrent models, representing customers, time steps, and daily features.

The data is divided into training, validation, and test sets at the customer level using stratified splitting and a fixed random seed. Feature scaling is fitted using active-day observations from the training customers, and class weighting is used to help address class imbalance.

## 5. Models

The project evaluates four approaches.

### 5.1 SimpleRNN

SimpleRNN provides a basic recurrent neural network baseline for learning patterns across daily customer activity sequences.

### 5.2 Gated Recurrent Unit (GRU)

GRU uses gating mechanisms to learn sequential relationships while maintaining a relatively compact recurrent architecture.

### 5.3 Long Short-Term Memory (LSTM)

LSTM uses memory cells and gating mechanisms to learn dependencies across a sequence of customer activity.

### 5.4 Logistic Regression Baseline

Logistic regression provides a traditional machine learning comparison. The 30-day sequence is flattened into a single feature vector before training the classifier.

The recurrent models use a sigmoid output for binary classification. The training process uses binary cross-entropy, the Adam optimizer, class weighting, and early stopping. GRU and LSTM configurations are compared using the hyperparameter settings defined in the project code.

## 6. Model Evaluation

The evaluation process separates model selection from final testing.

- **Validation:** Used to compare configurations and select the classification threshold for the final sequence model.
- **Cross-validation:** The `cv.py` script performs five-fold stratified cross-validation on the training and validation partitions, excluding the held-out test set.
- **Test evaluation:** The `evaluate_model.py` script evaluates the selected model on the held-out test set.
- **Baseline comparison:** Logistic regression provides a reference point for evaluating the recurrent models.

Evaluation should include relevant classification metrics such as accuracy, precision, recall, F1-score, and ROC-AUC where produced by the implementation. Because churn prediction can involve class imbalance, the results should not be interpreted using accuracy alone.

The sequence-model threshold is selected using validation predictions according to the F1-based selection procedure in the code. The logistic regression baseline uses its configured threshold.

## 7. Experimental Results

The final results must be populated using the actual output files generated by the project.

| Model | Test ROC-AUC | Test F1-score | Test Recall |
|---|---:|---:|---:|
| SimpleRNN | To be populated | To be populated | To be populated |
| GRU | To be populated | To be populated | To be populated |
| LSTM | To be populated | To be populated | To be populated |
| Logistic Regression | To be populated | To be populated | To be populated |

These entries are placeholders, not experimental results. Replace them with the actual values before submission. Include the selected configuration, chosen threshold, cross-validation results, and cohort sizes wherever the code reports them.

The best-performing model should be identified only after comparing the actual evaluation results.

## 8. Project Structure

The main scripts in the repository include:

- `prepare_sequences.py` — prepares customer activity sequences and churn labels.
- `train_model.py` — trains the machine learning and recurrent models.
- `cv.py` — evaluates selected configurations using stratified cross-validation.
- `evaluate_model.py` — performs final model evaluation.

The repository may also contain supporting data-preparation scripts, result files, and backup files.

## 9. Running the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/03240393gcbs-creator/MLA307-Customer-Churn-Prediction.git
cd MLA307-Customer-Churn-Prediction
```

### Step 2: Set up the Python environment

Create and activate a virtual environment, then install the dependencies required by the scripts.

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages after confirming the project's dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn joblib
```

If the code imports additional packages, install those as well. Record the tested dependency versions in a `requirements.txt` file for reproducibility.

### Step 3: Prepare the dataset

Place `mini_sparkify_event_data.json` in the location expected by `prepare_sequences.py`, then run:

```bash
python prepare_sequences.py
```

### Step 4: Train the models

```bash
python train_model.py
```

### Step 5: Run cross-validation

```bash
python cv.py
```

### Step 6: Evaluate the final model

```bash
python evaluate_model.py
```

Run the scripts in the order above, checking each script's expected inputs and generated outputs. If a script expects an intermediate file that is not present, follow the file paths defined in the code before proceeding.

## 10. Limitations

This project has several limitations:

- The observation period is limited to the available dataset.
- Churn occurring after the dataset ends cannot be observed.
- The dataset may contain relatively few churn cases compared with non-churn cases.
- Activity-based features may not capture every factor influencing customer cancellation.
- Results may vary with the data split, model configuration, and classification threshold.
- Predictive performance alone does not demonstrate that a retention intervention will reduce churn.

## 11. Conclusion

This project provides a comparison of recurrent sequence models and logistic regression for customer churn prediction using daily user activity. The approach combines temporal feature engineering, customer-level data splitting, model training, cross-validation, and held-out evaluation.

The final conclusions about model performance and the practical value of churn prediction will be based on the measured results. These results can help assess whether sequence-based approaches offer a useful improvement over the baseline and whether further evaluation for customer-retention applications is warranted.

## 12. Reproducibility

For reproducibility, retain the dataset source, document the software dependencies, use the configured random seed, and report the actual split sizes and model settings. The README, source code, generated results, and final report should describe the same experimental procedure.
=======
# MLA307-Customer-Churn-Prediction
# MLA307 – Customer Churn Prediction with Sequence Models

Mini project (Topic 1: Customer Churn Prediction). Predicts which music-streaming customers
go on to cancel, from 30 days of daily activity, using Simple RNN / GRU / LSTM models and a
logistic-regression baseline.

## Dataset
Mini Sparkify event data (`mini_sparkify_event_data.json`, Udacity Data Scientist Nanodegree).
Source URL: <ADD VERIFIED URL>. The file is not included in this repo. Place it in the repository root.

## Method (summary)
- One prediction point per customer: first *Cancellation Confirmation* (churn = 1) or last event (churn = 0).
- Input: 30 daily rows × 13 features, ending 1 day before the prediction point (GAP = 1). Cancellation events are removed from inputs.
- Models: Simple RNN, GRU, LSTM (tuned) vs logistic regression. Threshold chosen on validation data (max F1).
- Limitation: prediction points depend on each customer's own outcome (retrospective design).

## Setup
```
pip install -r requirements.txt
```

## Run order
1. `data_preparation/` – must contain `sequence_data.npz` (customer split and feature names; created by `prepare_data.py`).
2. `python prepare_sequences.py` – builds `model_results/sequence_data_v2.npz`, scaler, class weights.
3. `python train_model.py` – Simple RNN, GRU/LSTM tuning, seed check, saves `final_model.keras` and figures.
4. `python cv.py` – 5-fold cross-validation (`cv_auc.json`).
5. `python evaluate_model.py` – test metrics, ROC curve, confusion matrix, at-risk customer list.
6. `python report_summary.py` – prints all report numbers and saves `pattern_plot.png`.

## Outputs (`model_results/`)
`test_metrics.csv`, `classification_report.txt`, `confusion_matrix.csv`, `roc_curves.png`,
`training_curves.png`, `tuning_results.csv`, `seed_robustness_validation.csv`, `cv_auc.json`,
`test_churn_predictions.csv`, `at_risk_customers_top20.csv`, `pattern_plot.png`.

## Group members
Pema Lhamo 
Sonam Rinzin Lhendup 
Phuntsho Rangdrol
>>>>>>> e8f465c (Add report summary script, requirements, README)
