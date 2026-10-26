# MLA307 Customer Churn Prediction Using Sequence Models

## 1. Project Overview

This project focuses on predicting customer churn using sequence models. Customer churn occurs when a customer stops using a service. Predicting churn helps businesses identify customers who may leave and take action to improve customer retention.

The project uses the Mini Sparkify event dataset from Udacity. It contains customer activity records, such as listening to songs, interacting with the service, changing subscription levels, and cancellation-related events.

The project transforms customer activity into time-based sequences and compares different machine learning and deep learning models to predict customer churn.

**Course:** MLA307 – Sequence Models  
**Project Topic:** Customer Churn Prediction

### Group Members

- Pema Lhamo
- Sonam Rinzin
- Lhendup Phuntsho Rangdrol

## 2. Project Objectives

The main objectives of this project are to:

- Understand customer behaviour using historical activity data.
- Prepare customer activity records as time-based sequences.
- Identify patterns that may indicate customer churn.
- Develop and compare Simple RNN, GRU, and LSTM models.
- Compare sequence models with a Logistic Regression baseline.
- Evaluate model performance using appropriate classification metrics.
- Identify customers who may be at risk of leaving.
- Explain how businesses can use churn predictions to improve customer retention.

## 3. Dataset Description

### Dataset Source

The project uses the Mini Sparkify event dataset provided by Udacity.

**Reference:**

Udacity. (n.d.). *Mini Sparkify event data* [Data set]. https://udacity-dsnd.s3.amazonaws.com/sparkify/mini_sparkify_event_data.json

**Dataset URL:**  
https://udacity-dsnd.s3.amazonaws.com/sparkify/mini_sparkify_event_data.json

### Dataset Overview

The Mini Sparkify dataset contains event-level information about customers using a music streaming service. Each row represents an activity or event performed by a user.

The dataset contains information such as:

- Customer or user ID
- Event name
- Event timestamp
- Page or activity type
- Session information
- Device and location information
- Subscription level
- Cancellation-related events

The dataset is suitable for this project because customer behaviour changes over time. These changes can be represented as sequences and used to predict churn.

### Dataset Summary

The recorded data preparation output reported the following statistics:

| Description | Value |
|---|---:|
| Raw event records | 286,500 |
| Columns | 18 |
| Customers with a user ID | 225 |
| Rows with empty user IDs excluded | 8,346 |
| Exact duplicate rows in raw data | 0 |
| Churned customers | 52 |
| Non-churned customers | 173 |
| Date range | 1 October 2018 – 3 December 2018 |

These statistics are from the recorded `prepare_data.py` output. The sequence-building script should be rerun to confirm the statistics and data splits used by the final models.

The raw dataset is not required to be committed to GitHub. It can be downloaded from the original source.

## 4. Data Preparation and Sequence Construction

Raw event data cannot be directly used by sequence models. The data must be cleaned, transformed into numerical features, and arranged into sequences.

### 4.1 Data Cleaning

The data preparation process includes:

1. Loading the Mini Sparkify JSON dataset.
2. Removing records with missing or blank user IDs.
3. Converting timestamps into a suitable date and time format.
4. Sorting customer events by time.
5. Identifying churned and non-churned customers.
6. Creating numerical features from customer activity.
7. Preparing the data for model training.

### 4.2 Churn Labels

Cancellation-related events are used to identify churn.

- **Churn = 1:** The customer is labelled as churned.
- **Churn = 0:** The customer is labelled as non-churned.

### 4.3 Prediction Method

The model's sequence-building logic in `prepare_sequences.py` uses an outcome-anchored approach.

Each customer has one prediction point:

- For a churned customer, the prediction point is the first `Cancellation Confirmation` event.
- For a non-churned customer, the prediction point is the customer's last recorded activity.

The code defines:

```python
WINDOW, GAP = 30, 1
```

This means the model uses a 30-day observation window, ending one day before the prediction point. The gap is intended to reduce leakage from activity immediately associated with the prediction point.

The sequence is constructed from daily customer activity, with the oldest day first and the most recent observation day last.

**Important:** This describes the logic in `prepare_sequences.py`, which creates `sequence_data_v2.npz`. The separate `prepare_data.py` script creates `sequence_data.npz` and has its own cutoff logic. The model training workflow uses `sequence_data_v2.npz`.

### 4.4 Feature Engineering

The sequence-building code uses 13 daily activity features:

1. `event_count`
2. `sessions`
3. `songs`
4. `minutes_listened`
5. `thumbs_up`
6. `thumbs_down`
7. `add_to_playlist`
8. `add_friend`
9. `roll_advert`
10. `downgrade`
11. `upgrade`
12. `home`
13. `errors`

These features represent customer activity and engagement over time. Each customer's activity is represented as a sequence of 30 days, with 13 features per day.

### 4.5 Training, Validation, and Test Sets

The recorded output from `prepare_data.py` showed the following split:

| Dataset | Customers | Churned customers |
|---|---:|---:|
| Training | 157 | Not confirmed for the model's v2 data |
| Validation | 34 | Not confirmed for the model's v2 data |
| Testing | 34 | Not confirmed for the model's v2 data |

These figures belong to the recorded `prepare_data.py` output. They should not be treated as confirmed counts for `sequence_data_v2.npz` until `prepare_sequences.py` is run and its output is checked.

The training set is used to fit the models, the validation set is used for model selection and tuning, and the test set is used for final evaluation.

## 5. Models Used

### 5.1 Logistic Regression

Logistic Regression is used as a baseline model. It estimates the probability that a customer will churn. Comparing the sequence models with this baseline helps determine whether learning from sequences provides an advantage.

### 5.2 Simple Recurrent Neural Network (RNN)

A Simple RNN processes information sequentially and maintains a hidden state representing information from previous time steps. It is used to learn patterns in customer activity over the observation window.

### 5.3 Gated Recurrent Unit (GRU)

A GRU uses gates to control how information is retained or updated. It can learn temporal patterns while using a simpler structure than an LSTM.

### 5.4 Long Short-Term Memory (LSTM)

An LSTM uses memory cells and gates to learn relationships across time steps. It is suitable for sequence data because customer behaviour from earlier days may help predict future churn.

## 6. Model Training and Selection

The models are trained using the prepared customer sequences and their churn labels.

The project explores different model configurations, including hidden-unit sizes, dropout rates, and learning rates. Validation performance is used to compare configurations before final test evaluation.

The previously recorded best LSTM configuration was:

| Parameter | Value |
|---|---|
| Hidden units | 16 |
| Dropout rate | 0.40 |
| Learning rate | 0.005 |
| Validation loss | 0.489 |
| Validation ROC-AUC | 0.759 |
| Classification threshold | 0.50 |

These are previously recorded results and should be checked against the current model outputs before submission.

## 7. Model Evaluation

The models are evaluated using several classification metrics.

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** The proportion of predicted churners who actually churned.
- **Recall:** The proportion of actual churners correctly identified.
- **F1-score:** A balance between precision and recall.
- **ROC-AUC:** Measures how well the model separates churned and non-churned customers across different thresholds.
- **Confusion matrix:** Shows true positives, true negatives, false positives, and false negatives.

### 7.1 Previously Recorded Test Results

The following table contains the previously recorded test results. Confirm that these match the latest `test_metrics.csv` generated from the model's current data.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.667 | 0.381 | 0.800 | 0.516 | 0.783 |
| Simple RNN | 0.489 | 0.276 | 0.800 | 0.410 | 0.603 |
| LSTM | 0.667 | 0.222 | 0.200 | 0.211 | 0.694 |

### 7.2 Previously Recorded Confusion Matrices

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 22 | 13 | 2 | 8 |
| Simple RNN | 14 | 21 | 2 | 8 |
| LSTM | 28 | 7 | 8 | 2 |

### 7.3 Interpretation of Results

In the recorded results, Logistic Regression achieved the strongest ROC-AUC and F1-score among the three models. The Simple RNN achieved high recall but produced many false-positive predictions. The LSTM produced fewer false positives but identified fewer actual churners at the selected threshold.

These results show that a more complex sequence model does not automatically perform better than a simpler baseline. The relatively small dataset and limited number of churned customers may also affect model performance.

The final conclusions should be based on the latest evaluation results generated using `sequence_data_v2.npz`.

## 8. Cross-Validation and Robustness

Cross-validation is used to assess how consistently the models perform across different data splits.

The previously recorded out-of-fold ROC-AUC results were:

| Model configuration | ROC-AUC |
|---|---:|
| GRU, 32 units, dropout 0.20, learning rate 0.005 | 0.531 |
| LSTM, 32 units, dropout 0.40, learning rate 0.005 | 0.612 |
| GRU, 16 units, dropout 0.40, learning rate 0.005 | 0.571 |
| Logistic Regression | 0.561 |

These results suggest that model performance varies across configurations. Cross-validation provides additional information about generalisation, while the held-out test set provides the final evaluation.

## 9. Identifying Customers at Risk of Churn

The project generates churn probabilities for customers and ranks them from highest to lowest estimated risk.

Previously recorded high-risk examples include:

| Customer ID | Predicted churn probability | Actual label |
|---|---:|---|
| 200011 | 0.985 | Churned |
| 113 | 0.809 | Non-churned |
| 300009 | 0.652 | Non-churned |
| 200010 | 0.651 | Non-churned |
| 54 | 0.637 | Churned |

These examples should be checked against the latest `at_risk_customers_top20.csv` or `test_churn_predictions.csv` before submission.

A high predicted probability means that the model estimates a higher risk of churn. It does not guarantee that the customer will leave. Businesses should use predictions as decision-support information rather than certain outcomes.

## 10. Business Applications

A business can use churn predictions to support customer retention in several ways:

1. **Early identification:** Identify customers who may be considering leaving.
2. **Targeted communication:** Contact high-risk customers with relevant messages or support.
3. **Personalised offers:** Provide suitable promotions or subscription options where appropriate.
4. **Improved customer experience:** Investigate whether reduced engagement is linked to customer dissatisfaction.
5. **Resource allocation:** Focus retention efforts on customers who need attention most.
6. **Continuous monitoring:** Track model performance as customer behaviour changes.

False positives should also be considered. If the model incorrectly flags a customer as high risk, the business may spend money on an unnecessary retention action. If the model misses an actual churner, the business may lose an opportunity to retain that customer.

## 11. Project Structure

The repository includes scripts for sequence construction, model training, cross-validation, evaluation, and report generation.

The main files include:

```text
MLA307-Customer-Churn-Prediction/
├── prepare_data.py
├── prepare_sequences.py
├── train_model.py
├── cv.py
├── evaluate_model.py
├── report_summary.py
├── requirements.txt
├── README.md
├── data_preparation/
│   ├── sequence_data.npz
│   ├── member1_report_summary.txt
│   ├── dataset_summary.json
│   ├── customer_metadata_and_splits.csv
│   ├── daily_sequences.csv
│   ├── churn_vs_nonchurn_activity.png
│   └── class_weights.json
└── model_results/
    ├── test_metrics.csv
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── roc_curves.png
    ├── training_curves.png
    ├── tuning_results.csv
    ├── seed_robustness_validation.csv
    ├── cv_auc.json
    ├── test_churn_predictions.csv
    ├── at_risk_customers_top20.csv
    └── pattern_plot.png
```

This is an overview of the expected project files and outputs based on the available information. Actual paths and filenames may differ depending on the current repository and which scripts have been run.

The training, cross-validation, evaluation, and report summary scripts load `sequence_data_v2.npz`, which is created by `prepare_sequences.py`. The separate `prepare_data.py` script creates `sequence_data.npz` and additional data-preparation outputs.

## 12. Installation and Setup

### Requirements

- Python version compatible with the dependencies in `requirements.txt`
- Git
- pip
- A code editor such as Visual Studio Code

### Step 1: Clone the repository

```bash
git clone https://github.com/03240393gcbs-creator/MLA307-Customer-Churn-Prediction.git
cd MLA307-Customer-Churn-Prediction
```

### Step 2: Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the dataset

Download the Mini Sparkify dataset from:

https://udacity-dsnd.s3.amazonaws.com/sparkify/mini_sparkify_event_data.json

Place the downloaded file in the location expected by the data preparation script. Check the dataset path in the Python code if necessary.

## 13. How to Run the Project

Run the following commands from the project root directory, after activating the virtual environment and placing the dataset in the expected location.

### Step 1: Prepare customer sequences

```bash
python prepare_sequences.py
```

This script creates the 30-day customer sequences and saves `sequence_data_v2.npz` for the model workflow.

### Step 2: Train the models

```bash
python train_model.py
```

This script trains the models using the prepared sequence data.

### Step 3: Run cross-validation

```bash
python cv.py
```

This script evaluates model performance across cross-validation splits.

### Step 4: Evaluate the models

```bash
python evaluate_model.py
```

This script generates the final evaluation metrics and related outputs.

### Step 5: Generate the report summary

```bash
python report_summary.py
```

This script generates a summary of the available model results.

**Note:** Run `prepare_data.py` separately only if you need its additional dataset summary, visualisations, or `sequence_data.npz` output. The model-loading code identified for this project uses `sequence_data_v2.npz`.

## 14. Output Files

The model workflow is expected to produce the following outputs, depending on the scripts and configuration:

| Output file | Purpose |
|---|---|
| `test_metrics.csv` | Test-set evaluation metrics |
| `classification_report.txt` | Classification performance summary |
| `confusion_matrix.csv` | Correct and incorrect predictions |
| `roc_curves.png` | ROC curves for model comparison |
| `training_curves.png` | Model training and validation curves |
| `tuning_results.csv` | Results from model configuration experiments |
| `seed_robustness_validation.csv` | Validation results across random seeds |
| `cv_auc.json` | Cross-validation ROC-AUC results |
| `test_churn_predictions.csv` | Test customers and predicted churn information |
| `at_risk_customers_top20.csv` | Highest-risk customers |
| `pattern_plot.png` | Visualisation of customer activity patterns |

These filenames should be checked against the actual files generated by the current scripts.

## 15. Limitations

This project has several limitations:

- The Mini Sparkify dataset is a sample dataset and may not represent every real-world business.
- The number of customers and churned cases is limited.
- The dataset contains fewer churned customers than non-churned customers.
- Model results can change depending on the data split, model configuration, and decision threshold.
- A predicted probability is not a guarantee of future customer behaviour.
- The outcome-anchored method uses different prediction points for churned and non-churned customers. Results should be interpreted within this defined experimental setup.
- Further testing on additional data is required before using the models for real business decisions.

## 16. Future Improvements

Possible improvements include:

- Collecting more customer activity data.
- Testing additional sequence lengths and feature combinations.
- Addressing class imbalance carefully using the training data only.
- Comparing more model configurations and decision thresholds.
- Evaluating precision-recall curves alongside ROC-AUC.
- Testing a prospective prediction setup in which all customers are evaluated at a common prediction date.
- Improving model interpretability so businesses can understand the behaviour associated with high churn risk.

## 17. Conclusion

This project investigates how customer activity sequences can be used to predict customer churn. It compares Logistic Regression with Simple RNN, GRU, and LSTM approaches and evaluates their performance using classification metrics and cross-validation.

The previously recorded results show that the Logistic Regression baseline performed better than the tested Simple RNN and LSTM configurations on several test metrics. This demonstrates the importance of evaluating models carefully instead of assuming that a more complex neural network will always produce better predictions.

The project also shows how churn probabilities can help identify customers who may need attention. With further improvements in data quality, evaluation, and validation, sequence-based churn prediction can support more informed customer-retention decisions.

## 18. References

Udacity. (n.d.). *Mini Sparkify event data* [Data set]. https://udacity-dsnd.s3.amazonaws.com/sparkify/mini_sparkify_event_data.json

GitHub repository: https://github.com/03240393gcbs-creator/MLA307-Customer-Churn-Prediction
