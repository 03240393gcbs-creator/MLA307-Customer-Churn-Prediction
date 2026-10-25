# MLA307-Customer-Churn-Prediction
# Customer Churn Prediction Using Sequence Models

## 1. Project Overview

This project investigates customer churn prediction using sequential user activity data from a music-streaming service. It compares recurrent neural network models with a traditional machine-learning baseline to evaluate how well customer behavior over time can predict churn.

**Course:** MLA307 – Sequence Models  
**Project Topic:** Topic 1 – Customer Churn Prediction

## 2. Objectives

- Prepare customer event data for sequential modelling.
- Construct 30-day customer activity sequences.
- Train and evaluate GRU, LSTM, and simple RNN models.
- Compare sequence models against a logistic regression baseline.
- Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Discuss the limitations of the predictions and their potential business applications.

## 3. Dataset

The project uses the Mini Sparkify music-streaming event dataset.

The data contains customer activity events, including listening behavior and account-related events. A customer is labelled as churned if they have a recorded `Cancellation Confirmation` event.

**Dataset source:** [Insert the original dataset download URL]

**Dataset characteristics:**
- Approximately 286,500 event records
- 225 customers
- 18 original attributes
- Observation period: October–December 2018

The dataset is processed into 30-day customer activity sequences with 13 daily features.

*Note: Confirm these statistics against the final processed dataset before publishing.*

## 4. Models

The following models are evaluated:

1. **Logistic Regression:** Baseline model.
2. **Simple RNN:** Recurrent neural network baseline.
3. **GRU:** Gated Recurrent Unit sequence model.
4. **LSTM:** Long Short-Term Memory sequence model.

The models are compared using consistent customer-level training, validation, and test splits wherever applicable.

## 5. Repository Structure

```text
MLA307-Customer-Churn-Prediction/
├── data_preparation/
├── model_results/
├── backup_before_evaluation/
├── prepare_data.py
├── prepare_sequences.py
├── train_model.py
├── cv.py
├── evaluate_model.py
├── README.md
└── .gitignore
```

- `prepare_data.py` — prepares and cleans the raw event data.
- `prepare_sequences.py` — constructs customer-level sequences and labels.
- `train_model.py` — trains the prediction models.
- `cv.py` — performs cross-validation and supports model/threshold evaluation.
- `evaluate_model.py` — evaluates the trained models and generates performance results.
- `data_preparation/` — contains relevant data-preparation files or outputs.
- `model_results/` — contains saved model outputs, metrics, and evaluation artefacts.

*Update these descriptions if the scripts perform different tasks.*

## 6. Installation and Requirements

Install Python 3 and the dependencies required by the project.

```bash
git clone https://github.com/03240393gcbs-creator/MLA307-Customer-Churn-Prediction.git
cd MLA307-Customer-Churn-Prediction
```

Create and activate a virtual environment if desired, then install the project's dependencies:

```bash
python -m venv .venv
```

Activate the environment and install dependencies using the project's dependency file, if provided:

```bash
pip install -r requirements.txt
```

If the repository does not yet contain `requirements.txt`, create one listing the actual packages and versions used by the code.

## 7. How to Run

Run the scripts in the correct order after placing the dataset in the expected location.

```bash
python prepare_data.py
python prepare_sequences.py
python train_model.py
python cv.py
python evaluate_model.py
```

Check each script's arguments and file paths before running these commands. Update this section if a script requires additional arguments or if the scripts must be run in a different order.

## 8. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Recall is important because failing to identify a customer who will churn may prevent a timely retention intervention. Precision is also important because unnecessary retention offers can increase business costs.

## 9. Results

Summarize the final test-set results in a table.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.647 | 0.375 | 0.750 | 0.500 | 0.736 |
| Simple RNN | 0.647 | 0.300 | 0.375 | 0.333 | 0.582 |
| GRU | 0.676 | 0.400 | 0.750 | 0.522 | 0.716 |
| LSTM | Add final test results | Add | Add | Add | Add |

The GRU results shown above use a classification threshold of 0.41, while the other results shown use a threshold of 0.50 except where otherwise specified. Report the threshold for each model clearly and ensure the table matches the final evaluation output.

## 10. Limitations

The dataset is relatively small, and model performance may be affected by differences in customer observation periods and the timing of recorded events. In particular, the presence of leading days with no recorded activity may introduce a timing-related signal rather than purely behavioral patterns.

The results should therefore be interpreted cautiously and validated on a larger dataset before deployment.

## 11. Business Recommendations

Potential applications include:

- Generating a weekly list of customers at risk of churn.
- Testing targeted retention offers through a controlled pilot.
- Monitoring recall, precision, and false-positive rates.
- Retraining and evaluating models as new customer activity becomes available.

## 12. Reproducibility

The scripts and model outputs are provided to support reproducibility. Record the software dependencies, random seeds, data splits, model configurations, and classification thresholds used to generate the final reported results.

## 13. Author

**Author:** [Your name or student identifier]  
**Course:** MLA307 – Sequence Models

