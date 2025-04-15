# Credit Card Fraud Detection using Logistic Regression
This project implements a Credit Card Fraud Detection system using a basic Logistic Regression model built from scratch with NumPy. It aims to classify transactions as either fraudulent or legitimate based on features like transaction amount, transaction time, and account age.

# 🧠 Model Overview
The model is trained using gradient descent optimization and uses log loss as the cost function. All features are normalized using Min-Max Scaling, and a bias term is included in the feature matrix.

Key steps include:

* Feature scaling

* Logistic regression implementation

* Gradient descent training

* Probability thresholding for classification

* Evaluation using accuracy, precision, and recall

# 📊 Dataset
The dataset used is fraud-data.csv with the following relevant features:

* Transaction_Amount

* Transaction_Time

* Account_Age

* Fraudulent (target label: 1 for fraud, 0 for non-fraud)

# 🧮 How It Works
1. Normalize Data
Min-Max Normalization is applied to bring all features to the same scale.

2. Model Initialization
A logistic regression model is built manually using NumPy with sigmoid activation and a custom loss function.

3. Training
The model is trained using gradient descent for a set number of iterations and a given learning rate.

4. Prediction
After training, the model can predict the probability of a transaction being fraudulent.

5. Evaluation Metrics

* Accuracy

* Precision

* Recall

# 📈 Output
The training process shows the cost function convergence. After training, the model provides:

* Fraud probability for a sample transaction

* Final prediction (fraud or not)

* Overall model performance metrics

# 🚀 Getting Started
To run the project:

1. Clone the repo

2. Make sure Python 3 and the required libraries (numpy, pandas, matplotlib) are installed.

3. Add your fraud-data.csv file in the root directory.

4. Run the Python script.

```bash
  jupyter notebook CreditCardFraud.ipynb
```
# 📌 Example Prediction

```python
  new_transaction = np.array([1, 0.8, 0.5, 0.6])
```

This line predicts whether the given normalized transaction values indicate fraud.

# 📉 Evaluation Metrics Example

```yaml
Model Accuracy: 0.96
Precision: 0.92
Recall: 0.88
```
