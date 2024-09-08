import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np


# Set the random seed for reproducibility
np.random.seed(42)

# Number of clients
num_clients = 208

# Generate synthetic data
data = {
    'Client_ID': range(1, num_clients + 1),
    'Years_as_Client': np.random.randint(1, 30, num_clients),  # Between 1 and 30 years
    'Credit_Score': np.random.randint(300, 851, num_clients),  # Between 300 and 850
    'Num_Products': np.random.randint(1, 6, num_clients),  # Between 1 and 5 products
    'Annual_Income': np.random.randint(30000, 150001, num_clients),  # Between $30,000 and $150,000
    'Loan_Amount': np.random.randint(1000, 100001, num_clients),  # Between $1,000 and $100,000
    'Loan_Duration_Years': np.random.randint(1, 20, num_clients),  # Loan duration between 1 and 20 years
    'Missed_Payments': np.random.randint(0, 6, num_clients),  # Between 0 and 5 missed payments
    'Credit_Eligibility': np.random.randint(0, 2, num_clients)  # 0 or 1 for eligibility
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('client_credit_data.csv', index=False)

# Load dataset
data = pd.read_csv('client_credit_data.csv')

# Preprocess and select variables
# (data cleaning and feature selection code here)

# Train-test split
X = data.drop('credit_eligibility', axis=1)
y = data['credit_eligibility']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict and evaluate
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'ROC AUC Score: {roc_auc}')