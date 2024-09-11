
![Python application](https://github.com/afbeltranr/ml-credit-elegibility/workflows/Python%20application/badge.svg)


# Credit Eligibility Prediction using Logistic Regression

This project demonstrates the process of predicting whether a bank client is eligible for credit approval. We use a Logistic Regression model to handle binary outcomes such as "approved" or "not approved".

You can view the full analysis in the [Jupyter Notebook](./credit_elegibility_logreg.ipynb).

## Methods
- **Data preprocessing**: Creation of suitable variables that are distributed considering the underlying economic mechanisms and real-world phenomena that govern how wealth, income, and financial behavior are distributed. Distribution assessment using visualization and detection of correlated variables to drop and avoid multicollinearity.
-  **Response variable creation**: using real-world criteria, like thresholds on credit score and annual income.
- **Logistic Regression**: Applied to predict the binary outcome (credit approval: yes or no) .
- **Evaluation Metrics**: Kolmogorov-Smirnov (KS) , AUC/GINI, PSI score to evaluate the model.

## Results

After training the Logistic Regression model, the following key metrics were obtained:

- **KS Statistic**: `0.988`
  - **Interpretation**: The model demonstrates excellent discriminatory power, with a KS value close to 1, indicating that it can effectively distinguish between clients who are eligible and not eligible for credit.

- **AUC**: `0.993`
  - **Interpretation**: The model performs very well, with an AUC of 0.993, indicating a high probability of correctly ranking eligible clients above non-eligible ones.

- **GINI Coefficient**: `0.986`
  - **Interpretation**: With a GINI coefficient of 0.986, the model has near-perfect discriminatory power. A GINI value close to 1 reflects a high level of accuracy in distinguishing between classes.

- **PSI**: `0.184`
  - **Interpretation**: The PSI indicates moderate population stability between the training and test sets. A PSI value below 0.25 suggests the model generalizes well, with no significant population drift.

### Conclusion:
The model demonstrates excellent performance across all key metrics, with strong discriminatory power (high KS, AUC, and GINI) and stable generalization (acceptable PSI). This indicates that the Logistic Regression model is highly effective in predicting credit eligibility.
