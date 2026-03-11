**Loan Default Risk Classification**  
Brittany D’Erasmo

**Introduction**  
The purpose of this project was to build a classifier that can be used to predict whether a loan applicant is high-risk (likely to default on a loan) or low-risk (likely to successfully repay it), based on various financial, demographic, and behavioral features. Using Python libraries, two machine learning models were trained, tested and evaluated using historical loan data. The result provides insight to which model would perform better in real-world application by financial institutions to assess the risk of loan applicants.

**Dataset and Preprocessing**  
The dataset used for this project is the Loan Default Dataset found on [kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset), which contains 14,8670 observations. It contains applicant demographic and behavioral features, including gender, income, age, credit score and debt-to-income ratio; loan characteristics, such as loan amount, loan type, loan term and loan purpose; and property characteristics, like property value, occupancy type, number of units and region. In total, there are 34 categorical and numerical variables, including an ID column, as well as the target variable, Status, for which a value of 1 represents a defaulted loan, and a value of 0 represents a not defaulted loan.

Through exploratory data analysis, using Seaborn and Matplotlib, the following visualizations were generated:

* Bar plot to show the distribution of Status in the dataset  
* Kernel density plots to show the distribution of each numerical feature with skewness to show variance  
* Pair plot to show the relationship between pairs of variables  
* Box plots to show relationships between Status and some numerical features and check for outliers  
* Violin plots to show relationships between Status and some numerical features

Missing values in numerical columns were replaced with the median column value, and missing values in categorical columns were replaced with the mode column value. Categorical features were encoded and numerical features were scaled using scikit-learn. The ID and Year columns are (intuitively) unimportant and were dropped. The LTV (loan-to-value ratio) feature is derived from two other columns (loan amount divided by property value), and therefore not needed so it was dropped. Rate of interest, interest rate spread, and upfront charges are features that are typically based on, and determined after, risk assessment of the applicant, so including these could cause data leakage. In fact, when kept in the dataset for training, one of the models achieved 99% accuracy, a sign that the model might have cheated. Those columns were dropped. The dataset was split into 80% training set and 20% test set. 

**Observations**  
Using scikit-learn, Logistic Regression and Random Forest models were trained, tested and evaluated. Logistic Regression is a baseline statistical classification model that estimates the probability of loan default using coefficients. Random Forest is an ensemble model that uses multiple decision trees and captures nonlinear relationships. An attempt was made to implement Support Vector Machine as well, but after a few hours of runtime the model training was not successfully executed. This is likely due to the size of the dataset and the computational complexity of the SVM algorithm, so SVM was not included in the subsequent testing, evaluation and model comparison. 

The models were evaluated by accuracy, precision, recall and f1-score. The Random Forest model scored higher than Logistic Regression across all four metrics. With \~0.88 accuracy, \~0.92 precision, \~0.57 recall and \~0.70 f1-score, the Random Forest model correctly classified the most instances, resulted in the least false positives (precision) and least false negatives (recall), and had the better balance between precision and recall (F1-score). A confusion matrix for each shows that both models misclassified more defaulted loans than not defaulted loans, and that the Random Forest model correctly classified more instances than the Logistic Regression model.  
![image1](loan_default_classification/images/image1.png)
In real-word context, the misclassification of not default as default could lead to a financial loss for the loan provider. The misclassification of loan default as not default could lead to missed opportunities to offer a loan to some applicants that are wrongly identified as risky, leading to financial loss as well. The Random Forest model succeeds at both classifications more often than the Logistic Regression model. 

**Conclusion**  
This project demonstrates how supervised machine learning can be used to classify loan applicants as likely to default or not. Using historical data to train and test two models, it is apparent that Random Forest is superior for the task of assessing the risk of loan applicants.
