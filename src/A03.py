# Importing Necessary Libraries: 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Reading Data:
df = pd.read_csv("california_housing_train.csv")
# Creating Histogram:
df['median_house_value'].hist()
plt.title("Histogram of Median House Value")
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.savefig("figs/histogram_MHV.png")
plt.close()

# Viewing Imbalanced Data: 
df['median_house_value'] = np.where(df['median_house_value'] < 380000, 1, 0)
print(df['median_house_value'].value_counts())

# Splitting into Training and Testing Set: 
X = df.drop(columns= ['median_house_value']) 
Y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=120, stratify=Y)

# Making sure Y is Still Categorical: 
print(y_train.value_counts())
print(y_test.value_counts())

# Majority Undersampling: 

undersample = RandomUnderSampler(sampling_strategy='majority', random_state=120)

## Fitting only on Training Data: 
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

## Summarize class distributions
print(Counter(y_train_under))

## Apply Decision Tree Classification: 
DTC = DecisionTreeClassifier(min_samples_split=10, random_state=120)
DTC.fit(X_train_under, y_train_under)
train_preds = DTC.predict(X_train_under)
test_preds = DTC.predict(X_test)

## Model Evaluation: 
print('UnderSampling Train results: \n')
print(classification_report(y_train_under, train_preds))
print('UnderSampling Test results: \n')
print(classification_report(y_test, test_preds))

print("UnderSampling Confusion Matrix (Train Set): \n")
print(confusion_matrix(y_train_under, train_preds))
print("UnderSampling Confusion Matrix (Test Set): \n")
print(confusion_matrix(y_test, test_preds))


# Minority Oversampling: 

oversample = RandomOverSampler(sampling_strategy='minority')

## Fitting only on Training Data: 
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

## Summarize class distributions
print(Counter(y_train_over))

## Apply Decision Tree Classification: 
DTC.fit(X_train_over, y_train_over)
train_preds_over = DTC.predict(X_train_over)
test_preds_over = DTC.predict(X_test)

## Model Evaluation: 
print('OverSampling Train results: \n')
print(classification_report(y_train_over, train_preds_over))
print('OverSampling Test results: \n')
print(classification_report(y_test, test_preds_over))

print("OverSampling Confusion Matrix (Train Set): \n")
print(confusion_matrix(y_train_over, train_preds_over))
print("OverSampling Confusion Matrix (Test Set): \n")
print(confusion_matrix(y_test, test_preds_over))

# SMOTE:
sm = SMOTE(random_state=120)

## Fitting only on Training Data: 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

## Summarize class distributions
print(Counter(y_train_res))

## Apply Decision Tree Classification: 
DTC.fit(X_train_res, y_train_res)
train_preds_res = DTC.predict(X_train_res)
test_preds_res = DTC.predict(X_test)

## Model Evaluation: 
print('SMOTE Train results: \n')
print(classification_report(y_train_res, train_preds_res))
print('SMOTE Test results: \n')
print(classification_report(y_test, test_preds_res))

print("SMOTE Confusion Matrix (Train Set): \n")
print(confusion_matrix(y_train_res, train_preds_res))
print("SMOTE Confusion Matrix (Test Set): \n")
print(confusion_matrix(y_test, test_preds_res))

# Reproducibility: OverSampling Algorithm: 

## Keeping OverSample and DTC the Same:
oversample = RandomOverSampler(sampling_strategy='minority', random_state=120)
DTC = DecisionTreeClassifier(min_samples_split=10, random_state=120)

## Storing Metric Evaluations: 
acc_list = []
prec_list = []
rec_list = []


## For Loop
for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i, stratify=Y)

    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

    DTC.fit(X_train_over, y_train_over)

    y_pred = DTC.predict(X_test)

    acc_list.append(accuracy_score(y_test, y_pred))
    prec_list.append(precision_score(y_test, y_pred, zero_division=0))
    rec_list.append(recall_score(y_test, y_pred, zero_division=0))

## Storing Results:
results = pd.DataFrame({"accuracy": acc_list, "precision": prec_list, "recall": rec_list})

## Plotting Accuracy:
results['accuracy'].hist( bins = 10)
plt.title("Accuracy Across 30 Splits")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.savefig("figs/Accuracy30.png")
plt.close()

## Plotting Precision: 
results['precision'].hist(bins = 10)
plt.title("Precision Across 30 Splits")
plt.xlabel("Precision")
plt.ylabel("Count")
plt.savefig("figs/Precision30.png")
plt.close()

## Plotting Recall: 
results["recall"].hist(bins = 10)
plt.title("Recall Across 30 Splits")
plt.xlabel("Recall")
plt.ylabel("Count")
plt.savefig("figs/Recall30.png")
plt.close()


# Trying 100 Times to See Difference in Results:
oversample = RandomOverSampler(sampling_strategy='minority', random_state=120)
DTC = DecisionTreeClassifier(min_samples_split=10, random_state=120)

## Storing Metric Evaluations: 
acc_list = []
prec_list = []
rec_list = []


## For Loop
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i, stratify=Y)

    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

    DTC.fit(X_train_over, y_train_over)

    y_pred = DTC.predict(X_test)

    acc_list.append(accuracy_score(y_test, y_pred))
    prec_list.append(precision_score(y_test, y_pred, zero_division=0))
    rec_list.append(recall_score(y_test, y_pred, zero_division=0))

## Storing Results:
results = pd.DataFrame({"accuracy": acc_list, "precision": prec_list, "recall": rec_list})

## Plotting Accuracy:
results['accuracy'].hist( bins = 10)
plt.title("Accuracy Across 100 Splits")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.savefig("figs/Accuracy100.png")
plt.close()

## Plotting Precision: 
results['precision'].hist(bins = 10)
plt.title("Precision Across 100 Splits")
plt.xlabel("Precision")
plt.ylabel("Count")
plt.savefig("figs/Precision100.png")
plt.close()

## Plotting Recall: 
results["recall"].hist(bins = 10)
plt.title("Recall Across 100 Splits")
plt.xlabel("Recall")
plt.ylabel("Count")
plt.savefig("figs/Recall100.png")
plt.close()
