import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for splitting dataset
from sklearn.model_selection import train_test_split

# for feature scaling
from sklearn.preprocessing import StandardScaler

# for classifiler
from sklearn.linear_model import LogisticRegression

# for confusion matrix and scoring
from sklearn.metrics import confusion_matrix, accuracy_score


def LR():
    # import data
    dataset = pd.read_csv('Social_Network_Ads.csv')
    # print(dataset)

    ''' Results
      Age  EstimatedSalary  Purchased
0     19            19000          0
    [400 rows x 3 columns]
    '''

    # independent variables
    independent = dataset.iloc[:, :-1].values

    # dependent variable
    dependent = dataset.iloc[:, -1].values

    # print(independent, dependent)

    # splitting dataset into 4 parts for training
    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, train_size=0.8, random_state=1)

    # feature scaling for better results with Logistic Regression
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # fit/train dataset
    classifiler = LogisticRegression(random_state=1) # not necessary to have random_state
    classifiler.fit(x_train, y_train)

    # prediction a purchasing result by 30 years and 80000 income
    # purchase is 1, NOT is 0
    if int(classifiler.predict(sc.transform([[30, 80000]]))) == 1: # need sc.transform() because I transformed the original for the feature scaling
        print('The person will buy SUV')
    else:
        print('The person will NOT buy SUV')

    # confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=classifiler.predict(x_test))
    print(cm)

    # get the accuracy of our prediction
    print('The correctness is %.2f percent ' %accuracy_score(y_true=y_test, y_pred=classifiler.predict(x_test)))

if __name__ == '__main__':
    LR()
