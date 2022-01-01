import pandas as pd
import numpy as np 
import string
import re
import pickle
import matplotlib.pyplot as plt

# Importing the dataset for both fake and real news
df_fake = pd.read_csv("DataSet/Fake.csv")
df_true = pd.read_csv("DataSet/True.csv")

# Importing the class columns
df_fake["class"] = 0
df_true["class"] = 1

# Removing 10 rows from both for manual testing
# df_fake_manual_testing = df_fake.tail(10)
# for i in range(23480,23470,-1):
#     df_fake.drop([i],axis=0,inplace=True)
# df_true_manual_testing = df_true.tail(10)
# for i in range(21416,21406,-1):
#     df_true.drop([i],axis=0,inplace=True)

# df_fake_manual_testing["class"] = 0
# df_true_manual_testing["class"] = 1

# df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing],axis=0)
# df_manual_testing.to_csv("manual_testing.csv")

# Mergin the main fake and true columns
df_merge = pd.concat([df_fake,df_true],axis=0)

# Drop unwanted Columns
df = df_merge.drop(["title","subject","date"],axis=1    )
print(df.isnull().sum())

# Randomly Shuffle the dataframe
df = df.sample(frac=1)

df.reset_index(inplace=True)
df.drop(["index"],axis=1,inplace=True)

# Creating a function to convert the text in to lowercase

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df["text"] = df["text"].apply(wordopt)

# Defining the independent variable and dependent variable
x = df["text"]
y = df["class"]

# Train test split the data
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25)

# Convert text to the vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from joblib import dump 
filename = 'model_vectorization.sav'
pickle.dump(vectorization , open(filename, 'wb'))
# dump(vectorization , "model_vectorization")

# Logistic Regression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train , y_train)

y_pred_lr = LR.predict(xv_test)
print(accuracy_score(y_test , y_pred_lr))
print(classification_report(y_test,y_pred_lr))
print(f1_score(y_test , y_pred_lr))

# filename = 'model_LR.sav'
# pickle.dump(LR, open(filename, 'wb'))
# # dump(LR , "model_LR")

# # Decision tree classifier
# from sklearn.tree import DecisionTreeClassifier
# DT = DecisionTreeClassifier()
# DT.fit(xv_train , y_train)

# y_pred_dt = DT.predict(xv_test)
# print(accuracy_score(y_test , y_pred_dt))
# print(classification_report(y_test , y_pred_dt))

# filename = 'model_db.sav'
# pickle.dump(DT, open(filename, 'wb'))
# # dump(DT , "model_db")

# # Gradient Boosting Classifier
# from sklearn.ensemble import GradientBoostingClassifier
# GBC = GradientBoostingClassifier()
# GBC.fit(xv_train , y_train)

# y_pred_gbc = GBC.predict(xv_test)
# print(accuracy_score(y_test , y_pred_gbc))
# print(classification_report(y_test , y_pred_dt))

# filename = 'model_GBC.sav'
# pickle.dump(GBC , open(filename, 'wb'))
# # dump(GBC , "model_GBC")

# # Random Forest Classisifier
# from sklearn.ensemble import RandomForestClassifier
# RFC = RandomForestClassifier()
# RFC.fit(xv_train , y_train)

# y_pred_rfc = RFC.predict(xv_test)
# print(accuracy_score(y_test , y_pred_rfc))
# print(classification_report(y_test , y_pred_rfc))

# filename = 'model_RFC.sav'
# pickle.dump(RFC , open(filename, 'wb'))
# # dump(RFC , "model_RFC")

# # Gaussian Naive Bayes
# # from sklearn.naive_bayes import GaussianNB
# # GNB = GaussianNB()
# # GNB.fit(xv_train , y_train)

# # y_pred_gnb = GNB.predict(xv_test)
# # print(accuracy_score(y_test , y_pred_rfc))
# # print(classification_report(y_test , y_pred_rfc))

# # dump(GNB , "model_gnb")

# # Adaboost 
# from sklearn.ensemble import AdaBoostClassifier
# ADB = AdaBoostClassifier()
# ADB.fit(xv_train,y_train)

# y_pred_adb = ADB.predict(xv_test)
# print(accuracy_score(y_test,y_pred_adb))
# print(classification_report(y_test, y_pred_adb))

# filename = 'model_adb.sav'
# pickle.dump(ADB , open(filename, 'wb'))
# # dump(ADB, "model_adb")

# # SVM
# from sklearn.svm import SVC
# SVM = SVC()
# SVM.fit(xv_train , y_train)

# y_pred_svm = SVM.predict(xv_test)
# print(accuracy_score(y_test , y_pred_svm))
# print(classification_report(y_test, y_pred_svm))

# filename = 'model_svm.sav'
# pickle.dump(SVM , open(filename, 'wb'))
# dump(SVM , "model_svm")