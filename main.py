import pandas as pd
import numpy as np
import itertools

# Improting the dataset
df = pd.read_csv("DataSet/fake_or_real_news.csv")
print(df.head())

print(df.shape)
print(df.isnull().sum())

labels = df.label
# print(labels)

# Train test Split of the data
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(df['text'],labels,test_size=0.2,random_state=20)

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

vector = TfidfVectorizer(stop_words='english',max_df=0.7)

# Fit and Transform
tf_train = vector.fit_transform(x_train)
tf_test = vector.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train,y_train)

# Predictiong the values 
from sklearn.metrics import accuracy_score,confusion_matrix
y_pred = pac.predict(tf_test)
print(y_pred)

print("\n Comparision of test and predicted data \n")
comparision = pd.DataFrame()
comparision['Actual'] = y_test
comparision['predicted'] = y_pred
print(comparision)

score = accuracy_score(y_test,y_pred)
print(f"Accuracy Score : {score} ")