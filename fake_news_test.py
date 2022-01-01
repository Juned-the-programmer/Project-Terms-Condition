import pandas as pd
import re
import string

# Load the model
from joblib import load

saved_db = load("news_model/model_db") 
saved_adb = load("news_model/model_adb")
saved_GBC = load("news_model/model_GBC")
saved_LR = load("news_model/model_LR")
saved_RFC = load("news_model/model_RFC")
saved_svm = load("news_model/model_svm")

saved_vectors = load("model_vectorization")
# saved_model = pickle.load(open('model_DT', 'rb'))

# Convert text in to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer(analyzer='word',stop_words= 'english')

# Model Testing with Manual Entry
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

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = saved_vectors.transform(new_x_test)
    pred_DT = saved_db.predict(new_xv_test)
    pred_ADB = saved_adb.predict(new_xv_test)
    pred_GBC = saved_GBC.predict(new_xv_test)
    pred_LR = saved_LR.predict(new_xv_test)
    pred_RFC = saved_RFC.predict(new_xv_test)
    pred_SVM = saved_svm.predict(new_xv_test)

    return print("\n\n DT Prediction : {}\n ADB Prediction : {}\n GBC Prediction : {}\n LR Prediction : {}\n RFC Prediction : {}\n SVM Prediction : {} ".format(output_label(pred_DT[0]),
                                                                                                                                                                output_label(pred_ADB[0]),
                                                                                                                                                                output_label(pred_GBC[0]),
                                                                                                                                                                output_label(pred_LR[0]),
                                                                                                                                                                output_label(pred_RFC[0]),
                                                                                                                                                                output_label(pred_SVM[0])))

news = str(input("Enter the News for testing := "))
manual_testing(news)