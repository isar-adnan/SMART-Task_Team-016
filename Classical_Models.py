#======================================================================
# The classical machine learning models
# we use SVM, NB models and uses TF-IDF as a Text vectorization
# use pipeline which make textvectorization, create Machine learning model
# Train the model, Predict the categories in the testing dataset and Calculate the accuracy
#======================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import Load_dataset as data


# Train the model on SVM classifier, and predicts on the testing dataset
# Return the predicted values and the accuracy
def svmModel(df_train: pd.DataFrame, df_test: pd.DataFrame):

    pipeline_clf = Pipeline(
        [("TextVectorization", TfidfVectorizer()), ("model", SVC(kernel="linear"))]
    )

    pipeline_clf = pipeline_clf.fit(df_train["question"], df_train["updated_category"])
    predicted_svm = pipeline_clf.predict(df_test["question"])
    Accuracy = metrics.accuracy_score(df_test["updated_category"], predicted_svm)
    return predicted_svm, Accuracy


# Train the model on naive_bayes classifier, and predicts on the testing dataset
# Return the predicted values and the accuracy
def naive_bayes(df_train: pd.DataFrame, df_test: pd.DataFrame):
    pipeline_clf = Pipeline(
        [
            ("TextVectorization", TfidfVectorizer()), ("Model", MultinomialNB(alpha=1e-1)),
        ]
    )
    pipeline_clf = pipeline_clf.fit(df_train["question"], df_train["updated_category"])
    predicted_nb = pipeline_clf.predict(df_test["question"])
    Accuracy = metrics.accuracy_score(df_test["updated_category"], predicted_nb)
    return predicted_nb, Accuracy



