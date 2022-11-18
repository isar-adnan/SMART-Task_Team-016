# SMART-Task_Team-016

# SMART_TASK
## Task Description & Dataset 
Add the data to folder data like `/SMART-Task_Team-016/data/`

## Steps to Run File:
Clone the repository and add the files in the project folder like `/SMART-Task_Team-016/`

## Baseline1
- Run `main.py` file, it will run the following steps:
1. Load the dataset.
2. Preprocessing (remove null values, change text to lower case, text vectorization)
3. Category prediction: It will print the accuracy of all two models (SVM, Naive Bayes).
4. Type Prediction: Generate json file for with the predicted values  for evalution.
5.Ealasticsearch indexing.  ‘ES_Indexing.ipynb’
6.Ealasticsearch using builtin bm25. ‘ES_BM25.py’
7.Ealsticsearch advanced approach. ‘ES_Advanceapproach.py’
8.Ealsticsearch advanced method implemented using transformer-bert, ‘ES_Bert.py’


### Note:
We trained all the models on dbpedia dataset.
