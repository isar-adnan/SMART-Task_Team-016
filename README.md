# SMART-Task_Team-016

# SMART_TASK

## Task Description & Dataset 
Add the data to folder data like `/SMART-Task_Team-016/data/`

## Steps to Run File:
Clone the repository and add the files in the project folder like `/SMART-Task_Team-016/`

## Baseline1
- Run `main.py` file, it will run the following steps:
1. Load the dataset.  `Load_dataset.py`
2. Preprocessing (remove null values, change text to lower case, text vectorization)
3. Category prediction: It will print the accuracy of all two models (SVM, Naive Bayes).  `Classical_Models.py`
4. Type Prediction: Generate json file for with the predicted values  for evalution.    `PredictingType_model.py`
5. Elasticsearch indexing.  ‘ES_Indexing.ipynb’
6. Elasticsearch using builtin bm25. ‘ES_BM25.py’
7. Elsticsearch advanced approach. ‘ES_Advanceapproach.py’
8. Elsticsearch advanced method implemented using transformer-bert, ‘ES_Bert.py’
9. For evaluation process use evaluate.py file


### Note:
We trained all the models on dbpedia dataset. The ttl files can be downloaded from http://downloads.dbpedia.org/2016-10/core/. once downloaded instance_types_en.ttl.bz2 and short_abstract_en.ttl.bz2 these files need to be moved under data folder. In order to extract bz2 files run the Compressing_bz2.py file to get ttl files. For further improvements we have implemented elasticsearch indexing in .ipynb file to understand the process and seeing the results because we faced challenges in .py file.

Installation Packages: pandas,numpy,scikit-learn,elasticsearch (download link: https://www.elastic.co/downloads/past-releases/elasticsearch-7-17-6). Need to write pip install ealasticsearch==7.17.6 in commond promt
