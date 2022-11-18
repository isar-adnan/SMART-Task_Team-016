#=============================================
# import library
#=============================================
import json
import Load_dataset as db
import Classical_Models as md
import PredictingType_model as pt
import evaluate as ev
from elasticsearch import Elasticsearch
es = Elasticsearch()
from ES_BM25 import es_BM25
from ES_Advanceapproach import create_query_terms,ltr_featurevectors,ltr_prediction



if __name__ == "__main__":
    # loading DataSet
    print("="*100)
    print("Loading data: Training & Testing sets...")
    print("="*100)
    df_train, df_test = db.get_dataframe()
    df_train = db.UpdateCategroy(df_train)
    df_test = db.UpdateCategroy(df_test)
    print("Training set shape:", df_train.shape, "Testing set shape:", df_test.shape)
    

    print("="*100)
    print("Predicting category...")
    print("="*100)
    # Applying SVM for Category prediction
    predicted_svm, Accuracy = md.svmModel(df_train, df_test)
    print("Accuracy of SVM obtained is :", Accuracy.round(2))
    print("="*100)

     # Applying NB for Category prediction also
    predicted_NB, Accuracy = md.naive_bayes(df_train, df_test)
    print("Accuracy of NB obtained is :", Accuracy.round(2))
    print("="*100)

    print("="*100)
    print("Add the predicted category to the test dataset...")
    print("="*100)
    # we take the predicted cateory fron  the SVM classifier which has a best accuracy for the predicted_category
    df_test['predicted_category'] = predicted_svm

    # Use only the resource category for predicting type
    df_train_resource = df_train.loc[df_train['category'] == 'resource']
    df_test_resource = df_test.loc[df_test['predicted_category'] == 'resource']

    print("="*100)
    print("Creating the ResourceIndexing")
    print("="*100)
    # Create dictionary for the resource index consists of (key: all types, values: all questions related to that type)
    trainedResourceIndex = pt.Create_ResourceIndexDict(df_train_resource)
    corpus = list(trainedResourceIndex.values())

    # create instance of bm25 which reads in a corpus of text and does some indexing on it
    bm25 = pt.train_on_BM25(corpus)

    print("="*100)
    print("Use Bm25 to get top relevant for each query in the testing set...")
    print("="*100)
    # getting prediction from bm25 model and storing first k for top relevant types
    # save each query(question) as a key in the dictionay and the top relevant types as values for that key
    Prediction_results = {}
    for question in list(df_test_resource['question']):
        Prediction_results[question] = pt.Predict_types(bm25, trainedResourceIndex, corpus, question, k = 10)

    print("="*100)
    print("Generate Prediction file...")
    print("="*100)
    # Generate dictionary which consists of all columns in the testing dataset with the new predicted categories and the new predicted type)
    Final_prediction = pt.Generate_FinalPredicton(df_test, results=Prediction_results)

    # write the results in Final_prediction to the json file to prepare it for evalution
    f = open(f"Results/system_output_json_svm.json", "w")
    json.dump(Final_prediction, f)
    f.close()
    
    type_hierarchy, max_depth = ev.load_type_hierarchy('Evaluation/dbpedia_types.tsv')
    ground_truth = ev.load_ground_truth('data/smarttask_dbpedia_test.json', type_hierarchy)
    system_output = ev.load_system_output('Results/system_output_json_svm.json')
    ev.evaluate(system_output, ground_truth, type_hierarchy, max_depth)
    print("="*100)
    
    type_hierarchy, max_depth = ev.load_type_hierarchy('Evaluation/dbpedia_types.tsv')
    ground_truth = ev.load_ground_truth('data/smarttask_dbpedia_test.json', type_hierarchy)
    system_output = ev.load_system_output('Results/ES_BM25_system_output_json.json')
    ev.evaluate(system_output, ground_truth, type_hierarchy, max_depth)
    