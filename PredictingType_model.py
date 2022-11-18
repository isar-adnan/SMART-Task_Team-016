import pandas as pd
import numpy as np
import json
import os
import Load_dataset as db
import Classical_Models as md
from rank_bm25 import BM25Okapi


# Create dictionary for the resource index consists of (key: all types, values: all questions related to that type)
def Create_ResourceIndexDict(df: pd.DataFrame):

    index = 0
    resource_index = {}
    typesList = list(df['type'])
    for typelst in typesList:
        for term in typelst:
            if term in resource_index.keys():
                resource_index[term] = resource_index[term] + ' ' + df.iloc[index]['question'].lower()
            else:
                resource_index[term] = df.iloc[index]['question'].lower()
        index = index + 1
    return resource_index

# create instance of bm25 which reads in a corpus of text and does some indexing on it
def train_on_BM25(corpus):
    tokenized_corpus  = [s.split() for s in corpus] 
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Get type from (the keys in the resource Index dictionary) when the ranked value match the values in the dictionary
def get_Type(index, Rankedval):
    for key, value in index.items(): 
        if Rankedval == value:
            return key

# we give the query (the question) and see which documents are the most relevant = k
def Predict_types(bm25, train_indexs_dic, corpus, query, k):
    # take the query (question) tokanization
    tokenized_query = query.split()
   # get best scores
    doc_scores = bm25.get_scores(tokenized_query)
    #get top relevant doc 
    best_RankedDocs = bm25.get_top_n(tokenized_query, corpus, n=k)
    PredictedTypes = []
    for rankedVal in best_RankedDocs:
        # get type for the ranked values
        PredictedTypes.append(get_Type(train_indexs_dic, rankedVal))
    return PredictedTypes


# Generate dictionary which consists of all columns in the testing dataset with the new predicted categories and the new predicted type)
# Remove old categories and old types
def Generate_FinalPredicton(df_data, results: dict) -> dict:

    final_pred = df_data.to_dict('records')
    for value in final_pred:
        if value['predicted_category'] == 'string':
            value['result_category'] = 'literal'
            value["predictedType"] = ['string']

        if value['predicted_category'] == 'number':
            value['result_category'] = 'literal'
            value["predictedType"] = ['number']
            
        if value['predicted_category'] == 'date':
            value['result_category'] = 'literal'
            value["predictedType"] = ['date']

        if value['predicted_category'] == 'boolean':
            value['result_category'] = 'boolean'
            value["predictedType"] = ['boolean']

        if value['predicted_category'] == 'resource':
            value['result_category'] = 'resource'
            value['predictedType'] = results[value['question']]
        
        del value['category']
        del value['type']
        del value['updated_category']
        del value['predicted_category']

        value['type'] = value.pop('predictedType')
        value['category'] = value.pop('result_category')
        
    return final_pred



