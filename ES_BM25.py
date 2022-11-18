import numpy as np
import pandas as pd
import json
import os
import string
import re
from nltk.corpus import stopwords
from collections import Counter
from elasticsearch import Elasticsearch, helpers
import Load_dataset as db
from evaluate import load_type_hierarchy, get_type_path
from evaluate import load_ground_truth, load_system_output, evaluate
import warnings
warnings.filterwarnings("ignore")
import evaluate as ev
es = Elasticsearch()
###baseline retrieval using elasticsearch's built-in BM25 index
###

INDEX_NAME = 'questions'
stop_words = stopwords.words('english')

diff_ques_types = [ 'what', 'why', 'where','who', 'which','when', 'whom', 'whose']
stop_words = [word for word in stop_words if word not in diff_ques_types]

def textpreprocessing(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    text = re.sub(' +', ' ', text)
    word_list = [word for word in text.split() if word not in stop_words]
    text = " ".join(word_list)
    return text




##queries from json file
def load_queries(docs):
 
    resource_queries = {}
    count = 0
    for x in docs:
        if x['category'] != 'resource':
            count += 1
            continue
        
        if x['question'] is not None:
            q = textpreprocessing(x['question'])
        
            doc = {
                'question': q,
                'category': x['category'],
                'type': x['type']
            } 
            resource_queries.update({x['id']:doc})
        
    return resource_queries
##builtin bm25 from ES
def baseline_retrieval(es, query:str, field = 'abstract', index = INDEX_NAME):

    hits = es.search(index=index, size=200,
                query = {"bool": {"must": {"match": 
                                          {"abstract": query}}, 
                                 "must_not": {"match": 
                                            {"instance": "owl:Thing"}
                                            }}})['hits']['hits']
    hit_ids = [obj['_id'] for obj in hits]
    hit_types = [es.get(index=index, id=doc)["_source"].get("instance") for doc in hit_ids]
    result = [h[0] for h in Counter(hit_types).most_common(10)]
    
    return result


def es_BM25(es, data, index=INDEX_NAME):
    ##BM25 is evaluated for test inquiries
    ##a dictionary of test queries comprising the query id, question, and category as a arguments
    ##The query id, category, and anticipated kinds are all stored in a Dictionary.
    results = {} 
    for query_id, query in data.items():
        if len(query['question'])>0:
            response = baseline_retrieval(es, query['question'],  
                                        field = 'abstract', index=index)
            results.update({query_id:{
                                "id": query_id,
                                "category": query["category"],
                                "type": response
                                }
                            })
        else:
            continue
    return results



#if __name__ == "__main__":
es = Elasticsearch()
test = db.load_data('Results/system_output_json_svm.json')
test_queries = load_queries(test)
test_results = es_BM25(es, test_queries, index=INDEX_NAME)
   
f = open(f"Results/ES_BM25_system_output_json.json", "w")
json.dump(test_results, f)
f.close()
