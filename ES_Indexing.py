import traceback
import elasticsearch
import time
from elasticsearch import Elasticsearch, helpers
import nltk
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings("ignore")

es = Elasticsearch()
es.info()

FIELDS = ['abstract', 'instance']
INDEX_NAME = 'dbpedia'
INDEX_SETTINGS = {
    'mappings': {
            'properties': {
                'abstract': {
                    'type': 'text',
                    'term_vector': 'yes',
                    'analyzer': 'english'
                },
                'instance': {
                    'type': 'text',
                    'term_vector': 'yes',
                    'analyzer': 'english'
                }
            }
        }
    }

def generate_esindex():
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)    
    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)



def read_ttlfile(filename, size, enc='utf-8'):
   
    if size <= 0:
        print("Bigger size")
        return

    with open(filename, encoding=enc) as file:
        for i,line in enumerate(file):
            if (size >= 0) and (i >= size+1):
                break
            if i == 0: 
                continue
            print(line.strip())


stop_words = stopwords.words('english')

typesof_question=  ['what', 'why', 'where','who', 'which','when', 'whom', 'whose']
stop_words = [word for word in stop_words if word not in typesof_question]


def text_preprocessing(text):
    text = re.sub(' +', ' ', text)
    stop_words = stopwords.words('english')
    lst = [word for word in text.split() if word not in stop_words]
    text = " ".join(lst)
    return text


def abstracts_preprocessing(text):
    list = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)
    abstract = re.sub('[^a-z]', ' ', text)[0]
    #stripedspaces = re.sub('\s+', ' ', onlychars)
    #page = re.findall('<.*?>', text) 
    #abstract = re.findall('\".*?\"', text)[0]
    #abstract = re.sub(r'[^a-zA-Z0-9\s]', ' ', abstract).lower()
    if len(abstract)>2:
        abstract=abstract[1:-1]
        abstract = text_preprocessing(abstract)
    entity= list[0].split('/')[-1] 
    entity= entity[:-1].replace('_', ' ')
    return entity, abstract


def entitytype(text):
    list = re.findall('<.*?>', text) 
    entity = list[0].split('/')[-1][:-1].replace('_', ' ')
    entity_type = list[-1].split('/')[-1][:-1].replace('owl#', '').replace('_', ' ')
    if entity_type=='Thing':
        entity_type='owl:Thing'
    else:
        entity_type='dbo:'+entity_type

    return entity, entity_type


def category(text):
    list = re.findall('<.*?>', text) 
    entity = list[0].split('/')[-1][:-1]
    category = list[-1].split('/')[-1][:-1]
    category = category.replace('Category:','').replace('_', '')
    return entity.replace('_', ' '), category

size = 1000

def parse_abstracts(data):   

    with open(file="data\short_abstracts_en.ttl", encoding='utf-8') as file:
        for i,line in enumerate(file):
            if (size >= 0) and (i >= size+1):
                break
            if i == 0: 
                continue
            entity, abstract = abstracts_preprocessing(line)
            if len(abstract)>0 and len(entity)>0:
                data.update({
                        entity:{ 
                        "_id": entity,
                        "_source": {
                            "abstract": abstract,
                            "instance":'instance_placeholder'}}
                        })
        return list(data.keys())       

def parser_entity_type(data):
    lsentity = []
    with open(file="data\instance_types_en.ttl", encoding='utf-8') as file:
        for i,line in enumerate(file):
            if (size >= 0) and (i >= size+1):
                break
            if i == 0: 
                continue
            entity, entity_type = entitytype(line)
            if len(entity_type)>0 and len(entity)>0:
                lsentity.append(entity)
            try:
                data[entity]['instance']=entity_type
            except:
                pass
            


def entity_typeremoval(data):
    items = []
    for i,j in data.items():
        if len(data[i]["instance"])==0:
            items.append(i)
    for z in items:
        data.pop(z)
    print("After removal: ", len(data))





if __name__ == "__main__":
    #Create elastic search
    generate_esindex()
    # read from TTL file
    read_ttlfile("data/short_abstracts_en.ttl",2)
    
    read_ttlfile("data/instance_types_en.ttl", 2)
    
    data = {}

    parse_abstracts(data)
    parser_entity_type(data)
    print("-"*100)
    print("data")
    print(data)
    print("-"*100)
    entity_typeremoval(data)
    print("-"*100)
    list(data.values())
    
    batch_size = 100
    doc = list(data.values())
    for i in range(0, len(data), batch_size):
        actions = [{
            "_index": INDEX_NAME,
            "_id": x["_id"],
            "_source": {
                "abstract": x["abstract"],
                "instance": x["instance"]
            }
        } for x in doc[i:i+batch_size]]
        helpers.bulk(es, actions, index=INDEX_NAME, raise_on_error=False, raise_on_exception=False)
    print("-"*100)
    
    search_param={"match": {"instance": "owl:Thing"}}
    print("-"*100)
    response = es.search(index=INDEX_NAME, query=search_param)
    #print('Files matched', response['hits']['total']['value'])
    
    #response['hits']['hits'][0]['_source']['instance']