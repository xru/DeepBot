# -*- coding: utf-8 -*-
# Project : DeepBot
# Created by igor on 16/8/30
'''
Elasticsearch Setting and Initiation
'''

from elasticsearch import Elasticsearch
from engine.mdc_generator import utterance_generator

ES_HOST = {"host": "localhost", "port": 9200}
INDEX_NAME = "corpus"
TYPE_NAME = "chat"


def init(data_dir):
    bulk_data = []
    es = Elasticsearch(hosts=[ES_HOST])
    header = ["character1", "character2", "movie", "context", "utterance"]
    for row in utterance_generator(data_dir):
        data_dict = {}
        for i in range(len(row)):
            data_dict[header[i]] = row[i]

        op_dict = {
            "index": {
                "_index": INDEX_NAME,
                "_type": TYPE_NAME,
            }
        }

        bulk_data.append(op_dict)
        bulk_data.append(data_dict)

    if es.indices.exists(INDEX_NAME):
        print("deleting %s..." % (INDEX_NAME))
        res = es.indices.delete(index=INDEX_NAME)
        print(" response: %s" % (res))

    # setting
    request_body = {
        "settings": {
            "number_of_shards": 1,  # can't change,we use single machine so set 1
            "number_of_replicas": 0,
            "analysis": {

            }
        }
    }

    print("creating %s index..." % (INDEX_NAME))
    res = es.indices.create(index=INDEX_NAME, body=request_body, )
    print(" response: %s" % (res))

    # bulk index the data
    print("bulk indexing...")
    res = es.bulk(index=INDEX_NAME, body=bulk_data, refresh=True)

    # sanity check
    res = es.search(index=INDEX_NAME, size=2, body={"query": {"match_all": {}}})
    print(" response: '%s'" % (res))


if __name__ == '__main__':
    # data_dir = '../data/cmdc/'
    # init(data_dir)
    es = Elasticsearch(hosts=[ES_HOST])
    res = es.search(index=INDEX_NAME, size=2, body={"query": {"match_all": {}}})
    print(" response: '%s'" % (res))
