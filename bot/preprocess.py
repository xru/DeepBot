# -*- coding: utf-8 -*-
# Project : DeepBot
# Created by igor on 16/9/3
'''
Preporcess:
 - read raw data
 - build dictionary
 - load dictionary
 - update dictionary
 - generate train and test example with 80%-20%
'''
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from gensim import corpora
from nltk.tokenize import TweetTokenizer

from engine.mdc_generator import utterance_generator
from bot.config import *

tokenzier = TweetTokenizer()

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"

_START_VOCAB = [_PAD, _GO, _EOS, _UNK]


def cmd_cleaned_generator():
    pattern = re.compile(r"[^a-zA-Z1-9\.!?,' ]+")
    sub_EOS = re.compile(r"[\.?!]+")
    for context, utterance in utterance_generator(MOVIE_CORPUS_DIR):
        context = pattern.sub("", context)
        context = sub_EOS.sub(" " + _EOS, context)
        context = tokenzier.tokenize(context)

        utterance = pattern.sub("", utterance)
        utterance = sub_EOS.sub(" " + _EOS, utterance)
        utterance = tokenzier.tokenize(utterance)

        yield context, utterance


def load_dictionary(model_file_path):
    '''
    加载字典
    :param model_file_path:
    :return:
    '''
    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            dictionary = pickle.load(f)
        f.close()
        print("Load dictionary from %s" % model_file_path)
        return dictionary
    else:
        raise FileExistsError


def build_dictionary(generator, min_freq=5):
    dictionary_path = os.path.join(DATA_PATH, DICT_NAME)

    if os.path.exists(dictionary_path) and os.path.isfile(dictionary_path):
        print("Delete dictionary and rebuild")
        os.remove(dictionary_path)

    dictionary = corpora.Dictionary(c + u for c, u in generator)

    # 去除低频的ID
    filter_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if
                  docfreq < min_freq]

    dictionary.filter_tokens(filter_ids)
    dictionary.compactify()

    dictionary.add_documents([_START_VOCAB])

    pickle.dump(dictionary, open(dictionary_path, 'wb'))
    print("SVAE dictionary to %s" % (dictionary_path))

    return dictionary


def map_to_id(tokens, token2id, max_length):
    UNK_ID = token2id[_UNK]
    PAD_ID = token2id[_PAD]
    vec = [token2id.get(i, UNK_ID) for i in tokens]
    pad_length = max_length - len(vec)
    vec = np.pad(vec, (0, pad_length), mode="constant", constant_values=PAD_ID).astype(int)
    return vec


def choose_buckets(encoder_length, decoder_length, buckets):
    for x, y in buckets:
        if encoder_length <= x and decoder_length <= y:
            return x, y
    return None


def build_train_test_set(generator, dictionary, buckets):
    train_sets = [[] for i in range(len(buckets))]
    test_sets = [[] for i in range(len(buckets))]

    token2id = dictionary.token2id

    i = 0
    for c, u in generator:
        res = choose_buckets(len(c), len(u), buckets)

        if not res:  # doesn't in buckets, abandon this example
            continue

        c_vec = map_to_id(c, token2id, res[0])
        u_vec = map_to_id(u, token2id, res[1])
        concat_vec = np.append(c_vec, u_vec)

        index = buckets.index(res)
        if i % 5 == 0:
            test_sets[index].append(concat_vec)
        else:
            train_sets[index].append(concat_vec)
        i += 1

    for i in range(len(buckets)):
        x, y = buckets[i]
        train_data = np.array(train_sets[i], dtype=np.int32)
        test_data = np.array(test_sets[i], dtype=np.int32)
        print("Training Set with bucket of ({},{}) shape is {}".format(x, y, train_data.shape))
        print("Test Set with bucket of ({},{}) shape is {}".format(x, y, test_data.shape))

        train_path = os.path.join(DATA_PATH, TRAIN_FILE_NAME.format(x, y))
        test_path = os.path.join(DATA_PATH, TEST_FILE_NAME.format(x, y))
        np.savetxt(train_path, train_data, delimiter=",", fmt='%i')
        np.savetxt(test_path, test_data, delimiter=",", fmt='%i')
        print("Save training set to %s" % train_path)
        print("Save test set to %s" % test_path)
        print("\n")


def main():
    # build dictionary
    dictionary_path = os.path.join(DATA_PATH, DICT_NAME)
    generator = cmd_cleaned_generator()

    if os.path.exists(dictionary_path) and os.path.isfile(dictionary_path):
        dictionary = load_dictionary(dictionary_path)
    else:
        dictionary = build_dictionary(generator)
        generator = cmd_cleaned_generator()

    print(_START_VOCAB)
    v = [dictionary.token2id[i] for i in _START_VOCAB]
    print(v)
    print('\n')
    print("length of dictionary: %d" % len(dictionary.keys()))
    build_train_test_set(generator, dictionary, buckets=BUCKETS)


if __name__ == '__main__':
    main()
