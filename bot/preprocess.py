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

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"

_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_SYMBOLS_ID = [PAD_ID, GO_ID, EOS_ID, UNK_ID]


def cmd_cleaned_generator(min_length=1, max_length=24):
    pattern = re.compile(r"[^a-zA-Z1-9\.!?,' ]")
    for context, utterance in utterance_generator(MOVIE_CORPUS_DIR):
        context = tokenzier.tokenize(pattern.sub("", context))
        utterance = tokenzier.tokenize(pattern.sub("", utterance))
        if len(context) < min_length or len(context) > max_length \
                or len(utterance) < min_length or len(utterance) > max_length:
            continue
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

    dictionary = corpora.Dictionary([_START_VOCAB])
    dictionary.add_documents(c + u for c, u in generator)

    # 去除低频的ID
    filter_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if
                  docfreq < min_freq and tokenid not in _SYMBOLS_ID]

    dictionary.filter_tokens(filter_ids)
    dictionary.compactify()

    pickle.dump(dictionary, open(dictionary_path, 'wb'))
    print("SVAE dictionary to %s" % (dictionary_path))

    return dictionary


def map_to_id(tokens, token2id, max_length):
    vec = [token2id[i] for i in tokens]
    pad_length = max_length - len(vec)
    vec = np.pad(vec, (0, pad_length), mode="constant", constant_values=token2id[_UNK]).astype(int)

    return vec


def build_train_test_set(generator, dictionary):
    train_set = []
    test_set = []
    i = 0
    UNK_ID = dictionary.token2id["UNK"]
    token2id = defaultdict(lambda: UNK_ID)
    token2id.update(dictionary.token2id)
    for c, u in generator:
        c_vec = map_to_id(c, token2id, MAX_LENGTH)
        u_vec = map_to_id(u, token2id, MAX_LENGTH)

        concat_vec = np.append(c_vec, u_vec)
        if i % 5 == 0:
            test_set.append(concat_vec)
        else:
            train_set.append(concat_vec)
        i += 1

    train_set = np.array(train_set, int)
    test_set = np.array(test_set, int)

    print("Training Set shape {}".format(train_set.shape))
    print("Test Set shap {}".format(test_set.shape))

    train_path = os.path.join(DATA_PATH, TRAIN_FILE_NAME)
    test_path = os.path.join(DATA_PATH, TEST_FILE_NAME)
    np.savetxt(train_path, train_set, delimiter=",")
    np.savetxt(test_path, test_set, delimiter=",")

    print("Save training set to %s" % train_path)
    print("Save test set to %s" % test_path)


def main():
    # build dictionary
    dictionary_path = os.path.join(DATA_PATH, DICT_NAME)
    generator = cmd_cleaned_generator(MIN_LENGTH, MAX_LENGTH)

    if os.path.exists(dictionary_path) and os.path.isfile(dictionary_path):
        dictionary = load_dictionary(dictionary_path)
    else:
        dictionary = build_dictionary(generator)

    build_train_test_set(generator, dictionary)
    print("length of dictionary: %d" % len(dictionary.keys()))


if __name__ == '__main__':
    main()
