# -*- coding: utf-8 -*-
# Project : DeepBot
# Created by igor on 2016/10/26
'''
Seq2Seq model with TensorLayer
'''

import math
import random
import time
import os
import re
import sys

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
from tensorflow.python.platform import gfile

# Create vacabulary file(if it does not exit yet) from data file
_WORD_SPLIT = re.compile(rb"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(rb'\d')
normalize_digits = True  # replace all digits to 0

# Special vocabulary symbols
_PAD = b'_PAD'
_GO = b'_GO'
_EOS = b'_EOS'
_UNK = b'_UNK'
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
plot_data = True

# Model
BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]
num_layers = 3
size = 1024
# Training
learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
batch_size = 64
num_samples = 512  # sampled softmax
max_train_data_size = 100  # Limit on the size of training data
steps_per_checkpoint = 10  # Print, save frequence
# Save_model
model_file_name = "model_chat"
resume = False
is_npz = False  # if true save by npz file, otherwise ckpt file

# Data Path
RAW_DATA_PATH = '/data/cmd_corpus'
data_dir = '/data/deepbot'
train_dir = '/data/deepbot'
enc_vocab_size = 30000
dec_vocab_size = 30000


def read_data(source_path, target_path, buckets, EOS_ID, max_size=None):
    '''
    Read data from source and target files and put into buckets.
    Corresponding source data and target data in the same line.

    :param source_path: path to the files token-ids for the encode
    :param target_path: path to the files token-ids for the decode
    :param max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will read completely

    :return:
     data_set: a list of length len(buckets);data_set[n] contains a list
     of (encode, decode) pairs read from the provided data files
    '''
    data_set = [[] for _ in buckets]

    with gfile.GFile(source_path, mode='r') as source_file:
        with gfile.GFile(target_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 10000 == 0:
                    print(" reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def main_train():
    '''
    Training loop
    '''

    '''
    Step 1 : Prepare raw data
    '''
    print()
    print("Prapare raw data")
    enc_raw_path = os.path.join(RAW_DATA_PATH, 'train.enc')
    dec_raw_path = os.path.join(RAW_DATA_PATH, 'train.dec')
    enc_raw_dev_path = os.path.join(RAW_DATA_PATH, 'test.enc')
    dec_raw_dev_path = os.path.join(RAW_DATA_PATH, 'test.dec')

    '''
    Step 2 : Create Vocabulary
    '''
    print()
    print("Create vocabularies")
    enc_vocab_path = os.path.join(data_dir, 'vocab%d.enc' % enc_vocab_size)
    dec_vocab_path = os.path.join(data_dir, 'vocab%d.dec' % dec_vocab_size)
    print("Vocabulary of encoder : %s" % enc_vocab_path)
    print("Vocabulary of decoder : %s" % dec_vocab_path)
    tl.nlp.create_vocabulary(enc_vocab_path, enc_raw_path,
                             enc_vocab_size, tokenizer=None,
                             normalize_digits=normalize_digits,
                             _DIGIT_RE=_DIGIT_RE, _START_VOCAB=_START_VOCAB)
    tl.nlp.create_vocabulary(dec_vocab_path, dec_raw_path,
                             dec_vocab_size, tokenizer=None,
                             normalize_digits=True,
                             _DIGIT_RE=_DIGIT_RE, _START_VOCAB=_START_VOCAB)

    '''
    Step 3 : Tokenize Training and Testing data.
    '''
    print()
    print("Tokenize data")
    enc_train_ids_path = os.path.join(data_dir, 'train.ids%d.enc' % enc_vocab_size)
    dec_train_ids_path = os.path.join(data_dir, 'train.ids%d.dec' % dec_vocab_size)
    print("Tokenized Training data of encode : %s" % enc_train_ids_path)
    print("Tokenized Training data of decode : %s" % dec_train_ids_path)
    tl.nlp.data_to_token_ids(enc_raw_path, enc_train_ids_path, enc_vocab_path,
                             tokenizer=None, normalize_digits=normalize_digits,
                             UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(dec_raw_path, dec_train_ids_path, dec_vocab_path,
                             tokenizer=None, normalize_digits=normalize_digits,
                             UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)

    # create tokenized file for the development(testing data)

    enc_dev_ids_path = os.path.join(data_dir, 'test.ids%d.enc' % enc_vocab_size)
    dec_dev_ids_path = os.path.join(data_dir, 'test.ids%d.dec' % dec_vocab_size)
    print("Tokenized Testing data of encode : %s" % enc_train_ids_path)
    print("Tokenized Testing data of decode : %s" % dec_train_ids_path)
    tl.nlp.data_to_token_ids(enc_raw_dev_path, enc_dev_ids_path,
                             enc_vocab_path, tokenizer=None,
                             normalize_digits=normalize_digits,
                             UNK_ID=UNK_ID,
                             _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(dec_raw_dev_path, dec_dev_ids_path,
                             dec_vocab_path, tokenizer=None,
                             normalize_digits=normalize_digits,
                             UNK_ID=UNK_ID,
                             _DIGIT_RE=_DIGIT_RE)

    enc_train = enc_train_ids_path
    dec_train = dec_train_ids_path
    enc_dev = enc_dev_ids_path
    dec_dev = dec_dev_ids_path

    '''
    Step 4 : Load both tokenized Training and Testing data into buckets
    and compute their size.
    '''
    print()
    print("Read development(test) data into buckets")
    dev_set = read_data(enc_dev, dec_dev, buckets=BUCKETS, EOS_ID=EOS_ID)

    if plot_data:
        # Visualize the development (testing) data
        print("dev data:", BUCKETS[0], dev_set[0][0])
        vocab_enc, rev_vocab_enc = tl.nlp.initialize_vocabulary(enc_vocab_path)
        context = tl.nlp.word_ids_to_words(dev_set[0][0][0], rev_vocab_enc)
        word_ids = tl.nlp.words_to_word_ids(context, vocab_enc)
        print('enc word_ids:', word_ids)
        print('enc context:', context)
        # decode
        vocab_dec, rev_vocab_dec = tl.nlp.initialize_vocabulary(dec_vocab_path)
        context = tl.nlp.word_ids_to_words(dev_set[0][0][1], rev_vocab_dec)
        word_ids = tl.nlp.words_to_word_ids(context, vocab_dec)
        print("dec word_ids:", word_ids)
        print("dec context:", context)

    print()
    print("Read traing data into buckets (limit: %d)" % max_train_data_size)
    train_set = read_data(enc_train, dec_train, BUCKETS,
                          EOS_ID, max_train_data_size)
    if plot_data:
        print("train data:", BUCKETS[0], train_set[0][0])
        vocab_enc, rev_vocab_enc = tl.nlp.initialize_vocabulary(enc_vocab_path)
        context = tl.nlp.word_ids_to_words(train_set[0][0][0], rev_vocab_enc)
        word_ids = tl.nlp.words_to_word_ids(context, vocab_enc)
        print('enc word_ids:', word_ids)
        print('enc context:', context)
        # decode
        vocab_dec, rev_vocab_dec = tl.nlp.initialize_vocabulary(dec_vocab_path)
        context = tl.nlp.word_ids_to_words(train_set[0][0][1], rev_vocab_dec)
        word_ids = tl.nlp.words_to_word_ids(context,vocab_dec)
        print("dec word_ids:", word_ids)
        print("dec context:", context)

    train_bucket_sizes = [len(train_set[b]) for b in range(len(BUCKETS))]

    train_total_size = float(sum(train_bucket_sizes))

    print('the num of training data in each buckets: %s' % train_bucket_sizes)
    print('the num of training data: %d' % train_total_size)

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print('train_buckets_scale:', train_buckets_scale)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    try:
        main_train()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
        tl.ops.exit_tf(sess)
