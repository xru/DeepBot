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
train_dir = '/data/deepbot/model'
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
        word_ids = tl.nlp.words_to_word_ids(context, vocab_dec)
        print("dec word_ids:", word_ids)
        print("dec context:", context)

    train_bucket_sizes = [len(train_set[b]) for b in range(len(BUCKETS))]

    train_total_size = float(sum(train_bucket_sizes))

    print('the num of training data in each buckets: %s' % train_bucket_sizes)
    print('the num of training data: %d' % train_total_size)

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print('train_buckets_scale:', train_buckets_scale)

    '''
    Step 5 : Create model
    '''
    print()
    print("Create Embedding Attention Seq2Seq Model")
    with tf.variable_scope("model", resue=None):
        model = tl.layers.EmbeddingAttentionSeq2seqWrapper(
            enc_vocab_size,
            dec_vocab_size,
            BUCKETS,
            size,
            num_layers,
            max_gradient_norm,
            batch_size,
            learning_rate,
            learning_rate_decay_factor,
            forward_only=False)

    sess.run(tf.initialize_all_variables())
    tl.layers.print_all_variables()

    if resume:
        print("Load existing model" + "!" * 10)
        if is_npz:
            load_params = tl.files.load_npz(name=model_file_name + '.npz')
            tl.files.assign_params(sess, load_params, model)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, model_file_name + '.ckpt')

    '''
    Step 6 : Training
    '''
    print()
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    if __name__ == '__main__':
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            # Get a batch and make a step
            start_tiem = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id, PAD_ID, GO_ID, EOS_ID, UNK_ID)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_tiem) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics and run evals
            if current_step % steps_per_checkpoint == 0:
                # print statistics for the previous epoch
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step time %.2f "
                      "perplexity %.2f" % model.global_step.eval(),
                      model.learning_rate.eval(), step_time, perplexity)

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                # save model
                if is_npz:
                    tl.files.save_npz(model.all_params,
                                      name=model_file_name + '.npz')
                else:
                    print('Model is save to: %s' % (model_file_name + '.ckpt'))
                    checkpoint_path = os.path.join(train_dir, model_file_name + '.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                # Run evals on dev set
                for bucket_id in range(len(BUCKETS)):
                    if len(dev_set[bucket_id]) == 0:
                        print(" eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id, PAD_ID, GO_ID, EOS_ID, UNK_ID)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print(" eval:bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def main_decode():
    '''
    Create model and load parmaters.
    '''
    with tf.variable_scope('model', reuse=None):
        model_eval = tl.layers.EmbeddingAttentionSeq2seqWrapper(
            source_vocab_size=enc_vocab_size,
            target_vocab_size=dec_vocab_size,
            buckets=BUCKETS,
            size=size,
            num_layers=num_layers,
            max_gradient_norm=max_gradient_norm,
            batch_size=1,
            learning_rate=learning_rate,
            learning_rate_decay_factor=learning_rate_decay_factor,
            forward_only=True)
    sess.run(tf.initialize_all_variables())

    if is_npz:
        print("Load parameters from npz")
        load_parms = tl.files.load_npz(name=model_file_name + '.npz')
        tl.files.assign_params(sess, load_parms, model_eval)
    else:
        print("Load parameters from ckpt")
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model_eval.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("no %s exit" % ckpt.model_checkpoint_path)

    tl.layers.print_all_variables()

    # Load vocabularies
    enc_vocab_path = os.path.join(data_dir, 'vocab%d.enc' % enc_vocab_size)
    dec_vocab_path = os.path.join(data_dir, 'vocab%d.dec' % dec_vocab_size)
    enc_vocab, _ = tl.nlp.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = tl.nlp.initialize_vocabulary(dec_vocab_path)

    # Decode from standard input
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

    while sentence:
        token_ids = tl.nlp.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
        bucket_id = min([b for b in range(len(BUCKETS)) if BUCKETS[b][0] > len(token_ids)])
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model_eval.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id, PAD_ID, GO_ID, EOS_ID, UNK_ID)
        _, _, output_logits = model_eval.step(sess, encoder_inputs, decoder_inputs,
                                              target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    try:
        main_train()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
        tl.ops.exit_tf(sess)
