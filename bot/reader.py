# -*- coding: utf-8 -*-
# Project : DeepBot
# Created by igor on 16/9/16

'''
reader
'''
import os
import re

import tensorflow as tf

from bot.config import BUCKETS, DATA_PATH

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4000


def read_data(filename_queue, bucket):
    '''

    :param filename_queue:file queue
    :param bucket:(encoder_length,decoder_length)
    :return:
    '''

    class DataRecord(object):
        pass

    result = DataRecord()

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    recoder_defaults = [[1] for i in range(bucket[0] + bucket[1])]
    recoder = tf.decode_csv(value,
                            record_defaults=recoder_defaults)

    # encoder_input
    result.encoder = tf.pack(recoder[0:bucket[0]])
    # decoder_input
    result.decoder = tf.pack(recoder[bucket[0]:])

    return result


def _generate_encoder_and_decoder_batch(encoder, decoder, min_queue_example, batch_size, shuffle):
    '''
    generate batch data for train
    :param encoder:1-D Tensor of [bucket[0]]
    :param decoder:1-D Tensor of [bucket[1]
    :param min_queue_example:
    :param batch_size:
    :param shuffle:shuffle data
    :return:
        encoder_batch:2-D Tensor of [batch_size,bucket[0]]
        decoder_batch:2-D Tensor of [batch_size,bucket[1]]
    '''

    num_preprocess_threads = 4
    if shuffle:
        encoder_batch, decoder_batch = tf.train.shuffle_batch(
            [encoder, decoder],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_example + 3 * batch_size,
            min_after_dequeue=min_queue_example)
    else:
        encoder_batch, decoder_batch = tf.train.batch(
            [encoder, decoder],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_example + 3 * batch_size)

    return encoder_batch, decoder_batch


def distorted_inputs(filenames, batch_size, bucket, shuffle=False):
    '''
    :param filenames:list of filename
    :param batch_size:
    :param bucket:
    :return:
        encoder_batch:2-D Tensor of [batch_size,bucket[0]]
        decoder_batch:2-D Tensor of [batch_size,bucket[1]]
    '''

    for f in filenames:
        if not os.path.exists(f):
            raise ValueError("Failed to find file: %s" % f)

    filenames_queue = tf.train.string_input_producer(filenames)

    read_input = read_data(filenames_queue, bucket)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d case before starting to text.'
          'This will take a few minutes.' % min_queue_examples)
    return _generate_encoder_and_decoder_batch(
        read_input.encoder, read_input.decoder,
        min_queue_example=min_queue_examples,
        batch_size=batch_size,
        shuffle=shuffle)


def batch_data_with_buckets(batch_size):
    '''
    :return:bucket_inputs
     list of reader Tensor, every Tensor related to a bucket in BUCKETS
    '''
    bucket_inputs = []

    for bucket in BUCKETS:
        filesnames = [os.path.join(DATA_PATH, fn) for fn in os.listdir(DATA_PATH) if
                      re.search(r"train{}_{}.*\.csv".format(bucket[0], bucket[1]), fn)]
        encoder_batch, decoder_batch = distorted_inputs(filenames=filesnames, batch_size=batch_size, bucket=bucket)
        bucket_inputs.append((encoder_batch, decoder_batch))
    return bucket_inputs


if __name__ == '__main__':
    # Test one file
    # bucket = BUCKETS[0]
    # filesnames = [os.path.join(DATA_PATH, fn) for fn in os.listdir(DATA_PATH) if
    #               re.search(r"train{}_{}.*\.csv".format(bucket[0], bucket[1]), fn)]
    # print(filesnames)
    # encoder_batch, decoder_batch = distorted_inputs(filenames=filesnames, batch_size=32, bucket=bucket)
    #
    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     try:
    #         i = 0
    #         while not coord.should_stop():
    #             if i % (30000 // 32) == 0:
    #                 print(i)
    #             i += 1
    #             x, y = sess.run([encoder_batch, decoder_batch])
    #             # print(x)
    #             # print("______")
    #             # print(y)
    #             # print("\n")
    #     except tf.errors.OutOfRangeError:
    #         print("Done training -- epoch limit reached")
    #     finally:
    #         coord.request_stop()
    #     coord.join(threads)
    #     print(i)

    # Test all
    inputs = batch_data_with_buckets(32)
    print(inputs)
