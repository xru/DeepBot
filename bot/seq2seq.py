# -*- coding: utf-8 -*-
# Project : DeepBot
# Created by igor on 16/9/10

import random

import numpy as np
import tensorflow as tf


class Seq2SeqModel(object):
    '''
    Sequence-to-Sequence model with attention and for multiple buckets
    '''

    def __init__(self,
                 vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        '''
        Create the model
        Equal to build tf graph

        :param vocab_size: size of vocabulary
        :param buckets: a list of pairs (I,O),where I specifies maximum input length
            that will be processed in that bucket. and O specifies maximum output length
        :param size: number of units in each layer of model
        :param num_layers: number of layers in model
        :param max_gradient_norm: gradients will be clipped to maximally this norm
        :param batch_size: the size of the batches used during training:
        :param learning_rate:learning rate to strat with
        :param learning_rate_decay_factor: decay learning rate by this much when needed
        :param use_lstm:if true ,we use LSTM cells instead of GRU cells
        :param num_samples:number of samples for sampled softmax
        :param forward_only:if set,we do not construct the backward pass in the model
        :param dtype:the data type to use to store internal variables
        '''

        self.vocab_size = vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype
        )
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor
        )
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection
        output_projection = None
        softmax_loss_function = None

        if num_samples > 0 and num_samples < self.vocab_size:
            w = tf.get_variable("proj_w", [size, self.vocab_size], dtype=dtype)
            w_t = tf.transpose(w)  # [vocab_size,size]
            b = tf.get_variable('proj_b', [self.vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                               num_samples, self.vocab_size),
                    dtype)

            softmax_loss_function = sampled_loss

        # Create the internal mutil-layer cell for RNN
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # use embedding for the input and attention
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                embedding_size=size,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype)

        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in range(buckets[-1][0]):  # Last bucket is the biggest
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                      name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[batch_size],
                                                      name="weight{0}".format(i)))

        # targets are decoder inputs shifted by one
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses
        if forward_only:
            self.ouputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)

            if output_projection is not None:
                for b in range(len(buckets)):
                    self.ouputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.ouputs[b]]
        else:
            self.ouputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norm = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norm.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step
                ))
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        '''
        Run a step of the model feeding the given inputs
        :param session: tensorflow session to use
        :param encoder_inputs: list of numpy int vectors to feed as encoder inputs
        :param decoder_inputs: list of numpy int vectors to feed as decoder inputs
        :param target_weights: list of numpy vectors to feed as target weights
        :param bucket_id: which bucket of the model to use
        :param forward_only: whether to do the backward step or only forward
        :return:
            A triple consisting of gradient norm(or None if we did not do backward),
            average perplexity, and the outputs.

        :raises:
            ValueError: if length of encoder_inputs,decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id
        '''

        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_size) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weight length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights)), decoder_size)

        # feed_dict
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].naem] = decoder_inputs[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [
                self.updates[bucket_id],
                self.gradient_norm[bucket_id],
                self.losses[bucket_id]
            ]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.ouputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


if __name__ == '__main__':
    from bot.config import FLAGS

    _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    with tf.Session() as sess:
        model = Seq2SeqModel(
            8000,
            _buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor)
