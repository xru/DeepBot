# DeepBot build with tensorflow

## data preprocess

- max sequence length : 20 tokens

    mean of sequence length about 13, we can use it as variable param,
    I will try different way to process variable length

- train examples / test examples : 80% / 20%

after process:
```
['_PAD', '_GO', '_EOS', '_UNK']
[25221, 25222, 12707, 25223]


length of dictionary: 25224

Training Set with bucket of (5,10) shape is (32587, 15)
Test Set with bucket of (5,10) shape is (7979, 15)
Save training set to /data/deepbot/train5_10.csv
Save test set to /data/deepbot/test5_10.csv


Training Set with bucket of (10,15) shape is (47144, 25)
Test Set with bucket of (10,15) shape is (11836, 25)
Save training set to /data/deepbot/train10_15.csv
Save test set to /data/deepbot/test10_15.csv


Training Set with bucket of (20,25) shape is (52248, 45)
Test Set with bucket of (20,25) shape is (13230, 45)
Save training set to /data/deepbot/train20_25.csv
Save test set to /data/deepbot/test20_25.csv


Training Set with bucket of (40,50) shape is (34667, 90)
Test Set with bucket of (40,50) shape is (8617, 90)
Save training set to /data/deepbot/train40_50.csv
Save test set to /data/deepbot/test40_50.csv
```

## Buckets
First, I choose to use buckets for process different input length efficient.

There are some key point related:

- buckets size set
- batch buckets input generate, I will try to use Tensorflow reader
- model process logic

## Inputs
Graph build code:

```Python
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
```
In graph build process, self.encoder_inputs is a list of Tensors,
every Tensors shape is batch_size. So length of self.encoder_inputs
represent RNN time step length,every Tensor of self.encoder_inputs represent
a batch of data in some time step, we can think self.encoder_inputs has shape
of [time_step_length * batch], that means a column is a sample!

Step Code:
graph decide how to feed data

```Python
encoder_size, decoder_size = self.buckets[bucket_id]
# feed_dict
input_feed = {}
for l in range(encoder_size):
    input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
for l in range(decoder_size):
    input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
    input_feed[self.target_weights[l].naem] = decoder_inputs[l]
```
So we first choose a buckets and feed encoder_inputs with shape [encoder_size , batch],
and decoder_inputs with [decoder_size ,batch]
