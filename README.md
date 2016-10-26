# DeepBot

Implementation of chatbot

## Corpus

- [Cornell Moive Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [Ubuntu Dialogue Corpus v2.0](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

## Simple Analysis

Movie Dialogs
```
most length of context is 369
most length of utterance is 672
mean length of context is 13.097878
mean length of utterance is 13.576396
max,min,mean of abs(context_length -utterance_length) is 670.000000,0.000000,12.089620
most common words :
[('.', 483355), (',', 241764), ('I', 205665), ('?', 163683), ('you', 160900), ('the', 125099), ('to', 114413), ('a', 95555), ("'s", 94990), ("n't", 80776)]
```
This is a rough analysis, we need lot of preprocess to this data so that it can be used in training model.

But I jumped this process, I directly use data that preprocessed by [this](https://github.com/suriyadeepan/easy_seq2seq/blob/master/data/pull_data.sh). It very like translate data. Split data to four part:

- train.enc
- train.dec
- test.enc
- test.dec

In IR system, I still use orgin data for retrieval.

## Deep Learning Model

### End-To-End model
![Basic Seq2Seq model](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2016/04/nct-seq2seq.png)

[DeepBot model and data detail](./bot/README.md)

## Reference

- [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v3.pdf)
- [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909)
- [Chatbots with Seq2Seq](http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)
