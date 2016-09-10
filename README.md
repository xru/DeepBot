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

## Deep Learning Model

### End-To-End model
!(Basic Seq2Seq model)[http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2016/04/nct-seq2seq.png]

### Buckets
First, I choose to use buckets for process different input length efficient.

There are some key point related:

- buckets size set
- batch buckets input generate, I will try to use Tensorflow reader
- model process logic


## Reference

- [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v3.pdf)
- [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909)

