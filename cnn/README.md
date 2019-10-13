## Convolutional Neural Networks for Sentence Classification

### Requirements
Code is written in Python  and requires Theano  .

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/

### run all steps
```
ex. python python run_all.py 1000 1000 1000 10 100 50 A  
```
this runs the CNN system with 
N1= 1000 (number of positive tweets from emoticons corpus)
N1= 1000 (number of negative tweets from emoticons corpus)
N1= 1000 (number of neutral tweets from emoticons corpus)
epochs = 10
batch_size=100
w2vec_size=50
Subtask = A (use B for subtaskB)

### Data Preprocessing
To process the raw data, run

```
python process_data.py  
```

This will create a pickle object called `mr.p` in the same folder, which contains the dataset
in the right format.

 
### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -rand
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -static -word2vec
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

This will run the CNN-rand, CNN-static, and CNN-nonstatic models respectively in the paper.

### Using the GPU

For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

### Example output
CPU output:
```
epoch: 1, training time: 219.72 secs, train perf: 81.79 %, val perf: 79.26 %
epoch: 2, training time: 219.55 secs, train perf: 82.64 %, val perf: 76.84 %
epoch: 3, training time: 219.54 secs, train perf: 92.06 %, val perf: 80.95 %
```
GPU output:
```
epoch: 1, training time: 16.49 secs, train perf: 81.80 %, val perf: 78.32 %
epoch: 2, training time: 16.12 secs, train perf: 82.53 %, val perf: 76.74 %
epoch: 3, training time: 16.16 secs, train perf: 91.87 %, val perf: 81.37 %
```


CNN-rand: Our baseline model where all
words are randomly initialized and then modified
during training.

CNN-static: A model with pre-trained
vectors from word2vec. All words—
including the unknown ones that are randomly
initialized—are kept static and only
the other parameters of the model are learned.

• CNN-non-static: Same as above but the pretrained
vectors are fine-tuned for each task.


Ye Zhang has written a [very nice paper](http://arxiv.org/abs/1510.03820) doing an extensive analysis of model variants (e.g. filter widths, k-max pooling, word2vec vs Glove, etc.) and their effect on performance.
