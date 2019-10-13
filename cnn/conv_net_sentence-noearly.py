"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
#!/usr/bin/python3

import cPickle
import csv
import six.moves.cPickle as pickle
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys,os
import time
from collections import Counter
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def majority( a ):
    a=a.tolist()
    b = Counter(a)
    #print np.argmax(counts)
    return b.most_common(1)

def train_conv_net_distant(datasets, U, img_w, filter_hs=[3,4,5], hidden_units=[100,2],  dropout_rate=[0.5], shuffle_batch=True,
                   n_epochs=1, batch_size=50, lr_decay = 0.95, conv_non_linear="relu", activations=[Iden], sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1
    print "=======Distant CNN model=========="
    print img_h	
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], 
                                zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],
                                                                     Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, 
                            dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
   
    train_set = new_data[:n_batches*batch_size,:]
    val_set = datasets[1]
    test_set_x = datasets[2]
    
    
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    
    get_acc_val_model = theano.function([index], classifier.errors(y),
        givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
                y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
            
    #compile theano functions to get train/val/test errors
    get_acc_train_model = theano.function([index], classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                allow_input_downcast = True)     
    test_pred_layers_a = []
    test_pred_layers_b = []
    
    
    # If test set is large loading the whole test set can give GPU memory allocation error
    # So we make prediction only by taking a maximum of 2000 test examples at a time
    test_size = test_set_x.shape[0]
    test_batch_size = 2000
    test_iter = int(test_size/test_batch_size)
    extra_test_size = test_size - test_iter * test_batch_size
    
    test_layer0_input_a = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_batch_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output_a = conv_layer.predict(test_layer0_input_a, test_batch_size)
        test_pred_layers_a.append(test_layer0_output_a.flatten(2))
    test_layer1_input_a = T.concatenate(test_pred_layers_a, 1)
    test_y_pred_a = classifier.predict(test_layer1_input_a) #there is also the function predict_p 
    
    
    test_layer0_input_b = Words[T.cast(x.flatten(),dtype="int32")].reshape((extra_test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output_b = conv_layer.predict(test_layer0_input_b, extra_test_size)
        test_pred_layers_b.append(test_layer0_output_b.flatten(2))
    test_layer1_input_b = T.concatenate(test_pred_layers_b, 1)
    test_y_pred_b = classifier.predict(test_layer1_input_b)
    
    
    test_model_all_a = theano.function([x], test_y_pred_a, allow_input_downcast = True)   
    test_model_all_b = theano.function([x], test_y_pred_b, allow_input_downcast = True) 
    
    #start training over mini-batches
    print '=======Distant CNN model... training============='
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0  
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [get_acc_train_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [get_acc_val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, 
                                         time.time()-start_time, train_perf * 100., val_perf*100.))

    #prediction1 = np.zeros((test_iter, test_batch_size))
    #for j in xrange(test_iter):
    #    prediction1[j] = test_model_all_a(test_set_x[test_batch_size*j:test_batch_size*(j+1) , :])
    #    
    #prediction2 = test_model_all_b(test_set_x[-extra_test_size: , :])    
    #prediction = list(prediction1.reshape(test_batch_size * test_iter)) + list(prediction2)
 

    print "saving parameters"#return model parameters
    with open('obj.save', 'wb') as f:
        cPickle.dump(classifier.params, f, -1)

    print 'saved parameters'
    print classifier.params
    return 0

def train_conv_net(num_classes,datasets, U, img_w, filter_hs=[3,4,5], hidden_units=[100,2],  dropout_rate=[0.5], shuffle_batch=True,
                   n_epochs=25, batch_size=50, lr_decay = 0.95, conv_non_linear="relu", activations=[Iden], sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1
    print img_h	
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], 
                                zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],
                                                                     Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, 
                            dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    ####****loading parameters****#####
    ###else , if distant step uses non-static place it here
    with open('obj.save', 'rb') as f:
        tmp = cPickle.load(f)
    print 'loaded parameters' 
    for i in range(len(classifier.params)-1): #don't update Words
       classifier.params[i].set_value(tmp[i].get_value())
    print classifier.params

    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
   
    train_set = new_data[:n_batches*batch_size,:]
    val_set = datasets[1]
    test_set_x = datasets[2]
    
    
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    
    get_acc_val_model = theano.function([index], classifier.errors(y),
        givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
                y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
            
    #compile theano functions to get train/val/test errors
    get_acc_train_model = theano.function([index], classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                allow_input_downcast = True)     
    test_pred_layers_a = []
    test_pred_layers_b = []
    
    
    # If test set is large loading the whole test set can give GPU memory allocation error
    # So we make prediction only by taking a maximum of 2000 test examples at a time
    test_size = test_set_x.shape[0]
    test_batch_size = 2000
    test_iter = int(test_size/test_batch_size)
    extra_test_size = test_size - test_iter * test_batch_size
    
    test_layer0_input_a = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_batch_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output_a = conv_layer.predict(test_layer0_input_a, test_batch_size)
        test_pred_layers_a.append(test_layer0_output_a.flatten(2))
    test_layer1_input_a = T.concatenate(test_pred_layers_a, 1)
    test_y_pred_a = classifier.predict(test_layer1_input_a)
    test_post_y_pred_a = classifier.predict_p(test_layer1_input_a)  #get posteriors  
    
    test_layer0_input_b = Words[T.cast(x.flatten(),dtype="int32")].reshape((extra_test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output_b = conv_layer.predict(test_layer0_input_b, extra_test_size)
        test_pred_layers_b.append(test_layer0_output_b.flatten(2))
    test_layer1_input_b = T.concatenate(test_pred_layers_b, 1)
    test_y_pred_b = classifier.predict(test_layer1_input_b)
    test_post_y_pred_b = classifier.predict_p(test_layer1_input_b)   #get posteriors  
    
    test_model_all_a = theano.function([x], test_y_pred_a, allow_input_downcast = True)   
    test_model_all_b = theano.function([x], test_y_pred_b, allow_input_downcast = True) 
 
    test_post_model_all_a = theano.function([x], test_post_y_pred_a, allow_input_downcast = True)   
    test_post_model_all_b = theano.function([x], test_post_y_pred_b, allow_input_downcast = True) 

	
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    test_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0 
    done_looping = False

    while (epoch < n_epochs):
        start_time = time.time()
        #if (test_val_perf <= val_perf) :
        #    test_val_perf = val_perf
        #    
        #else : 
        #    break
 
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)

        train_losses = [get_acc_train_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [get_acc_val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)

        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, 
                                         time.time()-start_time, train_perf * 100., val_perf*100.))
   
            
    prediction1 = np.zeros((test_iter, test_batch_size))
    for j in xrange(test_iter):
        prediction1[j] = test_model_all_a(test_set_x[test_batch_size*j:test_batch_size*(j+1) , :])
        
    prediction2 = test_model_all_b(test_set_x[-extra_test_size: , :])    
    prediction = list(prediction1.reshape(test_batch_size * test_iter)) + list(prediction2)


    post_prediction1 = np.zeros((test_iter, test_batch_size,num_classes))
    #print post_prediction1.shape

    for j in xrange(test_iter):
        a = test_post_model_all_a(test_set_x[test_batch_size*j:test_batch_size*(j+1) , :])
        #print a.shape
        post_prediction1[j] = test_post_model_all_a(test_set_x[test_batch_size*j:test_batch_size*(j+1) , :])
    #print post_prediction1.shape
    
    post_prediction2 = test_post_model_all_b(test_set_x[-extra_test_size: , :]) 

    #print post_prediction2.shape

    post_prediction = list(post_prediction1.reshape((test_batch_size * test_iter),num_classes)) + list(post_prediction2)
    
    return prediction , post_prediction

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l, k, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)

    return x

def make_idx_data_cv(revs,word_idx_map, max_l, k, cv=10,  filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, valid = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        #if(len(sent)!=65): print type(sent)	
        sent.append(rev["y"])
        if rev["split"]==cv:            
            valid.append(sent)        
        else:  
            train.append(sent)
    #print len(train[1])
    train = np.array(train,dtype="int")
    valid = np.array(valid,dtype="int")

    return [train, valid]

def make_idx_test_data(test_revs, word_idx_map, max_l, k, filter_h = 5):
    """
    Transforms sentences into a 2-d matrix.
    """
    test = []
    for rev in test_revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        test.append(sent)  
        
    test = np.array(test,dtype="int")
    return test     

if __name__=="__main__":
    print "loading data...",
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    savefile= sys.argv[3] 
    resultsfile = sys.argv[4]
    n_epochs = int(sys.argv[5])
    hidden_unit = int(sys.argv[6])
    K = int(sys.argv[7])
    M = int(sys.argv[8])
    post_resultsfile = sys.argv[9]
    num_classes = int(sys.argv[10])
     
    #savefile="./input/"+str(name)+str(N0)+"pos_"+str(N1)+"neg_"+str(N2)+"neu"+".p"
    x = cPickle.load(open(savefile,"rb"))
    #train_revs, test_revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4] ,x[5]
    #max_l = max (np.max(pd.DataFrame(train_revs)["num_words"]) , np.max(pd.DataFrame(test_revs)["num_words"]))
    train_revs, test_revs, W, W2, word_idx_map, vocab ,distant_revs, distant_vocab , distant_W, distant_word_idx_map = x[0], x[1], x[2], x[3], x[4] ,x[5] ,x[6],x[7],x[8],x[9]
    max_l = max (np.max(pd.DataFrame(train_revs)["num_words"]) , np.max(pd.DataFrame(test_revs)["num_words"]))
    distant_max_l=np.max(pd.DataFrame(distant_revs)["num_words"])


    #print max_l
    print "data loaded!"
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
        Udistant = distant_W
    results = []
    sys_runs=3
    r = range(0,1)

    #the actual predictions 
    #if ("A" in savefile):test_gold="/usb_disk/twitter/Tweester-Journal-exp/SYSTEMS/CNN_System/scoring/SemEval2016_task4_subtaskA_test_gold.txt"
    #if ("B" in savefile):test_gold="/usb_disk/twitter/Tweester-Journal-exp/SYSTEMS/CNN_System/scoring/SemEval2016_task4_subtaskB_test_gold.txt"
    #
    #Y_pred_gold = pd.read_csv( test_gold, header=None,  delimiter="\t", quoting=csv.QUOTE_NONE)
    #y_test = np.array(Y_pred_gold[2])

    #the actual test data
    test = make_idx_test_data(test_revs, word_idx_map, max_l, k = K, filter_h = 5)    
    testacc = 0
    best_y_pred =[]
    best_post_y_pred =[]

    for i in r:
    #i = 1
    #if (i == 1):
        datasets = make_idx_data_cv(train_revs, word_idx_map, max_l, K, i,  filter_h=5)
        datasets.append(test)
        distant_datasets = make_idx_data_cv(distant_revs,distant_word_idx_map, distant_max_l, K, i,  filter_h=5)
        distant_datasets.append(test)

        l_prediction = train_conv_net_distant(distant_datasets,
                              Udistant,img_w=K,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[hidden_unit,num_classes], #hidden_units=[100,3] is used for 3-classes
                              shuffle_batch=True, 
                              n_epochs=1, 
                              sqr_norm_lim=9,
                              non_static=non_static,#non_static or False,
                              batch_size=M,
                              dropout_rate=[0.5])

    r = range(0,sys_runs)
    for i in r:
        y_prediction , post_y_prediction = train_conv_net(num_classes,datasets,
                              U,img_w=K,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[hidden_unit,num_classes], #3 is for 3 class model,
                              shuffle_batch=True, 
                              n_epochs=n_epochs, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50, #this is traininng on semeval only, gives better results when is set to 50
                              dropout_rate=[0.5])
        #acc = np.sum(y_test == y_prediction, axis = 0) * 100/float(len(y_test))
        #print 'Test Accuracy ' + ' : ' + str(acc) + ' %'
        #print ''
        #if (testacc <= acc ) :
        #    #keep this as best system
        #     best_y_pred = y_prediction
        #     best_post_y_pred = post_y_prediction
        #testacc = acc

        if (i == 0):
            Yn = np.asarray(y_prediction).astype(int)
            Yn = Yn[:,np.newaxis]
            #print Yn.shape
            temp_post = pd.DataFrame(post_y_prediction)
            post0=np.asarray(temp_post[0])
            post1=np.asarray(temp_post[1])
            if ("A" in savefile): post2=np.asarray(temp_post[2])
        
        else:
            Y1 = np.asarray(y_prediction).astype(int)
            Y1 = Y1[:,np.newaxis]
            #print Y1.shape
        
            Yn = np.column_stack((Yn , Y1)) #add a column with the new result
        
            temp_post = pd.DataFrame(post_y_prediction)
            t0=np.asarray(temp_post[0])
            t1=np.asarray(temp_post[1])
            if ("A" in savefile): t2=np.asarray(temp_post[2])
        
            post0 = np.column_stack((post0 , t0))
            post1 = np.column_stack((post1 , t1))
            if ("A" in savefile): post2 = np.column_stack((post2 , t2))

    ####Yn = np.asarray(Yn) 
    ####A=[]
    ####
    ####for row in Yn:
    ####    #print row
    ####    k = majority(row)
    ####    A.append(k[0][0])
    ####y_prediction = A
    ####
    p0=np.mean(post0, axis=1)
    p1=np.mean(post1, axis=1)
    if ("A" in savefile): p2=np.mean(post2, axis=1)
    if ("A" in savefile): post_y_prediction=np.column_stack((p0,p1,p2))
    if ("B" in savefile): post_y_prediction=np.column_stack((p0,p1))
    ###print post_y_prediction
    ####start=1
    ####count= len(y_prediction)
    ####
    ####idx = [num for num in range(start,start+count)]

    ####d = pd.DataFrame({ 'id': idx ,'nan':'NA','sentiment': y_prediction})
    ####d.to_csv(resultsfile, header=False ,index=False,sep='\t')
    #d = pd.DataFrame({ 'id': idx ,'nan':'NA','sentiment': best_y_pred})
    #d.to_csv(resultsfile+"best", header=False ,index=False,sep='\t')
    filename_post=post_resultsfile+'runs'+str(sys_runs)
    np.savetxt(filename_post, post_y_prediction, delimiter='\t')
    #COPY posteriors to EVALUATION FOLDER
    if ("A" in savefile) : EVALUATION_FOLDER = '/home/sasa/SYSTEMS/fusion/systems/subtaskA/cnn_train2017/'
    if ("B" in savefile) : #copy in D also
        EVALUATION_FOLDER = '/home/sasa/SYSTEMS/fusion/systems/subtaskD/cnn_train2017/'    
        command = 'cp '+filename_post+" " + EVALUATION_FOLDER+'/posteriors.csv' 
        os.system(command)
    EVALUATION_FOLDER = '/home/sasa/SYSTEMS/fusion/systems/subtaskB/cnn_train2017/'   
    command = 'cp '+filename_post+" " + EVALUATION_FOLDER+'/posteriors.csv' 
    os.system(command)


    ##np.savetxt(post_resultsfile+"best", best_post_y_pred, delimiter='\t')
