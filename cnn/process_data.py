#!/usr/bin/python3

import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import os
import csv


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
def read_otherdata_datasets(mode , more_data_folder,N0,N1,N2):
    print "read data distant corpus" + str(mode)
    if mode != "5-class" : 
        
        data_folder = more_data_folder #+ "without_removing_emoticons"
        
        filename =more_data_folder + "annotated_pos_class_our_corpus.csv" #"annotated_pos_class_our_corpus.csv" #"new_pos_class_no_emot_sen140.csv"
        train5 = pd.read_csv(filename, header=0, quoting=3,delimiter="\t")
        positives = train5['sentiment'] == 0
        df_pos = train5[positives]
        train5p = df_pos[1:N0]
        train5p.insert(0,'num','1234567')
        #if "annotated_neg_class_our_corpus" in filename:
        filename =more_data_folder + "annotated_neg_class_our_corpus.csv"  #"new_neg_class_no_emot_sen140.csv" #"annotated_neg_class_our_corpus.csv" 
        train5 = pd.read_csv(filename, header=0, quoting=3,delimiter="\t")
        negatives = train5['sentiment'] == 1
        df_pos = train5[negatives]
        train5n = df_pos[1:N1]
        train5n.insert(0,'num','1234567')
        #if "annotated_neutrals_our_corpus" in filename:
        filename =more_data_folder +"new_neutrals_our_corpus.csv" 
        train5 = pd.read_csv(filename, header=0, quoting=3,delimiter="\t")
        neutrals = train5['sentiment'] == 2
        df_pos = train5[neutrals]
        train5ne = df_pos[1:N2]
        #df.insert(idx, col_name, value)
        train5ne.insert(0,'num','1234567')
        print N2
        print train5p.shape
        print train5n.shape
        print train5ne.shape
        
        return train5p,train5n ,train5ne 

    else: 
        data_folder = more_data_folder #+ "without_removing_emoticons"
        
        filename =more_data_folder + "train_data_for_subtaskCE.csv" 
        train = pd.read_csv(filename, header=None, quoting=3,delimiter="\t")
        #train.insert(0,'num','1234567')
        
        very_negatives = train[len(train.columns)-2] == -2
        negatives = train[len(train.columns)-2] == -1
        neutrals = train[len(train.columns)-2] == 0
        positives = train[len(train.columns)-2] == 1
        very_positives = train[len(train.columns)-2] == 2
        
        df_pos = train[positives]
        df_neg = train[negatives]
        df_neu = train[neutrals]
        df_very_pos = train[very_positives]
        df_very_neg = train[very_negatives]
        
        return df_pos,df_neg,df_neu,df_very_pos,df_very_neg 


def read_train_datasets(mode, data_train,data_Test):
    print data_Test
    if mode != "5-class" : 
        #PARSE semeval train sentences first
        if os.path.exists(data_train) :
          train =  pd.read_csv( data_train, header=None,  delimiter="\t", quoting=csv.QUOTE_NONE)
       
          positives = train[len(train.columns)-2] == "positive"
          negatives = train[len(train.columns)-2] == "negative"
          neutrals = train[len(train.columns)-2] == "neutral"
          df_pos = train[positives]
          df_neg = train[negatives]
          df_neu = train[neutrals]
          #print df_pos.shape
          #print df_neg.shape
          #print df_neu.shape
       
        if os.path.exists(data_Test) :
          Test =  pd.read_csv( data_Test, header=None,  delimiter="\t", quoting=csv.QUOTE_NONE) #read the official test dataset also here
       
        return df_pos,df_neg,df_neu,Test
	
    else:
        #PARSE semeval train sentences first
        if os.path.exists(data_train) :
          train =  pd.read_csv( data_train, header=None,  delimiter="\t", quoting=csv.QUOTE_NONE)
        
          very_negatives = train[len(train.columns)-2] == -2
          negatives = train[len(train.columns)-2] == -1
          neutrals = train[len(train.columns)-2] == 0
          positives = train[len(train.columns)-2] == 1
          very_positives = train[len(train.columns)-2] == 2
        
          df_pos = train[positives]
          df_neg = train[negatives]
          df_neu = train[neutrals]
          df_very_pos = train[very_positives]
          df_very_neg = train[very_negatives]
        
        
        
        if os.path.exists(data_Test) :
          Test =  pd.read_csv( data_Test, header=None,  delimiter="\t", quoting=csv.QUOTE_NONE) #read the official test dataset also here
        
        return df_pos,df_neg,df_neu,df_very_pos,df_very_neg,Test
	

def build_data_cv_distant(mode , more_data_folder,N0,N1,N2 , cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    test_revs = []
    run_all=1 #change this to run for 2 step process TODO
    if (mode != "5-class") : 
        df_pos,df_neg , df_neu= read_otherdata_datasets(mode,more_data_folder,N0,N1,N2)
        pos_file = df_pos["review"]
        neg_file = df_neg["review"]
        if mode != "pos-neg":  neu_file = df_neu["review"]

    else: 
        df_pos,df_neg,df_neu,df_very_pos,df_very_neg= read_otherdata_datasets(mode,more_data_folder,N0,N1,N2)
        very_pos_file = df_very_pos[len(df_very_pos.columns)-1]#always get the line with the review
        very_neg_file = df_very_neg[len(df_very_neg.columns)-1]
        pos_file = df_pos[len(df_pos.columns)-1]
        neg_file = df_neg[len(df_neg.columns)-1]
        neu_file = df_neu[len(df_neu.columns)-1]

    vocab = defaultdict(float)
    if (mode != "5-class") : 
        for line in pos_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} 
            revs.append(datum)
        
        for line in neg_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)
        if mode != "pos-neg":
            for line in neu_file:       
                rev = []
                if (isfloat(line)): continue 
                rev.append(line.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":2, 
                          "text": orig_rev,                             
                          "num_words": len(orig_rev.split()),
                          "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
                revs.append(datum)
    else: #5class needs different data manipulation
 
        for line in pos_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} 
            revs.append(datum)
        for line in very_pos_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} 
            revs.append(datum)

    #with open(neg_file, "rb") as f:
        for line in neg_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":-1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)

        for line in very_neg_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":-2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)

        for line in neu_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)

    return revs,  vocab



def build_data_cv(mode,data_folder,data_Test, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    test_revs = []
    run_all=1 #change this to run for 2 step process TODO
    if (mode !="5-class") : df_pos,df_neg,df_neu,Test= read_train_datasets(mode,data_folder,data_Test)
    else: 
        df_pos,df_neg,df_neu,df_very_pos,df_very_neg,Test= read_train_datasets(mode,data_folder,data_Test)
        very_pos_file = df_very_pos[len(df_very_pos.columns)-1]
        very_neg_file = df_very_neg[len(df_very_neg.columns)-1]

    pos_file = df_pos[len(df_pos.columns)-1]
    neg_file = df_neg[len(df_neg.columns)-1]
    neu_file = df_neu[len(df_neu.columns)-1]

    test_file = Test[len(Test.columns)-1]

    vocab = defaultdict(float)
    if (mode != "5-class"):

        for line in pos_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} 
            revs.append(datum)

        for line in neg_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)

        for line in neu_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)

        for line in test_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"text": orig_rev,                             
                      "num_words": len(orig_rev.split())}
            test_revs.append(datum)

    else: #5-class needs different data preperation
 
        for line in pos_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} 
            revs.append(datum)
        for line in very_pos_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} 
            revs.append(datum)


        for line in neg_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":-1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)
        for line in very_neg_file:       
            rev = []
            if (isfloat(line)): continue 
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":-2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)

        for line in neu_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)} #to split gia ta test prepei na einai cv
            revs.append(datum)

        for line in test_file:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"text": orig_rev,                             
                      "num_words": len(orig_rev.split())}
            test_revs.append(datum)

    return revs, test_revs , vocab
    
def get_W(word_vecs, k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab,  k , min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":     
    w2v_file = sys.argv[1]
    fname = w2v_file.split('/')[-1] # input corpus
    if "Google" in fname : name='google'
    if "tweester" in fname : name='tweester'
    #mode = "neutral-vs-all" or "pos-neg"
    mode = sys.argv[2]
    #subtask="A" or subtask ="B"
    subtask=sys.argv[3]
    N0=int(sys.argv[4])
    N1=int(sys.argv[5])
    N2=int(sys.argv[6])
    testfile = sys.argv[7]
    K = int(sys.argv[8])
    savefile = str (sys.argv[9])

    if (subtask == "A"):
        data_folder = ["../DATA/DATA_SubtaskA/train2017.tsv",testfile] 
        print N2		
    if (subtask == "B" ): #subtask B is always pos-neg
        N2=0 #N2 should always be 0 for subtaskB
        data_folder = ["../DATA/DATA_SubtaskBD/train136-pos-neg.csv" , testfile] 
        #data_folder = ["/usb_disk/twitter/Tweester-Journal-exp/DATA/DATA_SubtaskBD/traindata_subtaskA_without_neutrals/train136-pos-neg.csv" , testfile]  
		
    if (subtask == "CE"):
        data_folder = ["../DATA/DATA_SubtaskCE/train2017.tsv",testfile]    
        
    more_data_folder="../DATA/otherdatasets/" 
    print "loading data...",        
    print "pickle file to load is : " + str(savefile)
    if not os.path.isfile(savefile) :
        distant_revs, distant_vocab = build_data_cv_distant(mode,more_data_folder,N0,N1,N2 , cv=10, clean_string=True)
        train_revs, test_revs, vocab = build_data_cv(mode,data_folder[0],data_folder[1],cv=10, clean_string=True)
        
        max_l = max (np.max(pd.DataFrame(train_revs)["num_words"]) , np.max(pd.DataFrame(test_revs)["num_words"])) #zero padding for a All sentence train+test evaluation (check if this is ok to do)
        distant_max_l = np.max(pd.DataFrame(distant_revs)["num_words"])
        
        print "data loaded!"
        print "number of sentences: " + str(len(train_revs))
        print "vocab size: " + str(len(vocab))
        print "max sentence length: " + str(max_l)
        print "loading word2vec vectors...",
        w2v = load_bin_vec(w2v_file, vocab)
        distant_w2v = load_bin_vec(w2v_file, distant_vocab)
        
        print "word2vec loaded!"
        print "num words already in word2vec: " + str(len(w2v))
        add_unknown_words(w2v, vocab,k= K , min_df=1)
        add_unknown_words(distant_w2v, distant_vocab,k= K , min_df=1)
        
        W, word_idx_map = get_W(w2v,k = K )
        distant_W, distant_word_idx_map = get_W(distant_w2v,k = K )
        
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab, k= K , min_df=1)
        W2, _ = get_W(rand_vecs ,k = K )
        out = open(savefile, "wb")
        cPickle.dump([train_revs, test_revs, W, W2, word_idx_map, vocab , distant_revs, distant_vocab ,distant_W,distant_word_idx_map], out)
        #cPickle.dump([train_revs, test_revs, W, W2, word_idx_map, vocab], open(savefile, "wb"))
        print "dataset created!"
        out.close()
    else:
        print "dataset exists proceed to next step"
