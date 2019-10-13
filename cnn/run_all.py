#!/usr/bin/python3

import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import os
import csv
from multiprocessing import Pool
from multiprocessing import Process
import subprocess
import time

start_time = time.time()
##############PARAMETERS
#process data for neutral-vs-all test
#mode = "neutral-vs-all" or "pos-neg"
#subtask="A" or subtask ="B" or "CE" #subtaskB should never be with neutral vs all
#########################
#TRY with both word2vec models , use a different model for each step
N0=int(sys.argv[1])
N1=int(sys.argv[2])
N2=int(sys.argv[3])

epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])
K = int(sys.argv[6])
subtask = sys.argv[7]
print subtask
if (K == 300):
    w2vec_file1 = "/usb_disk/twitter/Tweester-Journal-exp/PretrainedW2V/GoogleNews-vectors-negative300.bin" 
    w2vec_file2 = "/usb_disk/twitter/Tweester-Journal-exp/PretrainedW2V/tweesterjournal_corpus.minc50.iter5.300.w5.cbow.negative.bin" 
if (K==50):
    w2vec_file1 = "../PretrainedW2V/tweesterjournal_corpus.minc50.iter5.50.w5.cbow.negative.bin" 
    w2vec_file2 = "../PretrainedW2V/tweesterjournal_corpus.minc50.iter5.50.w5.cbow.negative.bin" 

hidden_unit = 200
n_epochs = epochs
 
w2vec_file = w2vec_file2

if "Google" in w2vec_file: name='google'
if "tweester" in w2vec_file : name='tweester'
first = name
#*****************step 1
######create the pickle file for neutral-vs-all experiment

if (subtask=="A"):
    num_classes = 3
    mode = "3-class" 
    #testfile ="../DATA/DATA_SubtaskA/test2016.tsv"
    #testyear='2016'
    testfile ="/usb_disk/SEMEVAL2017_TESTDATA/STEP_2_FINAL_READY_BY_SASA/A/test2017.tsv"
    testyear='test2017-b'

    savefile="./input/Subtask"+str(subtask)+str(name)+"_"+str(N0)+"pos_"+str(N1)+"neg_"+str(N2)+"neu"+"_NN"+str(K)+".p"
    
if (subtask=="CE"):
    num_classes = 5
    mode = "5-class" 
    testfile ="../DATA/DATA_SubtaskCE/test2016.tsv"
    
    savefile="./input/Subtask"+str(subtask)+str(name)+"_"+str(N0)+"pos_"+str(N1)+"neg_"+str(N2)+"neu"+"_NN"+str(K)+".p"


if (subtask=="B"):
    num_classes = 2
    #testfile ="../DATA/DATA_SubtaskBD/test2016.tsv"
    #testyear='2016'
    testfile ="/usb_disk/SEMEVAL2017_TESTDATA/STEP_2_FINAL_READY_BY_SASA/BD/test2017.tsv"
    testyear='test2017'

    print "=============Running for subtaskB now=================="
    mode = "pos-neg" 
    both="one"
    w2vec_file = w2vec_file2
    if "Google" in w2vec_file: name='google'
    if "tweester" in w2vec_file : name='tweester'
    if "emoticon" in w2vec_file : name='emo'
    n_epochs=epochs
    N2=0
    print N2

    savefile="./input/Subtask"+str(subtask)+str(name)+"_"+str(N0)+"pos_"+str(N1)+"neg_"+str(N2)+"neu"+"_NN"+str(K)+".p"
 

Command = "python process_data.py " + w2vec_file + " " + mode + " " + subtask + " " +str(N0) + " " +str(N1) + " " + str(N2)+ " " + testfile+ " " + str(K)+ " " +savefile
os.system(Command)

resultsfile = "./results/Subtask"+str(subtask)+"_NN"+str(K)+"_"+str(first)+"_"+str(n_epochs)+"epochs_"+"_"+str(name)+"_"+str(N0)+"pos_"+str(N1)+"neg_"+str(N2)+"neu"+ "_hidden_units" + str(hidden_unit)+"_batch_size"+str(batch_size)+str(testyear)
post_resultsfile = "./posteriors/Subtask"+str(subtask)+"_NN"+str(K)+"_"+str(first)+"_"+str(n_epochs)+"epochs_"+"_"+str(name)+"_"+str(N0)+"pos_"+str(N1)+"neg_"+str(N2)+"neu"+ "_hidden_units" + str(hidden_unit)+"_batch_size"+str(batch_size)+str(testyear)

Command = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,lib.cnmem=1,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec " + savefile + " " + resultsfile + " " + str(n_epochs)+ " " + str(hidden_unit)+ " " + str(K)+ " " + str(batch_size)+" " +post_resultsfile +" " + str(num_classes)
#Command = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv-net-sentence-original.py -nonstatic -word2vec " + savefile + " " + resultsfile + " " + str(n_epochs)+ " " + str(hidden_unit)+ " " + str(K)+ " " + str(batch_size)+" " +post_resultsfile +" " + str(num_classes)
#Command = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence-kim.py -nonstatic -word2vec " + savefile + " " + resultsfile + " " + str(n_epochs)+ " " + str(hidden_unit)+ " " + str(K)+ " " + str(batch_size)+" " +post_resultsfile +" " + str(num_classes)

os.system(Command)


end_time = time.time()
total=end_time-start_time
print "Time taken for CNN " + str(total) 