#first run word_to_vec.py to get the model

import csv
# Load a pre-trained model
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import scale
from nltk.tokenize import TweetTokenizer
import pickle
import numpy as np
import os
from sklearn.preprocessing import Imputer
import math

import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from rf_system_functions import *

#***************************************************************************#
#word2vec model parameters
modelname = sys.argv[1] #full path
num_features = int(sys.argv[2])    # Word vector dimensionality                   
model = Word2Vec.load(modelname)

numpos = int(sys.argv[3]) #select how many lines of aditional training data we are going to use
numneg = int(sys.argv[4])
numneu = int(sys.argv[5]) #use a number for neutras even for subtaskBD - at last the script will not take neutrals under consideration for BD
others = False #use this variable to tggle between adding aditional training features or not
#featrues and results files formats


#***********USE THIS for Charis thesis: score_A_2016_original.pl****************************#


#***************************************************************************#
## this code uses the following parameters - for semeval change only the subtask parameter

RF_path ="./trained_models/"  #path to store a Random forest trainned model

Results_path = "./results/" #path to store classification results 
Post_path = "./posteriors/" #path to store posteriors   

subtask = ".subtask-A" #to keep tha same format as the baseline features
#subtask = ".subtask-BD"
#subtask = ".subtask-CE"
traindata_path =  "/usb_disk/twitter/Tweester-Journal-exp/DATA/DATA_SubtaskA/" #whre to find the trainning data
traindata_name = "train13-16" #trainnind data file name format
testdata_name='test'
data_sep ="\t"  # use \t for tab delimeted data or "," for comma delimeted data. This option is also used in the results file format

path_our_corpus = "/usb_disk/twitter/sasa/create_corpus/"   #if additional data are going to be used , specify their path. keep the additional data in a specific format                    
w2vec_features_path = "/usb_disk/twitter/Tweester-Journal-exp/FEATURES/RF_system/" #the path to store and read features from
features_path_sep="\t" #the format of the features files
w2vec_features_path = w2vec_features_path + "numfeatures_" + str(num_features) +"/"



train_pattern =  "RF_" + traindata_name + subtask+ "_w2vec.csv" #this is the file name where we save the word2vec features


model_save_filename = RF_path + "RandomForest_"+"_NN"+str(num_features)+traindata_name+"_"+str(numpos)+"_pos"+str(numneg)+"_neg"+str(numneu)+"_neu" +".pkl"
filename = Results_path + "Results_" +"_NN"+str(num_features)+traindata_name+"_"+str(numpos)+"_pos"+str(numneg)+"_neg" +str(numneu)+"_neu"+ "_polarity"+".csv"
filename_post = Post_path + "Results_" +"_NN"+str(num_features)+traindata_name+"_"+str(numpos)+"_pos"+str(numneg)+"_neg" +str(numneu)+"_neu"+ "_posteriors" +".csv"

train_pattern =  "RF_" + traindata_name + subtask+ "_w2vec.csv" #this is the file name where we save the word2vec features

#***************************************************************************#



##################START - PROCESS####################################
#Read training data data from files 
features = os.path.join(w2vec_features_path ,train_pattern) #check if semeval train data exists

print features

print traindata_name
print subtask
print num_features
print w2vec_features_path
print traindata_path

if os.path.exists(features): #the features exist, just read them 
#traindata_name,testdata_name,subtask,num_features,features_path_general,traindata_path, mode
    mode ='train'
    trainDataVecs,labels = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path, mode )
    mode ='test'
    testDataVecs = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path, mode )
else :  #this means that the semeval features are not created , we must run the script to create them 
    print "--Creating features for semeval train and test dataset--"
    if subtask == ".subtask-A":task ='A'
    if subtask == ".subtask-BD":task ='BD'
    if subtask == ".subtask-CE":task ='CE'
    create_w2vec_features (num_features,traindata_path,w2vec_features_path,task,modelname)
    #now we must read them
    mode ='train'
    trainDataVecs,labels = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path, mode )
    mode ='test'
    testDataVecs = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path, mode )

#mode ='test'
#if os.path.exists(features): #the features exist, just read them 
#    trainDataVecs,labels,testDataVecs = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path, mode )
#else :  #this means that the semeval features are not created , we must run the script to create them 
#     print "--Creating features for semeval train and test dataset--"
#     if subtask == ".subtask-A":task ='A'
#     if subtask == ".subtask-BD":task ='BD'
#     if subtask == ".subtask-CE":task ='CE'
#     create_w2vec_features (num_features,traindata_path,w2vec_features_path,task,modelname)
#     #now we must read them
#     trainDataVecs,labels,testDataVecs = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path, mode )         

	 
if (numpos!=0 and numneg!=0 and numneu!=0) :
    print "***features from another dataset******"
    dataset='entweet'
    other_w2vec_frames_pos = pd.DataFrame()
    other_w2vec_frames_neg = pd.DataFrame()
    other_w2vec_frames_neu = pd.DataFrame()
    other_labels_pos=[]
    other_labels_neg=[]
    other_labels_neu=[]
    if (subtask == ".subtask-BD") : 
        runs=["pos","neg"]
    else : 
        runs=["pos","neg","neu"]
    for run in runs:
        if run == "pos" : 
            num = numpos
            othersP = w2vec_features_path +"RF_"+str(run) +"_"+str(numpos)+"_"+ dataset + "_w2vec.csv"
            if (not os.path.exists(othersP)) : create_w2vec_other_features(run, num_features,path_our_corpus,w2vec_features_path,model,num)
            other_w2vec_frames_pos, other_labels_pos =read_other_features(run,num,w2vec_features_path)

        if run == "neg" :       
            num = numneg
            othersN = w2vec_features_path +"RF_"+str(run) +"_"+str(numneg)+"_"+ dataset + "_w2vec.csv"
            if (not os.path.exists(othersN)): create_w2vec_other_features(run, num_features,path_our_corpus,w2vec_features_path,model,num)
            other_w2vec_frames_neg, other_labels_neg =read_other_features(run,num,w2vec_features_path)

        if run == "neu" :
            num = numneu
            othersNEU = w2vec_features_path +"RF_"+str(run) +"_"+str(numneu)+"_"+ dataset + "_w2vec.csv"
            if (not os.path.exists(othersNEU)): create_w2vec_other_features(run, num_features,path_our_corpus,w2vec_features_path,model,num) 
            else : other_w2vec_frames_neu, other_labels_neu =read_other_features(run,num,w2vec_features_path)

    if (subtask == ".subtask-A") :
        print other_w2vec_frames_pos.shape
        print other_w2vec_frames_neg.shape
        print other_w2vec_frames_neu.shape		
        Other_Train = np.concatenate([other_w2vec_frames_pos , other_w2vec_frames_neg,other_w2vec_frames_neu])
        Other_labels = other_labels_pos+other_labels_neg+ other_labels_neu
    if (subtask == ".subtask-BD") : 
        Other_Train = np.concatenate([other_w2vec_frames_pos , other_w2vec_frames_neg])
        Other_labels = other_labels_pos+other_labels_neg

    print Other_Train.shape
    print trainDataVecs.shape

    trainDataVecs=np.concatenate([trainDataVecs, Other_Train]) #
    labels+=Other_labels 

#check if we have saved trained models
#if os.path.exists(model_save_filename): #if trainned models already exist
#    with open(model_save_filename, 'rb') as f:
#        forest = pickle.load(f)
#else:		
print "******fitting a random forest to labeled training data...******"
    #all features are loaded -run training and classification process
forest = RandomForestClassifier( n_estimators = 100 ,n_jobs = 5)   
#from sklearn import svm
#forest=svm.SVC(kernel='linear') # linear, poly, rbf, sigmoid, precomputed , see doc     

forest.fit( trainDataVecs,  labels ) 
#    output = open(model_save_filename, 'wb')
#    pickle.dump(forest, output)
#    output.close()	
#    with open(model_save_filename, 'wb') as f: 
#        pickle.dump(forest, f)

#run classification process
print "******running classification******"
result = forest.predict(testDataVecs)        
#posteriors = forest.predict_proba(testDataVecs)
#prop = pd.DataFrame(posteriors)
#prop.to_csv(filename_post,header=None, index=False,sep='\t')
#result should be positive/negative/neutral
ItemID=[]
for i in range(0,len(testDataVecs)):
    ItemID .append( i + 1)
output = pd.DataFrame({ 'id':ItemID ,'nan':'NA','sentiment': result})
output.to_csv(filename, header=False ,index=False,sep='\t')


##new_results = []
##posteriors = np.asarray(posteriors)
##for list in posteriors:
##    #print(list)
##    if (list[0] > list[2] > list [1]) :  sentiment = 0
##    elif (list[1] > list[2] > list [0]): sentiment = 1
##    else: sentiment = 2
##
##    new_results.append(sentiment)
##
##ItemID=[]
##for i in range(0,len(testDataVecs)):
##    ItemID .append( i + 1)
##filename = filename +'b'
##output = pd.DataFrame({ 'id':ItemID ,'nan':'NA','sentiment': new_results})
##output.to_csv(filename, header=False ,index=False,sep='\t')

