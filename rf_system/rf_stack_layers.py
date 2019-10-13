#first run word_to_vec.py to get the model
#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.9735&rep=rep1&type=pdf
#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.5074&rep=rep1&type=pdf

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
from sklearn.neighbors import KNeighborsClassifier
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from rf_system_functions import *
import time

#start = time.time()
#time.sleep(8200)

#***************************************************************************#
#word2vec model parameters
modelname = sys.argv[1] #full path
num_features = int(sys.argv[2])    # Word vector dimensionality                   


numpos = int(sys.argv[3]) #select how many lines of aditional training data we are going to use
numneg = int(sys.argv[4])
numneu = int(sys.argv[5]) #use a number for neutras even for subtaskBD - at last the script will not take neutrals under consideration for BD

Task = sys.argv[6] #use A,BD,CE

#***************************************************************************#

#forest = RandomForestClassifier( n_estimators = 100 ,n_jobs = 5) 
from sklearn import svm
forest=svm.SVC(kernel='linear',probability=True) # linear, poly, rbf, sigmoid, precomputed , see doc  


#***************************************************************************#
## this code uses the following parameters - for semeval change only the subtask parameter

RF_path ="./trained_models/"  #path to store a Random forest trainned model

Results_path = "./results/" #path to store classification results 
Post_path = "./posteriors/" #path to store posteriors   

if Task == 'A' : subtask = ".subtask-A" #to keep tha same format as the baseline features
if Task == 'BD' : subtask = ".subtask-BD"
if Task == 'CE' : subtask = ".subtask-CE"
if subtask == ".subtask-A" : 
    traindata_path =  "../DATA/DATA_SubtaskA/" #where to find the trainning data
    #traindata_name = "train2017" #trainning data file name format
    #testdata_name="test2017"
    traindata_name = "train13-16" #trainning data file name format
    testdata_name="test2016"
    EVALUATION_FOLDER = '../EVALUATION/systems/subtaskA/sim_stacking_test2016/'
    if not os.path.exists(EVALUATION_FOLDER):
        os.makedirs(EVALUATION_FOLDER)
    make_sure_path_exists(EVALUATION_FOLDER)

if subtask == ".subtask-BD" : 
    traindata_path =  "../DATA/DATA_SubtaskBD/" #where to find the trainning data
    traindata_name = "train2017" #trainning data file name format - this is defined in the script that creates the word2vec features
    testdata_name="test2017"
    EVALUATION_FOLDERB='../EVALUATION/systems/subtaskB/sim_stacking_test2017/'
    EVALUATION_FOLDERD='../EVALUATION/systems/subtaskD/sim_stacking_test2017/'
    if not os.path.exists(EVALUATION_FOLDERB):
        os.makedirs(EVALUATION_FOLDERB)
    make_sure_path_exists(EVALUATION_FOLDERB)
	
    if not os.path.exists(EVALUATION_FOLDERD):
        os.makedirs(EVALUATION_FOLDERD)
    make_sure_path_exists(EVALUATION_FOLDERD)

if subtask == ".subtask-CE":
    traindata_path =  "../DATA/DATA_SubtaskCE/" #whre to find the trainning data
    traindata_name = "train2017" #trainning data file name format
    testdata_name= "test2017" 
    EVALUATION_FOLDERC='../EVALUATION/systems/subtaskC/sim_stacking_test2017/'
    EVALUATION_FOLDERE='../EVALUATION/systems/subtaskE/sim_stacking_test2017/'
    if not os.path.exists(EVALUATION_FOLDERC):
        os.makedirs(EVALUATION_FOLDERC)
    make_sure_path_exists(EVALUATION_FOLDERC)
	
    if not os.path.exists(EVALUATION_FOLDERE):
        os.makedirs(EVALUATION_FOLDERE)
    make_sure_path_exists(EVALUATION_FOLDERE)


data_sep ="\t"  # use \t for tab delimeted data or "," for comma delimeted data. This option is also used in the results file format

path_our_corpus = "/usb_disk/twitter/sasa/create_corpus/"   #if additional data are going to be used , specify their path. keep the additional data in a specific format                    
w2vec_features_path = "../FEATURES/RF_system/" #the path to store and read features from
features_path = w2vec_features_path
features_path_sep="\t" #the format of the features files
w2vec_features_path = w2vec_features_path + "numfeatures_" + str(num_features) +"/"



train_pattern =  "RF_" + traindata_name + subtask+ "_w2vec.csv" #this is the file name where we save the word2vec features
test_pattern =  "RF_" + testdata_name + subtask+ "_w2vec.csv" #this is the file name where we save the word2vec features


model_save_filename = RF_path + "RandomForest_"+subtask+"_NN"+str(num_features)+traindata_name+"_"+str(numpos)+"_pos"+str(numneg)+"_neg"+str(numneu)+"_neu" +".pkl"
filename = Results_path + "Results_" +subtask+"_NN"+str(num_features)+traindata_name+"_"+str(numpos)+"_pos"+str(numneg)+"_neg" +str(numneu)+"_neu"+ "_polarity"+".csv"
filename_post = Post_path + "Results_" +subtask+"_NN"+str(num_features)+traindata_name+"_"+str(numpos)+"_pos"+str(numneg)+"_neg" +str(numneu)+"_neu"+ "_posteriors" +".csv"

#***************************************************************************#


loaded = 0 #used to check if word2vec is loaded
##################START - PROCESS####################################
#Read training data data from files 
features = os.path.join(w2vec_features_path ,train_pattern) #check if semeval train data exists
print features
if os.path.exists(features): #the features exist, just read them 
    mode = 'train'
    trainDataVecs,labels  = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path,mode)
    #affin_train, affin_test = read_affin_features(features_path,traindata_name , subtask)

else :  #this means that the semeval features are not created , we must run the script to create them 
     print "--Creating features for semeval train dataset--"
     model = Word2Vec.load(modelname)
     loaded = 1
     mode = 'train'
     create_w2vec_features (num_features,traindata_path+traindata_name+'.tsv',w2vec_features_path,subtask,modelname)
     trainDataVecs,labels  = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path,mode)

features = os.path.join(w2vec_features_path ,test_pattern) #check if semeval train data exists
print features
if os.path.exists(features): #the features exist, just read them 
     mode = 'test'
     testDataVecs = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path,mode)         
else :  #this means that the semeval features are not created , we must run the script to create them 
     print "--Creating features for semeval test dataset--"
     if loaded == 0 : model = Word2Vec.load(modelname)
     create_w2vec_features (num_features,traindata_path+testdata_name+'.tsv',w2vec_features_path,subtask,modelname)
     mode = 'test'
     testDataVecs  = read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,w2vec_features_path,traindata_path,mode)

   
if (numpos!=0 and numneg!=0 and numneu!=0) :
    print "***features from another dataset******"
    dataset='entweet'
    if os.path.exists(features): model = Word2Vec.load(modelname) #means that features existed so no word2vec was loaded before
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
            else : other_w2vec_frames_pos, other_labels_pos =read_other_features(run,num,w2vec_features_path)

        if run == "neg" :       
            num = numneg
            othersN = w2vec_features_path +"RF_"+str(run) +"_"+str(numneg)+"_"+ dataset + "_w2vec.csv"
            if (not os.path.exists(othersN)): create_w2vec_other_features(run, num_features,path_our_corpus,w2vec_features_path,model,num)
            other_w2vec_frames_neg, other_labels_neg =read_other_features(run,num,w2vec_features_path)

        if run == "neu" :
            num = numneu
            othersNEU = w2vec_features_path +"RF_"+str(run) +"_"+str(numneu)+"_"+ dataset + "_w2vec.csv"
            if (not os.path.exists(othersNEU)): create_w2vec_other_features(run, num_features,path_our_corpus,w2vec_features_path,model,num) 
            other_w2vec_frames_neu, other_labels_neu =read_other_features(run,num,w2vec_features_path)

    if (subtask == ".subtask-A") : 
        Other_Train = np.concatenate([other_w2vec_frames_pos , other_w2vec_frames_neg,other_w2vec_frames_neu])
        Other_labels = other_labels_pos+other_labels_neg+ other_labels_neu
    if (subtask == ".subtask-BD") : 
        Other_Train = np.concatenate([other_w2vec_frames_pos , other_w2vec_frames_neg])
        Other_labels = other_labels_pos+other_labels_neg

    print Other_Train.shape
    print trainDataVecs.shape
    print len(Other_labels)
    print len(labels)

    trainDataVecs=np.concatenate([trainDataVecs, Other_Train]) #
    labels+=Other_labels 
    print len(labels)

if subtask == ".subtask-BD":
    print "******fitting a random forest to labeled training data...******"
    #all features are loaded -run training and classification process
     
    #forest = KNeighborsClassifier()      
    forest.fit( trainDataVecs,  labels )
    #run classification process
    print "******running classification******"
    posteriors_taskB = forest.predict_proba(testDataVecs)
    results_taskB = forest.predict(testDataVecs)

    #read the test file to get the second column
    for root, directories, filenames in os.walk(traindata_path): 
        for fname in filenames:
            if (testdata_name in fname):
                test_data = pd.read_csv( os.path.join(root,fname), header=None,  delimiter="\t" , quoting=3 ) #this is the file with the features that has a header
    test_column=[]
    for topic in  test_data[1]  :
        test_column.append (topic)
    ItemID=[]
    for i in  test_data[0]  :
         ItemID.append (i)
 
    final_result = []
    for res in results_taskB:
        if res == 0 : sent = 'positive'
        if res == 1 : sent = 'negative'
        final_result.append(sent)

    output = pd.DataFrame({ 'id':ItemID ,'nan':test_column,'sentiment': final_result})
    output.to_csv(filename, header=False ,index=False,sep='\t')
    #np.savetxt(filename_post , posteriors_taskB , delimiter='\t');
    ##COPY posteriors to EVALUATION FOLDER    
    #command = 'cp '+filename_post+" " + EVALUATION_FOLDERB+'/posteriors.csv' 
    #os.system(command)
    #command = 'cp '+filename_post+" " + EVALUATION_FOLDERD+'/posteriors.csv' 
    #os.system(command)

    #os.chdir ('./scoring')
    #command = 'perl SemEval2016_task4_test_scorer_subtaskB.pl ' +'.'+ filename
    #os.system(command)
    #command = 'perl create_file_forD.pl '+'.'+ filename
    #os.system(command)
    #command = 'perl SemEval2016_task4_test_scorer_subtaskD.pl ' +'.'+ filename+'D'
    #os.system(command)
	
#trainDataVecs=np.column_stack([trainDataVecs, affin_train])
#testDataVecs=np.column_stack([testDataVecs, affin_test])
#print trainDataVecs.shape
if subtask == ".subtask-A":
    print "======splitting into three different models====="
    print "1. pos-vs-neg"
    
    positive_indexes = [i for i,x in enumerate(labels) if x ==  0]
    negative_indexes = [i for i,x in enumerate(labels) if x == 1]
    pos_labels = [labels[i] for i in positive_indexes]
    neg_labels = [labels[i] for i in negative_indexes]
    
    ytrain_pos = trainDataVecs[([positive_indexes])]
    ytrain_neg = trainDataVecs[([negative_indexes])]
    
    ytrain=np.concatenate([ytrain_pos, ytrain_neg])
    xtrain = pos_labels + neg_labels
    print ytrain.shape

    print "******fitting a random forest to labeled training data...******"
    #all features are loaded -run training and classification process
    #forest = RandomForestClassifier( n_estimators = 100 ,n_jobs = 5) 
    #forest = KNeighborsClassifier()
           
    forest.fit( ytrain,  xtrain )
    #run classification process
    print "******running classification******"
    print testDataVecs.shape
    posteriors_stackA = forest.predict_proba(testDataVecs)
    train_posteriors_stackA = forest.predict_proba(trainDataVecs)
    resultA = forest.predict(testDataVecs)
    
    print "======splitting into three different models====="
    print "2. pos-vs-neutral"
    positive_indexes = [i for i,x in enumerate(labels) if x ==  0]
    neutral_indexes = [i for i,x in enumerate(labels) if x == 2]
    
    pos_labels = [labels[i] for i in positive_indexes]
    neu_labels = [labels[i] for i in neutral_indexes]
    
    ytrain_pos = trainDataVecs[([positive_indexes])]
    ytrain_neu = trainDataVecs[([neutral_indexes])]
    
    ytrain=np.concatenate([ytrain_pos, ytrain_neu])
    xtrain = pos_labels + neu_labels
    
    print "******fitting a random forest to labeled training data...******"
    #all features are loaded -run training and classification process
    #forest = RandomForestClassifier( n_estimators = 100 ,n_jobs = 5) 
    #from sklearn import svm
    #forest=svm.SVC(kernel='linear') 	
    #forest = KNeighborsClassifier()      #we could also use KNN here
    forest.fit( ytrain,  xtrain )
    #run classification process
    print "******running classification******"
    posteriors_stackB = forest.predict_proba(testDataVecs)
    train_posteriors_stackB = forest.predict_proba(trainDataVecs)
    resultB = forest.predict(testDataVecs)
    
    print "======splitting into three different models====="
    print "3. negative-vs-neutral"
    negative_indexes = [i for i,x in enumerate(labels) if x ==  1]
    neutral_indexes = [i for i,x in enumerate(labels) if x == 2]
    
    neg_labels = [labels[i] for i in negative_indexes]
    neu_labels = [labels[i] for i in neutral_indexes]
    
    ytrain_neg = trainDataVecs[([negative_indexes])]
    ytrain_neu = trainDataVecs[([neutral_indexes])]
    
    ytrain=np.concatenate([ytrain_neg, ytrain_neu])
    xtrain = neg_labels + neu_labels
    
    print "******fitting a random forest to labeled training data...******"
    #all features are loaded -run training and classification process
    #forest = RandomForestClassifier( n_estimators = 100 ,n_jobs = 5)  
    
	#forest = KNeighborsClassifier()      
    forest.fit( ytrain,  xtrain )
    #run classification process
    print "******running classification******"
    posteriors_stackC = forest.predict_proba(testDataVecs)
    resultC = forest.predict(testDataVecs)
    train_posteriors_stackC = forest.predict_proba(trainDataVecs)
    print "======Combine results of the three different models====="

    print "======KNeighborsClassifier====="    
    
    neigh = KNeighborsClassifier()
    
    Train = np.column_stack((train_posteriors_stackA , train_posteriors_stackB, train_posteriors_stackC))
    neigh.fit(Train, labels)
    Test =  np.column_stack((posteriors_stackA , posteriors_stackB,posteriors_stackC))
    result = neigh.predict(Test)
    
    result_post = neigh.predict_proba(Test) 

    np.savetxt(filename_post , result_post , delimiter='\t');
    command = 'cp '+filename_post+" " + EVALUATION_FOLDER+'/posteriors.csv' 
    os.system(command)

    
    ItemID=[]
    for i in range(0,len(testDataVecs)):
        ItemID .append( i + 1)# + 11378)
    filename = filename +'neigh'
    output = pd.DataFrame({ 'id':ItemID ,'nan':'NA','sentiment': result})
    output.to_csv(filename, header=False ,index=False,sep='\t')
	
if subtask == '.subtask-CE':

    print "======splitting into 10 different models====="
    # (0,-1), (0,1), (0,2), (0,-2) 
    # (-1,2), (-1,-2), (-1,1) 
    # (1,2),  (1,-2) 
    # (-2,2)
     
    case1 = [0,-1]
    case2 = [0,1]
    case3 = [0,2]
    case4 = [0,-2]
    case5 = [-1,2]
    case6 = [-1,-2]
    case7 = [-1,1]
    case8 = [1,2]
    case9 = [1,-2]
    case10 = [-2,2]
    
    cases = [case1,case2,case3,case4,case5,case6,case7,case8,case9,case10];
    
    run=0
    for case in cases:
        class1 = case[0]
        class2 = case[1]
        
        positive_indexes = [i for i,x in enumerate(labels) if x ==  class1]
        negative_indexes = [i for i,x in enumerate(labels) if x == class2 ]
        pos_labels = [labels[i] for i in positive_indexes]
        neg_labels = [labels[i] for i in negative_indexes]
        
        ytrain_pos = trainDataVecs[([positive_indexes])]
        ytrain_neg = trainDataVecs[([negative_indexes])]
        
        ytrain=np.concatenate([ytrain_pos, ytrain_neg])
        xtrain = pos_labels + neg_labels
        print ytrain.shape
        
        print "******fitting a random forest to labeled training data...******"
        #all features are loaded -run training and classification process
        forest = KNeighborsClassifier() #RandomForestClassifier( n_estimators = 100 ,n_jobs = 5) 
        forest.fit( ytrain,  xtrain )
        #run classification process
        print "******running classification******"
        posteriors_stackA = forest.predict_proba(testDataVecs)
        train_posteriors_stackA = forest.predict_proba(trainDataVecs)
        filename_post_test = filename_post + "_test"+str(run)
        filename_post_train = filename_post + "_train"+str(run)
        np.savetxt(filename_post_test , posteriors_stackA , delimiter="\t")
        np.savetxt(filename_post_train , train_posteriors_stackA, delimiter="\t")
        run = run+1
    
    print "======Combine results of the three different models====="
    trainDataVecs = pd.DataFrame() 
    testDataVecs = pd.DataFrame()
    
    for run in range(0,9):
        filename_post_test = filename_post + "_test"+str(run)
        filename_post_train = filename_post + "_train"+str(run)
        train_data = pd.read_csv( filename_post_train, header=None,  delimiter="\t" , quoting=3 ) 
        test_data = pd.read_csv( filename_post_test, header=None,  delimiter="\t" , quoting=3 )
        trainDataVecs = pd.concat([train_data , trainDataVecs],axis=1) 
        testDataVecs = pd.concat([test_data , testDataVecs],axis=1) 
    
        
    trainDataVecs=np.asarray(trainDataVecs)
    testDataVecs=np.asarray(testDataVecs)
    print trainDataVecs.shape
    print testDataVecs.shape
    print "======KNeighborsClassifier====="
    #open the posteriors of all the saved binary models
    
    neigh = KNeighborsClassifier()
    neigh.fit(trainDataVecs, labels) 
    result = neigh.predict(testDataVecs)
    result_post = neigh.predict_proba(testDataVecs)
    np.savetxt(filename_post +'neigh' , result_post , delimiter='\t');
    
    #read the test file to get the second column
    for root, directories, filenames in os.walk(traindata_path): 
        for fname in filenames:
            if (testdata_name in fname):
                test_data = pd.read_csv( os.path.join(root,fname), header=None,  delimiter="\t" , quoting=3 ) #this is the file with the features that has a header
    test_column=[]
    for topic in  test_data[1]  :
        test_column.append (topic)
    ItemID=[]
    for i in  test_data[0]:  
         ItemID.append (i)
    
    output = pd.DataFrame({ 'id':ItemID ,'nan':test_column,'sentiment': result})
    output.to_csv(filename, header=False ,index=False,sep='\t')
    np.savetxt(filename_post , result_post , delimiter='\t');
	
	
    #COPY posteriors to EVALUATION FOLDER    
    command = 'cp '+filename_post+" " + EVALUATION_FOLDERC+'/posteriors.csv' 
    os.system(command)
    command = 'cp '+filename_post+" " + EVALUATION_FOLDERE+'/posteriors.csv' 
    os.system(command)

	
    #os.chdir ('./scoring')
    
    #command = 'perl SemEval2016_task4_test_scorer_subtaskC.pl ' +'.'+ filename
    #os.system(command)
    #command = 'perl create_file_forE.pl '+'.'+ filename
    #os.system(command)
    #print '.'+ filename+'E'
    #command = 'perl SemEval2016_task4_test_scorer_subtaskE.pl ' +'.'+ filename+'E'
    #os.system(command)