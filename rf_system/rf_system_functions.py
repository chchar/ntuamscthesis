#!/usr/bin/python3
import csv
# Load a pre-trained model
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
import nltk
nltk.data.path.append('/home/admin_nlp/nltk_data/')
from nltk.corpus import stopwords
from sklearn.preprocessing import scale
from nltk.tokenize import TweetTokenizer
import pickle
import numpy as np
import os,errno
 
from sklearn.preprocessing import Imputer
import math
from multiprocessing import Pool
from multiprocessing import Process
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
reload(sys)  
sys.setdefaultencoding('utf8')

 
pattern_split = re.compile(r"\W+")

def stack_column_wise(average_vectors,vaders_sentiments_pos,vaders_sentiments_neg,affin_sentiment):

    Y1 = np.asarray(affin_sentiment)
    Y=np.column_stack((average_vectors,Y1))
    Y3 = np.asarray(vaders_sentiments_pos)
    Y=np.column_stack((Y,Y3))
    Y3 = np.asarray(vaders_sentiments_neg)
    Y=np.column_stack((Y,Y3))
    DataVecs=Y
    return DataVecs

def use_google_word2vec(w2v_file):
    
    model = Word2Vec.load_word2vec_format(w2v_file, binary=True)
    return model

	
reload(sys)  
sys.setdefaultencoding('utf8')


filenameAFINN = 'AFINN-en-165.txt'
afinn = dict(map(lambda (w, s): (w, int(s)), [ 
            ws.strip().split('\t') for ws in open(filenameAFINN) ]))

			# Word splitter pattern
pattern_split = re.compile(r"\W+")

#for every sentence add the sentiment score from AFINN
def get_sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence. 
    """
    words = pattern_split.split(text.lower())
    sentiments = map(lambda word: afinn.get(word, 0), words)
    if sentiments:
        # How should you weight the individual word sentiments? 
        # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
        
    else:
        sentiment = 0
    return sentiment


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

#compile regular expressions that match repeated characters and emoji unicode
emoji = re.compile(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]',re.UNICODE)
multiple = re.compile(r"(.)\1{1,}", re.DOTALL)

def format(tweet):

    #strip emoji
    stripped = emoji.sub('',tweet)

    #strip URLs
    stripped = re.sub(r'http[s]?[^\s]+','', stripped)

    #strip "@name" components
    stripped = re.sub(r'(@[A-Za-z0-9\_]+)' , "" ,stripped)

    #strip html '&amp;', '&lt;', etc.
    stripped = re.sub(r'[\&].*;','',stripped)

    #strip punctuation
    stripped = re.sub(r'[#|\!|\-|\+|:|//]', " ", stripped)

    #strip the common "RT"
    stripped = re.sub( 'RT.','', stripped)

    #strip whitespace down to one.
    stripped = re.sub('[\s]+' ,' ', stripped).strip()

    #strip multiple occurrences of letters
    stripped = multiple.sub(r"\1\1", stripped)

    #strip all non-latin characters
    #if we wish to deal with foreign language tweets, we would need to first
    #translate them before taking this step.

    stripped = re.sub('[^a-zA-Z0-9|\']', " ", stripped).strip()
    

    return stripped    
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    string = format(review)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", review) 
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
     
    #words = tknzr.tokenize(review_text)
    # 3. Convert words to lower case and split them
    words = string.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    #print len(words)
    return(words)
    
def read_word2vec_features(num_features,mode,pattern,features_path_w2vec,traindata_path , traindata_name,subtask) :
    print mode + "... Reading features now from " + features_path_w2vec
    
    w2vec_frames = pd.DataFrame()
    w2vec_test_frames= pd.DataFrame()
    Train_data = pd.DataFrame() 
    Test_data = pd.DataFrame()
    print "----------pattern----------"
    print pattern
    print "----------pattern----------"
    sentiments = pd.DataFrame()
    for root, directories, filenames in os.walk(features_path_w2vec):
        for filename in filenames:

            if pattern in filename:
                if "w2vec" in filename:
                    Train_data = pd.read_csv( os.path.join(root,filename), header=0,  delimiter="\t" , quoting=3 ) #this is the file with the features that has a header
                    w2vec_frames = pd.concat([Train_data , w2vec_frames])
 
    TrainDataVecs= np.asarray(w2vec_frames)
    print "********* loaded semeval features shape*********"
    print mode
    print TrainDataVecs.shape
    print "*********loaded semeval features shape*********"

    if (mode =="train"):
        #get also the polarity labels for the train files
        print traindata_path
        if (subtask == ".subtask-A"):
            for root, directories, filenames in os.walk(traindata_path):
                for filename in filenames: 
                    if traindata_name in filename:
                        Train_data = pd.read_csv( os.path.join(root,filename), header=None,  delimiter="\t" , quoting=3 ) #this is the original file with the tweets-no header here
                        labels=[]
                        if (subtask == ".subtask-A"):
                            for sentiment in Train_data[2]:
                                if sentiment == "positive" : sent=0
                                if sentiment == "negative" : sent=1
                                if sentiment == "neutral"  : sent=2
                                labels.append( sent )
        if (subtask == ".subtask-CE"):
             for root, directories, filenames in os.walk(traindata_path): 
                for filename in filenames: 
                    if traindata_name in filename:
                        Train_data = pd.read_csv( os.path.join(root,filename), header=None,  delimiter="\t" , quoting=3 ) #this is the original file with the tweets-no header here
                        labels=[]
                        for sentiment in Train_data[2]:
                            labels.append( sentiment )
        if (subtask == ".subtask-BD"):
             for root, directories, filenames in os.walk(traindata_path): 
                for filename in filenames: 
                    if traindata_name in filename:
                        Train_data = pd.read_csv( os.path.join(root,filename), header=None,  delimiter="\t" , quoting=3 ) #this is the original file with the tweets-no header here
                        labels=[]
                        for sentiment in Train_data[2]:
                            if sentiment == "positive" : sent=0
                            if sentiment == "negative" : sent=1
                            labels.append( sent )
        return TrainDataVecs  , labels 

    
    if (mode == "test") : return TrainDataVecs


def read_train_test_w2vec(traindata_name,testdata_name,subtask,num_features,features_path_general,traindata_path, mode ):
    
    if (mode=="train"):
        pattern = "RF_" + traindata_name + subtask
        data = traindata_name 
        trainDataVecs,labels = read_word2vec_features(num_features,mode,pattern,features_path_general,traindata_path , traindata_name,subtask) 
        return trainDataVecs,labels
    if (mode=="test"):
        pattern =  "RF_" + testdata_name + subtask 
        testDataVecs = read_word2vec_features(num_features,mode,pattern,features_path_general,traindata_path , testdata_name,subtask)

        from numpy import float32
        new_data = testDataVecs.astype(float32)
        testDataVecs = new_data
        print "Checkinf for NaN and Inf"
        print "np.inf=", np.where(np.isnan(testDataVecs))
        print "is.inf=", np.where(np.isinf(testDataVecs))
        print "np.max=", np.max(abs(testDataVecs))

        return testDataVecs


def read_affin_features(features_path,traindata_name , subtask) :
    #features_path is one level up from word2vec features path
    mode="train"
    affin_frames = pd.DataFrame()
    Train_data = pd.DataFrame()
    pattern = "RF_" + traindata_name + subtask

    for root, directories, filenames in os.walk(features_path):
        for filename in filenames:
            #print filename
            if pattern in filename:
                if "affin" in filename:
                    Train_data = pd.read_csv( os.path.join(root,filename), header=0,  delimiter="\t" , quoting=3 )
                    affin_frames = pd.concat([Train_data , affin_frames])
    mode="test"
    pattern =  "RF_" + mode + subtask 
    for root, directories, filenames in os.walk(features_path):
        for filename in filenames:
            if pattern in filename:
                if "affin" in filename:
                    test_affin_frames = pd.read_csv( os.path.join(root,filename), header=0,  delimiter="\t" , quoting=3 )
    print affin_frames.shape
    print test_affin_frames.shape
    return affin_frames , test_affin_frames

def create_w2vec_features (num_features,filename,features_path,subtask,modelname):
    #for semeval training and test datasets only 
    retval = os.getcwd()     
    #os.chdir( '/usb_disk/twitter/Tweester-Journal-exp/TOOLS/create_features/w2vec_features/' )
    command = "python w2vec_features.py"+" "+modelname+" "+subtask +" " +str(num_features) +" " + filename +" " + "rf"
    os.system(command)

def w2vec_features_from_files (num_features,subtask,modelname,input_file):
    #for semeval training and test datasets only 
    command = "python w2vec_features.py "+" "+ modelname +" " + subtask +" " +str(num_features) +" " +str(input_file) 
    os.system(command)


def create_affin_features (num_features,traindata_path,features_path):
    retval = os.getcwd()
    os.chdir( '/usb_disk/twitter/Tweester-Journal-exp/TOOLS/create_features/affin_features/' )
    os.system('python create_affin_features.py ')



def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def run_mini_chunks( run , Train_data , model, results_path ,dataset,num_features,num):
    delimiter_results = "\t"

    clean_train_reviews = []
    print results_path
    filename = results_path +"RF_"+str(run) +"_"+str(num)+"_"+ dataset + "_w2vec.csv"

    if (not os.path.exists(filename)):
        print "creating features .... "
        for review in Train_data[2]:
            if not (isfloat(review)): #else:			
                clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
        
        trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
        trainDataVecs = Imputer().fit_transform(trainDataVecs)
        output = pd.DataFrame(trainDataVecs)
        
        #output w2vec
        output.to_csv( filename, sep=delimiter_results, index=False, quoting=3,header=True)
        
def create_w2vec_other_features(run , num_features,path_our_corpus,features_path,w2vecmodel,num):
    ##################################################################################################################
    
    model = w2vecmodel #model is loaded, no need to load it again
    dataset = "entweet" #this means our generic large corpus
    pos="none"
    neg="none"
    neu="none"
    if run == "pos" : pos="pos_class_no_emot_sen140.csv" #we have previosly annotated the downloaded data from tweeter
    if run == "neg" : neg="neg_class_no_emot_sen140.csv"
    if run == "neu" : neu="neutrals_our_corpus.csv"
    results_path=features_path
    for root, directories, filenames in os.walk(path_our_corpus):
        for filename in filenames:
            #print filename
            if pos in filename:                
                train5 = pd.read_csv(os.path.join(root,filename), header=0, quoting=3,delimiter="\t")
                positives = train5['sentiment'] == 0
                df_pos = train5[positives]
                train5p = df_pos[1:num]    
                Y = np.asarray (train5p["id"])
                Y=np.column_stack((Y,train5p["sentiment"]))     
                Y=np.column_stack((Y,train5p["review"]))
                Y5p = pd.DataFrame(Y)
				
            if  neg in filename:
                train5 = pd.read_csv(os.path.join(root,filename), header=0, quoting=3,delimiter="\t")
                negatives = train5['sentiment'] == 1
                df_neg = train5[negatives]
                train5n =  df_neg[1:num]    
                Y = np.asarray (train5n["id"])
                Y=np.column_stack((Y,train5n["sentiment"]))     
                Y=np.column_stack((Y,train5n["review"]))
                Y5n = pd.DataFrame(Y)
				
            if neu in filename:                
                train5 = pd.read_csv(os.path.join(root,filename), header=0, quoting=3,delimiter="\t")
                neutrals = train5['sentiment'] == 2
                df_neu = train5[neutrals]
                train5ne =  df_neu[1:num]    
                Y = np.asarray (train5ne["id"])
                Y=np.column_stack((Y,train5ne["sentiment"]))     
                Y=np.column_stack((Y,train5ne["review"]))
                Y5ne = pd.DataFrame(Y)
        
        #Train_data = pd.concat([Y5p,Y5n,Y5ne])
    
        if run == "pos":
            Train_data = Y5p 
        if run == "neg":
            Train_data = Y5n
        if run == "neu":
            Train_data = Y5ne
        print "----created features ------" 

        run_mini_chunks(run , Train_data , model, results_path ,dataset,num_features,num)

    
def read_other_features( run,num,features_path):
    print "...reading other features function"
    dataset="entweet"
    if ( num!= 0) :
        other_w2vec_frames = pd.DataFrame()
        other_labels=[]
        if run == "pos" : 
           sent=0

        if run == "neg" : 
           sent=1

        if run == "neu" : 
           sent=2

        for root, directories, filenames in os.walk(features_path):
            for filename in filenames: 

                dataW2vec = str(run) +"_"+str(num)+"_"+ dataset+"_w2vec"
                #print dataW2vec
                if dataW2vec in filename:
                    w2vec_frames = pd.read_csv( os.path.join(root,filename), header=0,  delimiter="\t" , quoting=3 )
                    other_w2vec_frames = pd.concat([other_w2vec_frames , w2vec_frames])

        for sentiment in range(0,other_w2vec_frames.shape[0]): #add num times the sentiment value
            other_labels.append( sent )

 
        other_TrainDataVecs  = np.asarray(other_w2vec_frames)        

        return other_TrainDataVecs, other_labels

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

			