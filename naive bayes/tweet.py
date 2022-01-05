import pandas as pd                                                        # importing dataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as MB

tweet_data = pd.read_csv("Disaster_tweets_NB.csv",encoding = "ISO-8859-1") # importing dataset

# data cleansing and eda part***********************************************************************************************

tweet_data = tweet_data.drop(["id","keyword","location"], axis = 1)        # dropping nominal columns

tweet_data.duplicated().sum()                                              # checking and remiving duplicate rows                            
tweet_data = tweet_data.drop_duplicates()
tweet_data.info()                                                          # checking for data types and null values

stop_words = []                                                            # importing stop words
with open("stop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):                                                     # removing expressions
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()                             # removing words length less than three
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


tweet_data.text = tweet_data.text.apply(cleaning_text)

tweet_data = tweet_data.loc[tweet_data.text != " ",:]                     # removing empty rows

# model building *********************************************************************************************************
# splitting data to training and testing

tweet_train, tweet_test = train_test_split(tweet_data, test_size = 0.20 )

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer = split_into_words).fit(tweet_data.text)

# Defining BOW for all messages
all_tweet_matrix = tweet_bow.transform(tweet_data.text)

# For training messages
train_tweet_matrix = tweet_bow.transform(tweet_train.text)

# For testing messages
test_tweet_matrix = tweet_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_tweet_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape 

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape 

classifier_mb = MB()                                                     # initialising and fitting multinomial NB
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)

accuracy_test = accuracy_score(test_pred_m, tweet_test.target) 
pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train = accuracy_score(train_pred_m, tweet_train.target) 
pd.crosstab(train_pred_m, tweet_train.target)


classifier_mb_lap = MB(alpha = 3)                                         # laplace transformation
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap


# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap ==tweet_train.target)
accuracy_train_lap

